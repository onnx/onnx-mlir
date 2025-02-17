// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

enum class FPConversion {
  UnknownType,
  F32,
  F16,
  BF16,
  F8E4M3FN,
  F8E4M3B11FNUZ,
  F8E4M3FNUZ,
  F8E5M2,
  F8E5M2FNUZ
};

class IgnoreDiagnostic {
public:
  IgnoreDiagnostic(DiagnosticEngine &diagEngine) : diagEngine(diagEngine) {
    id = diagEngine.registerHandler(
        [](mlir::Diagnostic & /*diag*/) { return success(); });
  }

  ~IgnoreDiagnostic() {
    // Reset to the previous state.
    diagEngine.eraseHandler(id);
  }

private:
  DiagnosticEngine &diagEngine;
  DiagnosticEngine::HandlerID id;
};

Operation *createOpWithNewType(PatternRewriter &rewriter, Operation *op,
    SmallVector<Value> &operands, llvm::SmallVector<Type> &types) {
  // adapted from
  // third-party/llvm-project/mlir/lib/Conversion/MemRefToSPIRV/MapMemRefStorageClassPass.cpp

  // Start composing new op
  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
      types, op->getAttrs(), op->getSuccessors());

  // Create the new op
  return rewriter.create(state);
}

bool isQuarkGeneratedCastOp(
    Operation *op, const FloatType fromType, const FloatType toType) {
  if (!isa<mlir::ONNXCastOp>(op)) {
    return false;
  }
  auto castOp = cast<mlir::ONNXCastOp>(op);
  return castOp.getTo() == toType &&
         cast<TensorType>(castOp.getInput().getType()).getElementType() ==
             fromType;
}

bool isInputCastOp(
    Operation *op, const FloatType fromType, const FloatType toType) {
  return isQuarkGeneratedCastOp(op, toType, fromType);
}

bool isOutputCastOp(
    Operation *op, const FloatType fromType, const FloatType toType) {
  return isQuarkGeneratedCastOp(op, fromType, toType);
}

llvm::APFloat convert(llvm::APFloat apFloat, FloatType toType) {
  bool ignored;
  apFloat.convert(
      toType.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &ignored);
  return apFloat;
}

mlir::DenseElementsAttr getDenseElementAttrFromConstOp(Operation *constOp,
    FloatType dstFloatType, mlir::ShapedType dstShapedTy,
    DenseElementsAttr denseValueAttr) {
  std::vector<APFloat> newValues;
  llvm::transform(denseValueAttr.template getValues<llvm::APFloat>(),
      std::back_inserter(newValues),
      [&](llvm::APFloat apFloat) -> llvm::APFloat {
        return convert(apFloat, dstFloatType);
      });
  // Create a new const op with the same data and shape but 'ToFPTy'
  // element type
  return mlir::DenseElementsAttr::get(dstShapedTy, llvm::ArrayRef(newValues));
}

LogicalResult createNewConstantOp(PatternRewriter &rewriter,
    mlir::ONNXConstantOp constantOp, FloatType dstFloatTy,
    ShapedType dstShapedTy) {

  auto sparseValue =
      dyn_cast_or_null<ElementsAttr>(constantOp.getSparseValueAttr());

  // Avoid rewriting any const whose element type is not 'FromFPTy'
  ElementsAttr valueAttr;
  if (sparseValue) {
    return rewriter.notifyMatchFailure(
        constantOp->getLoc(), "operation doesn't need type conversion");
  }
  valueAttr = dyn_cast<ElementsAttr>(constantOp.getValueAttr());

  if (valueAttr.getType() == dstShapedTy) {
    return rewriter.notifyMatchFailure(
        constantOp->getLoc(), "operation doesn't need type conversion");
  }

  // Avoid rewriting non-dense constants (for now)
  auto denseValueAttr = dyn_cast<DenseElementsAttr>(valueAttr);
  if (!denseValueAttr) {
    return rewriter.notifyMatchFailure(
        constantOp->getLoc(), "not a dense value");
  }

  auto newDenseValueAttrTy = denseValueAttr.getType().clone(dstFloatTy);
  auto newValueAttr = getDenseElementAttrFromConstOp(
      constantOp, dstFloatTy, newDenseValueAttrTy, denseValueAttr);

  // Replace the original const op with the new one
  rewriter.replaceOpWithNewOp<mlir::ONNXConstantOp>(constantOp, dstShapedTy,
      Attribute(), newValueAttr, mlir::FloatAttr(), mlir::ArrayAttr(),
      mlir::IntegerAttr(), mlir::ArrayAttr(), mlir::StringAttr(),
      mlir::ArrayAttr());

  return success();
}

/// Generic pattern to capture operations generated by the Quark quantizer.
/// Operations are captured from an onnx.Cast {to = bf16} and the pattern
/// converts a sequence of onnx.Casts and Ops(F32) to a single Bf16 operation.
/// Example:
///
///  onnx.Cast(bf16)   onnx.Cast(bf16)  {to=f32}
///          \          /
///        onnx.Add(f32,f32)
///              |
///        onnx.Cast(f32) {to=bf16}
///
///   is converted to:
///
///      onnx.Add(bf16, bf16)
///
class GenericPattern : public OpRewritePattern<mlir::ONNXCastOp> {
public:
  GenericPattern(const FloatType fromType, const FloatType toType,
      mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::ONNXCastOp>(context), fromType(fromType),
        toType(toType) {}

  LogicalResult matchAndRewrite(
      mlir::ONNXCastOp castOp, PatternRewriter &rewriter) const override {
    auto castInput = castOp.getInput();
    if (!isOutputCastOp(castOp, fromType, toType)) {
      return rewriter.notifyMatchFailure(castOp->getLoc(),
          "Conversion should start from an "
          "ONNXCastOp(from=f32 to=bf16)");
    }

    auto *op = castInput.getDefiningOp();
    if (!op) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Block Argument found.");
    }

    auto shapedTy = cast<ShapedType>(op->getResult(0).getType());
    auto dstShapedTy = shapedTy.clone(toType);

    if (auto constOp = dyn_cast<mlir::ONNXConstantOp>(op)) {
      return createNewConstantOp(rewriter, constOp, toType, dstShapedTy);
    }

    SmallVector<Value> operands;
    llvm::transform(
        op->getOperands(), std::back_inserter(operands), [&](Value operand) {
          auto *operandOp = operand.getDefiningOp();
          if (!operandOp)
            return operand;

          if (onnx_mlir::isNoneValue(operand)) {
            return operand;
          }
          if (auto constOp = dyn_cast<mlir::ONNXConstantOp>(operandOp)) {
            auto newValueAttr = constOp.getValueAttr();
            Value constValue = rewriter.create<ONNXConstantOp>(
                operandOp->getLoc(), mlir::Attribute(), newValueAttr);
            return constValue;
          }
          operandOp->dump();
          return operandOp->getOperand(0);
        });

    SmallVector<Type> resultTypes;
    llvm::SmallDenseMap<Operation *, int64_t> resultOps;
    llvm::transform(op->getOpResults(), std::back_inserter(resultTypes),
        [&](OpResult result) -> Type {
          auto *opResult = result.getOwner();

          llvm::for_each(opResult->getUsers(), [&](Operation *userOp) {
            resultOps[userOp] = result.getResultNumber();
          });
          if (llvm::none_of(opResult->getUsers(), [&](Operation *userOp) {
                resultOps[userOp] = result.getResultNumber();
                return isOutputCastOp(userOp, fromType, toType);
              })) {
            return result.getType();
          }
          assert(llvm::all_of(opResult->getUsers(), [&](Operation *userOp) {
            return isOutputCastOp(userOp, fromType, toType);
          }));
          auto resultType = cast<TensorType>(result.getType());
          return resultType.clone(toType);
        });

    // TODO: FXML-4779 - allow operations with regions to be converted.
    bool hasNoRegion = op->getNumRegions() == 0;
    bool isNewTypeCompatible = hasNoRegion;

    Operation *newOp = nullptr;
    if (hasNoRegion) {
      // Try to create the ONNX operation with the new type.
      newOp = createOpWithNewType(rewriter, op, operands, resultTypes);

      // MLIR has validators for operations that fail if the types are
      // unsupported. We make use of this here by seeing if the type
      // replacement in 'newOp' is acceptable. If it isn't, we fallback to the
      // casting solution. Note that IgnoreDiagnostic is there to absorb
      // diagnostics that would be produced in the cases 'newOp' is not
      // acceptable. Since we will not use the 'newOp, the diagnostic is
      // irrelevant.
      IgnoreDiagnostic diag(op->getContext()->getDiagEngine());
      isNewTypeCompatible &= succeeded(mlir::verify(newOp));
    }

    if (!isNewTypeCompatible) {
      newOp->erase();
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "cannot create operation with this type");
    }

    for (auto [op, resultNumber] : resultOps) {
      if (op == castOp)
        continue;
      rewriter.replaceAllOpUsesWith(op, newOp->getResult(resultNumber));
    }
    rewriter.replaceOp(castOp, newOp->getResult(resultOps[castOp]));
    return success();
  }

private:
  const FloatType fromType;
  const FloatType toType;
};

/// Pass for converting all 'FromFPTy' values to 'ToFPTy' in a FuncOp
class LegalizeOpsWithCastToType : public PassWrapper<LegalizeOpsWithCastToType,
                                      OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeOpsWithCastToType)

  LegalizeOpsWithCastToType() = default;
  LegalizeOpsWithCastToType(const LegalizeOpsWithCastToType &pass)
      : mlir::PassWrapper<LegalizeOpsWithCastToType,
            OperationPass<func::FuncOp>>() {}

  Option<FPConversion> fromType{*this, "from-type",
      ::llvm::cl::desc("Source element type that needs to be converted"),
      ::llvm::cl::init(FPConversion::F32),
      ::llvm::cl::values(
          clEnumValN(FPConversion::UnknownType, "", "Unknown type"),
          clEnumValN(FPConversion::F32, "f32", "Convert from F32"),
          clEnumValN(FPConversion::BF16, "bf16", "Convert from BF16"))};

  ::mlir::Pass::Option<FPConversion> toType{*this, "to-type",
      ::llvm::cl::desc("Destination element type to legalize operations "
                       "surrounded by casts."),
      ::llvm::cl::init(FPConversion::BF16),
      ::llvm::cl::values(
          clEnumValN(FPConversion::UnknownType, "", "Unknown type"),
          clEnumValN(FPConversion::F32, "f32", "Convert to F32"),
          clEnumValN(FPConversion::BF16, "bf16", "Convert to BF16"))};

  StringRef getArgument() const override {
    return "legalize-quark-quantized-ops";
  }

  static inline FloatType getFPType(MLIRContext *ctx, FPConversion type) {
    switch (type) {
    case FPConversion::BF16:
      return FloatType::getBF16(ctx);
    case FPConversion::F32:
      return FloatType::getF32(ctx);
    default:
      llvm_unreachable("unknown type");
    }
  }

  void runOnOperation() override {

    auto &ctx = this->getContext();
    Operation *op = this->getOperation();

    // auto module = op->getParentOfType<ModuleOp>();
    // if (!module->hasAttr("producer.name")) {
    //   return;
    // }

    op->dump();
    RewritePatternSet greedyPatterns(&ctx);

    // Some patterns above may add some reshape, match them
    greedyPatterns.insert<GenericPattern>(
        getFPType(&ctx, this->fromType), getFPType(&ctx, this->toType), &ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(applyPatternsAndFoldGreedily(
            op, std::move(greedyPatterns), config)))
      return signalPassFailure();
  }
};

} // namespace

/// Factory function for creating the bf16-to-f32 pass object
std::unique_ptr<mlir::Pass> onnx_mlir::createLegalizeQuarkQuantizedOpsPass() {
  return std::make_unique<LegalizeOpsWithCastToType>();
}
