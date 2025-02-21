// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrHelper.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/LegalizeQuarkQuantizedOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

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
    SmallVector<Value> &operands, llvm::SmallVector<Location> &locs,
    llvm::SmallVector<Type> &types, const FloatType toFloatType) {

  auto fusedLoc = FusedLoc::get(op->getContext(), locs);

  // Requires special treatment for ONNXCastOp.
  if (auto onnxCastOp = dyn_cast<mlir::ONNXCastOp>(op)) {
    // llvm::errs() << "here\n";
    assert(operands.size() == 1 && "CastOp must have a simple operand");
    auto inputValue = operands.back();
    auto onnxCastTy = cast<mlir::TensorType>(inputValue.getType());
    auto newOnnxCastTy = onnxCastTy.clone(toFloatType);
    return rewriter.create<mlir::ONNXCastOp>(fusedLoc, newOnnxCastTy,
        operands.back(), onnxCastOp.getSaturate(), toFloatType);
  }

  // adapted from
  // third-party/llvm-project/mlir/lib/Conversion/MemRefToSPIRV/MapMemRefStorageClassPass.cpp

  // Start composing new op
  OperationState state(fusedLoc, op->getName().getStringRef(), operands, types,
      op->getAttrs(), op->getSuccessors());

  // Create the new op
  return rewriter.create(state);
}

bool isQuarkGeneratedCastOp(
    Operation *op, const FloatType fromFloatType, const FloatType toFloatType) {
  if (!isa<mlir::ONNXCastOp>(op)) {
    return false;
  }
  auto castOp = cast<mlir::ONNXCastOp>(op);
  return castOp.getTo() == toFloatType &&
         cast<TensorType>(castOp.getInput().getType()).getElementType() ==
             fromFloatType;
}

bool isInputCastOp(
    Operation *op, const FloatType fromFloatType, const FloatType toFloatType) {
  return isQuarkGeneratedCastOp(op, toFloatType, fromFloatType);
}

bool isOutputCastOp(
    Operation *op, const FloatType fromFloatType, const FloatType toFloatType) {
  return isQuarkGeneratedCastOp(op, fromFloatType, toFloatType);
}

llvm::APFloat convertF32ToBf16(
    const llvm::fltSemantics &bf16Semantics, uint32_t f32Value) {
  return llvm::APFloat(bf16Semantics,
      llvm::APInt(sizeof(uint16_t) * 8, static_cast<uint16_t>(f32Value >> 16)));
}

mlir::DenseElementsAttr getDenseElementAttrFromConstOp(
    const llvm::fltSemantics &bf16Semantics, ShapedType toShapedTy,
    DenseElementsAttr &constantValues) {
  std::vector<APFloat> newValues;
  // constantValues.dump();
  llvm::transform(constantValues.getValues<APFloat>(),
      std::back_inserter(newValues), [&](APFloat fp32Apfloat) -> llvm::APFloat {
        return convertF32ToBf16(
            bf16Semantics, fp32Apfloat.bitcastToAPInt().getZExtValue());
      });
  // Create a new const op with the same data and shape but 'ToFPTy'
  // element type
  // toShapedTy.dump();
  return mlir::DenseElementsAttr::get(toShapedTy, llvm::ArrayRef(newValues));
}

mlir::Value createNewConstantOp(PatternRewriter &rewriter,
    mlir::ONNXConstantOp constantOp, Location loc, FloatType fromFloatTy,
    FloatType toFloatTy, ShapedType toShapedTy) {

  // Avoid rewriting any const whose element type is not 'FromFPTy'
  ElementsAttr valueAttr = cast<ElementsAttr>(constantOp.getValueAttr());

  if (valueAttr.getElementType() != fromFloatTy) {
    return constantOp;
  }

  if (valueAttr.getElementType() == toFloatTy) {
    llvm::errs() << "Here I am\n";
    return constantOp;
  }

  ElementsAttr newValueAttr;

  auto &bf16Semantics = toFloatTy.getFloatSemantics();
  // toFloatTy.dump();

  if (valueAttr.isSplat()) {
    auto f32Value = valueAttr.getSplatValue<llvm::APFloat>();
    newValueAttr = mlir::DenseElementsAttr::get(
        toShapedTy, convertF32ToBf16(bf16Semantics,
                        f32Value.bitcastToAPInt().getZExtValue()));
  } else {
    onnx_mlir::OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    DenseElementsAttr valueElements =
        elementsBuilder.toDenseElementsAttr(valueAttr);
    newValueAttr = getDenseElementAttrFromConstOp(
        bf16Semantics, toShapedTy, valueElements);
  }

  // toShapedTy.dump();
  auto constantValue = rewriter.create<mlir::ONNXConstantOp>(loc, toShapedTy,
      Attribute(), newValueAttr, mlir::FloatAttr(), mlir::ArrayAttr(),
      mlir::IntegerAttr(), mlir::ArrayAttr(), mlir::StringAttr(),
      mlir::ArrayAttr());
  return constantValue;
}

LogicalResult createAndReplaceConstantOp(PatternRewriter &rewriter,
    mlir::ONNXCastOp outputCastOp, mlir::ONNXConstantOp constantOp,
    llvm::SmallVector<Location> &constantLocs, FloatType fromFloatTy,
    FloatType toFloatTy) {

  auto shapedTy = cast<ShapedType>(constantOp->getResult(0).getType());
  if (shapedTy.getElementType() != fromFloatTy) {
    return rewriter.notifyMatchFailure(
        constantOp->getLoc(), "operation doesn't need type conversion");
  }

  auto toShapedTy = shapedTy.clone(toFloatTy);
  if (shapedTy == toShapedTy) {
    return rewriter.notifyMatchFailure(
        constantOp->getLoc(), "operation doesn't need type conversion");
  }

  if (!onnx_mlir::isDenseONNXConstant(constantOp.getResult())) {
    return rewriter.notifyMatchFailure(
        constantOp->getLoc(), "Constant should have a dense attribute.");
  }

  auto constFusedLocs = FusedLoc::get(constantOp->getContext(), constantLocs);

  auto valueAttr = createNewConstantOp(
      rewriter, constantOp, constFusedLocs, fromFloatTy, toFloatTy, toShapedTy);

  // Replace the original const op with the new one
  rewriter.replaceOp(outputCastOp, valueAttr);

  return success();
}

/// Generic pattern to capture operations generated by the Quark quantizer.
/// Operations are captured from an onnx.Cast {to = bf16} and the pattern
/// converts a sequence of onnx.Casts and Ops(F32) to a single Bf16 operation.
/// Example:
///
///    OpA (ty=bf16)    OpB (ty=bf16)
///       \               /
///  onnx.Cast(bf16)   onnx.Cast(bf16)  {to=f32}
///          \          /
///        onnx.Add(f32,f32)
///              |
///        onnx.Cast(f32) {to=bf16}
///
///   is converted to:
///
///      onnx.Add(OpA, OpB)
///
class GenericPattern : public OpRewritePattern<mlir::ONNXCastOp> {
public:
  GenericPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::ONNXCastOp>(context),
        fromFloatType(FloatType::getF32(context)),
        toFloatType(FloatType::getBF16(context)) {}

  LogicalResult matchAndRewrite(
      mlir::ONNXCastOp castOp, PatternRewriter &rewriter) const override {
    if (!isOutputCastOp(castOp, fromFloatType, toFloatType)) {
      return rewriter.notifyMatchFailure(castOp->getLoc(),
          "Conversion should start from an "
          "ONNXCastOp(from=f32 to=bf16)");
    }

    auto *op = castOp.getInput().getDefiningOp();
    if (!op) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Block Argument found.");
    }

    if (op->getNumRegions() > 0) {
      return rewriter.notifyMatchFailure(castOp->getLoc(),
          "Does not support conversion of operations with region.");
    }

    if (!isa<mlir::ONNXDialect>(op->getDialect())) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Only supports onnx-mlir operations.");
    }

    if (llvm::any_of(llvm::concat<const mlir::Type>(
                         op->getOperandTypes(), op->getResultTypes()),
            [&](const Type ty) { return isa<UnrankedTensorType>(ty); })) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Only supports ranked types.");
    }

    llvm::SmallVector<Location> newOpLocations{op->getLoc(), castOp->getLoc()};

    if (auto constOp = dyn_cast<mlir::ONNXConstantOp>(op)) {
      return createAndReplaceConstantOp(rewriter, castOp, constOp,
          newOpLocations, fromFloatType, toFloatType);
    }

    // Track operations to be erase if verifier fails to validate legalization.
    // This would happen when the target datatype (BF16) used isn't compatible
    // with the new op.
    SmallVector<Operation *> createdOpsTrackerForFailedVerifier;

    SmallVector<Value> operands;
    llvm::transform(
        op->getOperands(), std::back_inserter(operands), [&](Value operand) {
          // llvm::errs() << "Operand value";
          // operand.dump();
          auto *operandOp = operand.getDefiningOp();
          if (!operandOp)
            return operand;

          if (onnx_mlir::isNoneValue(operand))
            return operand;

          if (auto constOp = dyn_cast<mlir::ONNXConstantOp>(operandOp)) {
            auto shapedTy = cast<ShapedType>(constOp->getResult(0).getType());
            auto toShapedTy = shapedTy.clone(toFloatType);

            auto constValue = createNewConstantOp(rewriter, constOp,
                constOp->getLoc(), fromFloatType, toFloatType, toShapedTy);

            createdOpsTrackerForFailedVerifier.push_back(
                constValue.getDefiningOp());
            return constValue;
          }

          if (!isInputCastOp(operandOp, fromFloatType, toFloatType)) {
            return operand;
          }

          newOpLocations.push_back(operandOp->getLoc());
          return cast<mlir::ONNXCastOp>(operandOp).getInput();
        });

    if (llvm::any_of(llvm::concat<const mlir::Type>(
                         op->getOperandTypes(), op->getResultTypes()),
            [&](const Type ty) {
              auto unrankedType = dyn_cast<UnrankedTensorType>(ty);
              return unrankedType &&
                     unrankedType.getElementType() == fromFloatType;
            })) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Only supports ranked types.");
    }
    // llvm::errs() << "here2\n";
    if (llvm::any_of(operands, [&](auto operand) {
          auto tensorType = dyn_cast<TensorType>(operand.getType());
          return tensorType && tensorType.getElementType() == fromFloatType;
        })) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Missing input ONNXCastOp(from=bf16 to=f32)");
    }
    // llvm::errs() << "here3\n";
    SmallVector<Type> resultTypes;
    llvm::SmallDenseMap<Operation *, int64_t> resultOps;
    llvm::transform(op->getOpResults(), std::back_inserter(resultTypes),
        [&](OpResult result) -> Type {
          auto *opResult = result.getOwner();
          if (onnx_mlir::isNoneValue(result)) {
            return result.getType();
          }

          llvm::for_each(opResult->getUsers(), [&](Operation *userOp) {
            resultOps[userOp] = result.getResultNumber();
          });
          if (llvm::none_of(opResult->getUsers(), [&](Operation *userOp) {
                resultOps[userOp] = result.getResultNumber();
                return isOutputCastOp(userOp, fromFloatType, toFloatType);
              })) {
            return result.getType();
          }
          assert(llvm::all_of(opResult->getUsers(), [&](Operation *userOp) {
            return isOutputCastOp(userOp, fromFloatType, toFloatType);
          }));
          auto resultType = cast<TensorType>(result.getType());
          return resultType.clone(toFloatType);
        });

    // Try to create the ONNX operation with the new type.
    Operation *newOp = createOpWithNewType(
        rewriter, op, operands, newOpLocations, resultTypes, toFloatType);
    // llvm::errs() << "here6\n";
    // MLIR has validators for operations that fail if the types are
    // unsupported. We make use of this here by seeing if the type
    // replacement in 'newOp' is acceptable. If it isn't, we fallback to the
    // casting solution. Note that IgnoreDiagnostic is there to absorb
    // diagnostics that would be produced in the cases 'newOp' is not
    // acceptable. Since we will not use the 'newOp, the diagnostic is
    // irrelevant.
    IgnoreDiagnostic diag(op->getContext()->getDiagEngine());
    bool isNewTypeCompatible;
    if (auto info = newOp->getName().getRegisteredInfo()) {
      // llvm::errs() << "here17\n";
      isNewTypeCompatible = succeeded(info->verifyInvariants(newOp));
    } else {
      // llvm::errs() << "here18\n";
      isNewTypeCompatible = succeeded(mlir::verify(newOp));
    }
    if (isa<mlir::ONNXCastOp>(op)) {
      mlir::OpPrintingFlags flags;
      flags.elideLargeElementsAttrs(16);
      op->getParentOp()->print(llvm::errs(), flags);
    }
    // llvm::errs() << "here19\n";
    // mlir::OpPrintingFlags flags;
    // flags.elideLargeElementsAttrs(16);
    if (!isNewTypeCompatible) {
      // Make sure to erase any additional op that was created so that the
      // rewriter doesn't trigger any changes by this pass.
      llvm::for_each(createdOpsTrackerForFailedVerifier,
          [&](auto *createdOp) { createdOp->erase(); });
      newOp->erase();
      // llvm::errs() << "here10\n";
      // castOp.dump();
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "cannot create operation with this type");
    }
    // flags.elideLargeElementsAttrs(16);
    // llvm::errs() << "Starting\n";
    // op->getParentOp()->print(llvm::errs(), flags);
    // castOp.dump();
    // newOp->getResult(resultOps[castOp]).dump();

    for (auto [op, resultNumber] : resultOps) {
      rewriter.replaceAllOpUsesWith(op, newOp->getResult(resultNumber));
    }

    return success();
  }

private:
  const FloatType fromFloatType;
  const FloatType toFloatType;
};

/// Pass for converting all 'FromFPTy' values to 'ToFPTy' in a FuncOp
class LegalizeOpsWithCasttoFloatType
    : public PassWrapper<LegalizeOpsWithCasttoFloatType,
          OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeOpsWithCasttoFloatType)

  LegalizeOpsWithCasttoFloatType() = default;
  LegalizeOpsWithCasttoFloatType(const LegalizeOpsWithCasttoFloatType &pass)
      : mlir::PassWrapper<LegalizeOpsWithCasttoFloatType,
            OperationPass<func::FuncOp>>() {}

  StringRef getArgument() const override {
    return "legalize-quark-quantized-ops";
  }

  void runOnOperation() override {

    auto &ctx = this->getContext();
    Operation *op = this->getOperation();

    // mlir::OpPrintingFlags flags;
    // flags.elideLargeElementsAttrs(16);
    // llvm::errs() << "Starting\n";
    // op->print(llvm::errs(), flags);
    RewritePatternSet greedyPatterns(&ctx);

    onnx_mlir::getLegalizeQuarkQuantizedOpsPatterns(greedyPatterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(greedyPatterns))))
      return signalPassFailure();
    // llvm::errs() << "Finishing\n";
    // op->print(llvm::errs(), flags);
  }
};

} // namespace

void onnx_mlir::getLegalizeQuarkQuantizedOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<GenericPattern>(patterns.getContext());
}

/// Factory function for creating the bf16-to-f32 pass object
std::unique_ptr<mlir::Pass> onnx_mlir::createLegalizeQuarkQuantizedOpsPass() {
  return std::make_unique<LegalizeOpsWithCasttoFloatType>();
}
