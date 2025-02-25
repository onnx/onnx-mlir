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

Operation *createOpWithNewType(PatternRewriter &rewriter, Operation *op,
    SmallVector<Value> &operands, llvm::SmallVector<Location> &locs,
    llvm::SmallVector<Type> &types, const FloatType toFloatType) {

  auto fusedLoc = FusedLoc::get(op->getContext(), locs);

  // Requires special treatment for ONNXCastOp as we need to modify its 'to'
  // attribute.
  if (auto onnxCastOp = dyn_cast<mlir::ONNXCastOp>(op)) {
    assert(operands.size() == 1 && "CastOp must have a simple operand");
    auto inputValue = operands.back();
    auto onnxCastTy = cast<mlir::TensorType>(inputValue.getType());
    auto newOnnxCastTy = onnxCastTy.clone(toFloatType);
    return rewriter.create<mlir::ONNXCastOp>(fusedLoc, newOnnxCastTy,
        operands.back(), onnxCastOp.getSaturate(), toFloatType);
  }

  // Start composing new op
  OperationState state(fusedLoc, op->getName().getStringRef(), operands, types,
      op->getAttrs(), op->getSuccessors());

  // Create the new op
  rewriter.setInsertionPoint(op);
  return rewriter.create(state);
}

bool isQuarkGeneratedCastOp(
    Operation *op, const FloatType fromFloatType, const FloatType toFloatType) {
  if (!isa_and_nonnull<mlir::ONNXCastOp>(op)) {
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
  llvm::transform(constantValues.getValues<APFloat>(),
      std::back_inserter(newValues), [&](APFloat fp32Apfloat) -> llvm::APFloat {
        return convertF32ToBf16(
            bf16Semantics, fp32Apfloat.bitcastToAPInt().getZExtValue());
      });
  return mlir::DenseElementsAttr::get(toShapedTy, llvm::ArrayRef(newValues));
}

mlir::Value createNewConstantOp(PatternRewriter &rewriter,
    mlir::ONNXConstantOp constantOp, Location loc, FloatType fromFloatTy,
    FloatType toFloatTy) {
  // Avoid rewriting any const whose element type is not 'FromFPTy'
  ElementsAttr valueAttr = cast<ElementsAttr>(constantOp.getValueAttr());
  if (valueAttr.getElementType() != fromFloatTy) {
    return constantOp;
  }

  if (valueAttr.getElementType() == toFloatTy) {
    return constantOp;
  }

  auto shapedTy = cast<ShapedType>(valueAttr.getType());
  auto toShapedTy = shapedTy.clone(toFloatTy);

  ElementsAttr newValueAttr;

  auto &bf16Semantics = toFloatTy.getFloatSemantics();
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

  if (!onnx_mlir::isDenseONNXConstant(constantOp.getResult())) {
    return rewriter.notifyMatchFailure(
        constantOp->getLoc(), "Constant should have a dense attribute.");
  }
  auto constFusedLocs = FusedLoc::get(constantOp->getContext(), constantLocs);

  auto valueAttr = createNewConstantOp(
      rewriter, constantOp, constFusedLocs, fromFloatTy, toFloatTy);

  // Replace the original const op with the new one
  rewriter.replaceOp(outputCastOp, valueAttr);

  return success();
}

/// Generic pattern to capture operations generated by the Quark quantizer.
/// Operations are captured from an onnx.Cast {to = bf16} and the pattern
/// converts a sequence of onnx.Casts and Ops(F32) to a single Bf16 operation.
/// Example:
///
///    ONNXConstant (ty=f32)
///       \               /
///  onnx.Cast(f32) {to=bf16}
///
///   is converted to:
///
///      ONNXConstant (ty=bf16)
///
class ONNXConstantOpPattern : public OpRewritePattern<mlir::ONNXCastOp> {
public:
  ONNXConstantOpPattern(mlir::MLIRContext *context, PatternBenefit benefit)
      : mlir::OpRewritePattern<mlir::ONNXCastOp>(context, benefit),
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

    llvm::SmallVector<Location> newOpLocations{op->getLoc(), castOp->getLoc()};
    if (!isa<mlir::ONNXConstantOp>(op)) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Not a constant operation");
    }

    auto constOp = dyn_cast<mlir::ONNXConstantOp>(op);
    if (!cast<ShapedType>(constOp.getType()).hasRank()) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Only supports ranked types.");
    }
    return createAndReplaceConstantOp(
        rewriter, castOp, constOp, newOpLocations, fromFloatType, toFloatType);
  }

private:
  const FloatType fromFloatType;
  const FloatType toFloatType;
};

/// Generic pattern to capture operations generated by the Quark quantizer.
/// Operations are captured from an onnx.Cast {to = bf16} and the pattern
/// converts a sequence of onnx.Casts and Ops(F32) to a single Bf16 operation
/// (when such conversion is legal).
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
  GenericPattern(mlir::MLIRContext *context, PatternBenefit benefit)
      : mlir::OpRewritePattern<mlir::ONNXCastOp>(context, benefit),
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

    if (llvm::any_of(llvm::concat<const mlir::Type>(
                         op->getOperandTypes(), op->getResultTypes()),
            [&](const Type ty) { return isa<UnrankedTensorType>(ty); })) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Only supports ranked types.");
    }

    llvm::SmallVector<Location> newOpLocations{op->getLoc(), castOp->getLoc()};

    if (isa<mlir::ONNXConstantOp>(op)) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "ONNXConstantOp has its own pattern.");
    }

    // Track operations to be erase if verifier fails to validate
    // legalization. This would happen when the target datatype (BF16) used
    // isn't compatible with the new op.
    SmallVector<Operation *> createdOpsTrackerForFailedVerifier;
    auto removeTrackedOps = [&]() {
      llvm::for_each(createdOpsTrackerForFailedVerifier,
          [&](auto *createdOp) { createdOp->erase(); });
    };

    // Operands of this new op can be: NoValue's, constants, the operand of an
    // input cast, or the current op's operand.
    SmallVector<Value> operands;
    llvm::transform(
        op->getOperands(), std::back_inserter(operands), [&](Value operand) {
          auto *operandOp = operand.getDefiningOp();
          if (!operandOp)
            return operand;

          if (onnx_mlir::isNoneValue(operand))
            return operand;

          if (auto constOp = dyn_cast<mlir::ONNXConstantOp>(operandOp)) {
            auto constValue = createNewConstantOp(rewriter, constOp,
                constOp->getLoc(), fromFloatType, toFloatType);

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

    if (llvm::any_of(operands, [&](auto operand) {
          auto tensorType = dyn_cast<TensorType>(operand.getType());
          return tensorType && tensorType.getElementType() == fromFloatType;
        })) {
      // Make sure to erase any additional op that was created so that the
      // rewriter doesn't trigger any changes by this pass.
      removeTrackedOps();
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Missing input ONNXCastOp(from=bf16 to=f32)");
    }

    SmallVector<Type> resultTypes;
    llvm::SmallDenseMap<Value, int64_t> opResultsMap;

    // We now need to create the output types, and map the values of each result
    // to the op's result number. This will allow us to replace multiple
    // ONNXCast ops that are used by multiple results to each of the result of
    // the new operation.
    for (const auto &result : op->getOpResults()) {
      if (onnx_mlir::isNoneValue(result)) {
        resultTypes.push_back(result.getType());
        continue;
      }

      SmallVector<OpOperand *> opResultUsers = llvm::to_vector(llvm::map_range(
          result.getUses(), [](OpOperand &use) { return &use; }));

      llvm::for_each(opResultUsers, [&](mlir::OpOperand *user) {
        if (isOutputCastOp(user->getOwner(), fromFloatType, toFloatType)) {
          opResultsMap[user->getOwner()->getResult(0)] =
              result.getResultNumber();
        } else {
          opResultsMap[result] = result.getResultNumber();
        }
      });

      size_t numOutputCastOps =
          llvm::count_if(opResultUsers, [&](mlir::OpOperand *user) {
            return isOutputCastOp(user->getOwner(), fromFloatType, toFloatType);
          });

      if (numOutputCastOps == 0) {
        resultTypes.push_back(result.getType());
        continue;
      }

      // Sanity check so that the every result that connected to an ONNXCast
      // isn't always connected to anything else.
      if (numOutputCastOps != opResultUsers.size()) {
        // Make sure to erase any additional op that was created so that the
        // rewriter doesn't trigger any changes by this pass.
        removeTrackedOps();
        return rewriter.notifyMatchFailure(
            castOp->getLoc(), "Missing input ONNXCastOp(from=bf16 to=f32)");
      }

      auto resultType = cast<TensorType>(result.getType());
      resultTypes.push_back(resultType.clone(toFloatType));
    }

    // Try to create the ONNX operation with the new type.
    Operation *newOp = createOpWithNewType(
        rewriter, op, operands, newOpLocations, resultTypes, toFloatType);

    // MLIR has validators for operations that fail if the types are
    // unsupported. We make use of this here by seeing if the type
    // replacement in 'newOp' is acceptable. If it isn't, we fallback to the
    // casting solution. Note that IgnoreDiagnostic is there to absorb
    // diagnostics that would be produced in the cases 'newOp' is not
    // acceptable. Since we will not use the 'newOp, the diagnostic is
    // irrelevant.
    onnx_mlir::IgnoreDiagnostic diag(op->getContext()->getDiagEngine());
    bool isNewOpValid;
    if (auto info = newOp->getName().getRegisteredInfo()) {
      isNewOpValid = succeeded(info->verifyInvariants(newOp));
    } else {
      isNewOpValid = succeeded(mlir::verify(newOp));
    }

    if (!isNewOpValid) {
      // Make sure to erase any additional op that was created so that the
      // rewriter doesn't trigger any changes by this pass.
      removeTrackedOps();
      newOp->erase();

      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "cannot create operation with this type");
    }

    for (auto [value, resultNumber] : opResultsMap) {
      rewriter.replaceAllUsesWith(value, newOp->getResult(resultNumber));
    }
    return success();
  }

private:
  const FloatType fromFloatType;
  const FloatType toFloatType;
};

/// Cast To Cast pattern that canonicalizes to a single Cast operation.
/// Example:
///     onnx.Cast(bf16){to=f32}
///              |
///     onnx.Cast(f32){to=<Other ElmTy>}
///              |
///     Op(f32) {to=bf16}
///
///   is converted to:
///
///     onnx.Cast(bf16){to=<Other ElmTy>}
///
class CastToCastPattern : public OpRewritePattern<mlir::ONNXCastOp> {
public:
  CastToCastPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::ONNXCastOp>(context),
        fromFloatType(FloatType::getF32(context)),
        toFloatType(FloatType::getBF16(context)) {}

  LogicalResult matchAndRewrite(
      mlir::ONNXCastOp castOp, PatternRewriter &rewriter) const override {

    auto inputCastOp =
        dyn_cast_or_null<mlir::ONNXCastOp>(castOp.getInput().getDefiningOp());
    if (!inputCastOp) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Input isn't a ONNXCastOp");
    }

    if (!inputCastOp->getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "Input ONNXCastOp doesn't have single use");
    }

    auto inputCastOperandTy =
        cast<TensorType>(inputCastOp.getInput().getType());
    auto inputCastTy = cast<TensorType>(inputCastOp.getType());
    if (inputCastOperandTy.getElementType() != toFloatType ||
        inputCastTy.getElementType() != fromFloatType) {
      return rewriter.notifyMatchFailure(castOp->getLoc(),
          "Can only canonicalize ONNXCastOp(from=bf16 "
          "to=f32) -> ONNXCastOp(from=f32 to=SomeType)");
    }

    auto castTy = cast<TensorType>(castOp.getType());
    if (inputCastOperandTy.getElementType() == castOp.getTo()) {
      return rewriter.notifyMatchFailure(castOp->getLoc(),
          "Cannotcanonicalize ONNXCastOp(from=bf16 "
          "to=f32) -> ONNXCastOp(from=f32 to=bf16)");
    }

    auto newOnnxCastTy = inputCastOperandTy.clone(castTy.getElementType());
    auto newONNXCastOp = rewriter.create<mlir::ONNXCastOp>(
        FusedLoc::get(
            castOp->getContext(), {inputCastOp->getLoc(), castOp->getLoc()}),
        newOnnxCastTy, inputCastOp.getInput(), inputCastOp.getSaturate(),
        castOp.getTo());
    rewriter.replaceOp(castOp, newONNXCastOp);

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

    RewritePatternSet greedyPatterns(&ctx);

    onnx_mlir::getLegalizeQuarkQuantizedOpsPatterns(greedyPatterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(greedyPatterns))))
      return signalPassFailure();
  }
};

} // namespace

void onnx_mlir::getLegalizeQuarkQuantizedOpsPatterns(
    RewritePatternSet &patterns) {
  PatternBenefit highPriority(1000);
  patterns.insert<GenericPattern>(patterns.getContext(), highPriority);
  patterns.insert<ONNXConstantOpPattern>(patterns.getContext(), highPriority);
  patterns.insert<CastToCastPattern>(patterns.getContext());
}

/// Factory function for creating the bf16-to-f32 pass object
std::unique_ptr<mlir::Pass> onnx_mlir::createLegalizeQuarkQuantizedOpsPass() {
  return std::make_unique<LegalizeOpsWithCasttoFloatType>();
}
