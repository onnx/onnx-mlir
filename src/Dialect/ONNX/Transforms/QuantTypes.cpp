// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights reserved.

#include <memory>

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

namespace onnx_mlir {

/* NOTE:
 * This conversion pass does not touch types where Q/DQ's are missing.
 * There is no insertion of "fake" quantization nodes.
 * This may create ops taking quantized operands and producing fp32 results
 * or vice-versa
 */

template <typename QdqOp>
class QuantTypesFrom : public mlir::OpRewritePattern<QdqOp> {
  using mlir::OpRewritePattern<QdqOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      QdqOp op, mlir::PatternRewriter &rewriter) const override {

    // Should not convert when input comes from blockArg or Cast node
    // TODO: Fix for cast node by either of:
    //  - Updating the cast to quantized type
    //  - Add qcast after cast
    mlir::Value input = op->getOperand(0);
    mlir::Operation *inputOp = input.getDefiningOp();
    if (inputOp == nullptr || mlir::isa<mlir::ONNXCastOp>(inputOp))
      return rewriter.notifyMatchFailure(
          op, "Not converting blockArg or Cast output");

    // Should not convert types when the output is used by 'return' op
    mlir::Value result = op->getResult(0);
    if (llvm::any_of(result.getUsers(), [](mlir::Operation *op) {
          return mlir::isa<mlir::func::ReturnOp>(op);
        }))
      return rewriter.notifyMatchFailure(
          op, "Not converting return value from function");

    auto scaleOp = mlir::dyn_cast_if_present<mlir::ONNXConstantOp>(
        op->getOperand(1).getDefiningOp());
    auto zeropointOp = mlir::dyn_cast_if_present<mlir::ONNXConstantOp>(
        op->getOperand(2).getDefiningOp());
    if (!scaleOp || !zeropointOp)
      return rewriter.notifyMatchFailure(op, "Scale/Zeropoint not constant");

    auto scale = mlir::dyn_cast_if_present<mlir::DenseIntOrFPElementsAttr>(
        scaleOp.getValueAttr());
    auto zeropoint = mlir::dyn_cast_if_present<mlir::DenseIntOrFPElementsAttr>(
        zeropointOp.getValueAttr());
    if (!scale || !zeropoint)
      return rewriter.notifyMatchFailure(
          op, "Scale/Zeropoint not DenseElementsAttr");

    // TODO: Add support for per-channel quantization
    if (scale.getNumElements() != 1 || zeropoint.getNumElements() != 1)
      return rewriter.notifyMatchFailure(op, "Scale/Zeropoint not scalar");

    auto inType = mlir::cast<mlir::TensorType>(input.getType());
    mlir::Type inElemType = inType.getElementType();
    auto outType = mlir::cast<mlir::TensorType>(result.getType());
    mlir::Type outElemType = outType.getElementType();
    unsigned flags = outElemType.isSignedInteger();

    mlir::Type storageType;
    mlir::Type expressedType;

    if constexpr (std::is_same_v<QdqOp, mlir::ONNXDequantizeLinearOp>) {
      storageType = inElemType;
      expressedType = outElemType;
    } else if constexpr (std::is_same_v<QdqOp, mlir::ONNXQuantizeLinearOp>) {
      storageType = outElemType;
      expressedType = inElemType;
    }

    // Get existing quantized type if available
    mlir::quant::QuantizedType oldQuantType;
    if (oldQuantType = mlir::dyn_cast<mlir::quant::QuantizedType>(storageType);
        oldQuantType)
      storageType = oldQuantType.getStorageType();

    // Create quantized type
    mlir::Type quantType = mlir::quant::UniformQuantizedType::get(flags,
        storageType, expressedType,
        scale.template getSplatValue<mlir::APFloat>().convertToDouble(),
        storageType.isSignedInteger()
            ? zeropoint.template getSplatValue<mlir::APInt>().getSExtValue()
            : zeropoint.template getSplatValue<mlir::APInt>().getZExtValue(),
        mlir::quant::QuantizedType::getDefaultMinimumForInteger(
            storageType.isSignedInteger(), storageType.getIntOrFloatBitWidth()),
        mlir::quant::QuantizedType::getDefaultMaximumForInteger(
            storageType.isSignedInteger(),
            storageType.getIntOrFloatBitWidth()));

    if (oldQuantType && oldQuantType != quantType)
      return rewriter.notifyMatchFailure(op, "Unequal quant types");

    // To avoid disrupting other uses of input, create a copy of defining op
    mlir::Operation *newInputOp = rewriter.clone(*inputOp);

    // Change input tensor to be quant type
    auto results = inputOp->getOpResults();
    unsigned int resultIdx = std::distance(
        results.begin(), std::find(results.begin(), results.end(), input));
    rewriter.modifyOpInPlace(newInputOp, [&]() {
      newInputOp->getResult(resultIdx).setType(outType.clone(quantType));
    });
    {
      onnx_mlir::IgnoreDiagnostic diag(rewriter.getContext()->getDiagEngine());
      if (mlir::failed(mlir::verify(newInputOp)))
        return rewriter.notifyMatchFailure(op, "Quant types not allowed");
    }

    // Remove the Q/DQ op
    rewriter.replaceAllUsesWith(result, newInputOp->getResult(resultIdx));
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

class QuantTypesPass : public mlir::PassWrapper<QuantTypesPass,
                           mlir::OperationPass<mlir::func::FuncOp>> {
  mlir::StringRef getArgument() const override { return "quant-types"; }

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<QuantTypesFrom<mlir::ONNXDequantizeLinearOp>,
        QuantTypesFrom<mlir::ONNXQuantizeLinearOp>>(ctx);
    if (failed(mlir::applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createQuantTypesPass() {
  return std::make_unique<QuantTypesPass>();
}

} // namespace onnx_mlir
