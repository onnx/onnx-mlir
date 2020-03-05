//===- onnx_rewrite.cpp - ONNX High Level Optimizer -----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters for operations in the ONNX dialect
// that can be rewritten by using other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/dialect/onnx/onnx_ops.hpp"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/onnx_rewrite.inc"

//===----------------------------------------------------------------------===//
// Rewrite:
// %0 = onnx.ConvNoBiasOp(%D : tensor<DShape>, %K)
//     {pads = [b0, b1, ... bK, e0, e1, ..., eK]} ->
//         tensor<OutShape>
//
// as:
// %0 = onnx.PadConstantValuePasOp(%D)
//     {pads = [0, 0, b0, b1, ... bK, 0, 0, e0, e1, ..., eK]} ->
//     tensor<DPaddedShape>
// %1 = onnx.ConvNoBias(%0 : tensor<DPaddedShape>, %K) {pads = [0, ..., 0]} ->
//     tensor<OutShape>
//===----------------------------------------------------------------------===//
struct SplitConvOpPattern : public RewritePattern {
  SplitConvOpPattern(MLIRContext *context)
      : RewritePattern(ONNXConvNoBiasOp::getOperationName(),
                       {ONNXPadConstantValuePadOp::getOperationName(),
                        ONNXConvNoBiasOp::getOperationName()},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // If convolution does not use padding then no rewrite is required.
    ONNXConvNoBiasOp convOp = llvm::dyn_cast<ONNXConvNoBiasOp>(op);
    auto padsAttribute = convOp.padsAttr();
    if (!padsAttribute)
      return matchFailure();

    // If auto_pad is VALID then no padding happens and no rewrite isrequired.
    auto autoPad = convOp.auto_pad();
    if (autoPad == "VALID")
      return matchFailure();

    auto data = op->getOperands()[0];
    auto inputShape = data.getType().cast<TensorType>().getShape();

    // Dimensionality of the input:
    //              inputRank
    //      |----------------------|
    // D : (N x C x D1 x D2 x ... DK)
    //              |______________|
    //                  inputDims
    //
    int64_t inputRank = inputShape.size();
    int64_t inputDims = inputRank - 2;

    // If all pads values are equal to zero then no rewrite is required.
    bool allZeros = true;
    for (auto padsValue : padsAttribute.getValue()) {
      if (padsValue.cast<IntegerAttr>().getInt() > 0) {
        allZeros = false;
        break;
      }
    }

    if (allZeros)
      return matchFailure();

    // Create padding vector for the explicit padding op attribute.
    SmallVector<int64_t, 4> pads(2 * inputRank, 0);
    SmallVector<int64_t, 4> outPaddedShape(inputRank, 0);
    outPaddedShape[0] = inputShape[0];
    outPaddedShape[1] = inputShape[1];
    for (int i = 0; i < inputDims; ++i) {
      int64_t beginPad =
          padsAttribute.getValue()[i].cast<IntegerAttr>().getInt();
      int64_t endPad =
          padsAttribute.getValue()[inputDims + i].cast<IntegerAttr>().getInt();
      pads[i + 2] = beginPad;
      pads[inputRank + i + 2] = endPad;
      outPaddedShape[i + 2] += beginPad + inputShape[i + 2] + endPad;
    }

    // Create padding operation.
    auto inputElemType = data.getType().cast<TensorType>().getElementType();
    ONNXPadConstantValuePadOp paddingOp =
        rewriter.create<ONNXPadConstantValuePadOp>(
            loc, RankedTensorType::get(outPaddedShape, inputElemType), data,
            rewriter.getI64ArrayAttr(pads), FloatAttr::get(inputElemType, 0),
            StringAttr::get("constant", loc->getContext()));

    SmallVector<int64_t, 4> newConvPads(2 * inputDims, 0);
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    ONNXConvNoBiasOp newConvOp = rewriter.create<ONNXConvNoBiasOp>(
            loc, tensorType, paddingOp.getResult(), convOp.getOperands()[1],
            convOp.auto_padAttr(), convOp.dilationsAttr(),
            convOp.groupAttr(), convOp.kernel_shapeAttr(),
            rewriter.getI64ArrayAttr(newConvPads),
            convOp.stridesAttr());

    rewriter.replaceOp(op, newConvOp.getResult());
    return matchSuccess();
  };
};
} // end anonymous namespace

/// on the ONNXReduceSumSquareOp.
void ONNXConvNoBiasOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SplitConvOpPattern>(context);
}
