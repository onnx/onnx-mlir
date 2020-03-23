//===----------- ONNXRewrite.cpp - ONNX High Level Optimizer --------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters for operations in the ONNX dialect
// that can be rewritten by using other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {

// Check whether an ArrayAttr contains non-zero values or not.
bool hasNonZeroInArrayAttr(ArrayAttr attrs) {
  bool allZeros = true;
  if (attrs) {
    for (auto attr: attrs.getValue()) {
      if (attr.cast<IntegerAttr>().getInt() > 0) {
        allZeros = false;
        break;
      }
    }
  }
  return !allZeros;
}

// Create an ArrayAttr of IntergerAttr(s) of zero values.
// This function is used for padding attribute in MaxPoolSingleOut.
ArrayAttr createArrayAttrOfZeros(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  int nElements = origAttrs.getValue().size();
  SmallVector<int64_t, 4> vals(nElements, 0);
  return rewriter.getI64ArrayAttr(vals);
}

// Pad a ArrayAttr with zeros.
//
// pads = [B1, B2, ... Bk, E1, E2, ..., Ek]
//
// becomes:
//
// pads = [0,... 0, B1, B2, ... Bk, 0,... 0, E1, E2, ..., Ek]
//         |_____|                  |_____|
//                 nZeros                    nZeros
//
// This function is used for padding attribute in MaxPoolSingleOut.
ArrayAttr insertZerosForNonPaddedDims(
    PatternRewriter &rewriter, ArrayAttr origAttrs, int extensionLength) {
  int nDims = (int) origAttrs.getValue().size() / 2;
  int nElements = (nDims + extensionLength) * 2;
  SmallVector<int64_t, 4> pads(nElements, 0);
  for (int i = 0; i < nDims; ++i) {
    int64_t beginPad = origAttrs.getValue()[i].cast<IntegerAttr>().getInt();
    int64_t endPad =
        origAttrs.getValue()[nDims + i].cast<IntegerAttr>().getInt();
    pads[i + extensionLength] = beginPad;
    pads[nDims + extensionLength + i + extensionLength] = endPad;
  }
  return rewriter.getI64ArrayAttr(pads);
}

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Transform/ONNX/ONNXRewrite.inc"

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

/// on the ONNXMaxPoolSingleOutOp.
void ONNXMaxPoolSingleOutOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MaxPoolSingleOutOpPaddingPattern>(context);
}
/// on the ONNXConvNoBiasOp.
void ONNXConvNoBiasOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SplitConvOpPattern>(context);
}
