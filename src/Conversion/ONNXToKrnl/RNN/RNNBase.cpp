//===--------------- RNNBase.cpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines base functions for lowerng the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"

using namespace mlir;

Value applyActivation(ConversionPatternRewriter &rewriter, Location loc,
    RNNActivation activation, Value scalarOperand) {
  Value res;

  MemRefType scalarMemRefType =
      MemRefType::get({}, scalarOperand.getType(), {}, 0);
  Value alloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
  rewriter.create<StoreOp>(loc, scalarOperand, alloc);

  std::vector<mlir::NamedAttribute> attributes;
  if (activation.alpha) {
    attributes.emplace_back(
        rewriter.getNamedAttr("alpha", activation.alpha.getValue()));
  }
  if (activation.beta) {
    attributes.emplace_back(
        rewriter.getNamedAttr("beta", activation.beta.getValue()));
  }

  if (activation.name.equals_lower("relu"))
    res = rewriter.create<ONNXReluOp>(loc, scalarMemRefType, alloc);
  else if (activation.name.equals_lower("tanh"))
    res = rewriter.create<ONNXTanhOp>(loc, scalarMemRefType, alloc);
  else if (activation.name.equals_lower("sigmoid"))
    res = rewriter.create<ONNXSigmoidOp>(loc, scalarMemRefType, alloc);
  else if (activation.name.equals_lower("affine"))
    emitError(loc, "Unsupported activation");
  else if (activation.name.equals_lower("leakyrelu"))
    res = rewriter.create<ONNXLeakyReluOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name.equals_lower("thresholdedrelu"))
    res = rewriter.create<ONNXThresholdedReluOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name.equals_lower("scaledtanh"))
    emitError(loc, "Unsupported activation");
  else if (activation.name.equals_lower("hardsigmoid"))
    res = rewriter.create<ONNXHardSigmoidOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name.equals_lower("elu"))
    res = rewriter.create<ONNXEluOp>(loc, scalarMemRefType, alloc, attributes);
  else if (activation.name.equals_lower("softsign"))
    res = rewriter.create<ONNXSoftsignOp>(loc, scalarMemRefType, alloc);
  else if (activation.name.equals_lower("softplus"))
    res = rewriter.create<ONNXSoftplusOp>(loc, scalarMemRefType, alloc);
  else
    return res;

  Value result = rewriter.create<LoadOp>(loc, res);
  return result;
}
