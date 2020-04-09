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
    RNNActivation activation, Value input, Type elementType) {
  Value res;

  auto zeroIndex = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
  SmallVector<Value, 4> IVs{zeroIndex};

  MemRefType scalarMemRefType = MemRefType::get({1}, elementType, {}, 0);
  Value alloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
  rewriter.create<StoreOp>(loc, input, alloc, IVs);

  std::vector<mlir::NamedAttribute> attributes;
  if (activation.alpha) {
    attributes.emplace_back(
        rewriter.getNamedAttr("alpha", activation.alpha.getValue()));
  }
  if (activation.beta) {
    attributes.emplace_back(
        rewriter.getNamedAttr("beta", activation.beta.getValue()));
  }

  if (activation.name == "relu")
    res = rewriter.create<ONNXReluOp>(loc, scalarMemRefType, alloc);
  else if (activation.name == "tanh")
    res = rewriter.create<ONNXTanhOp>(loc, scalarMemRefType, alloc);
  else if (activation.name == "sigmoid")
    res = rewriter.create<ONNXSigmoidOp>(loc, scalarMemRefType, alloc);
  else if (activation.name == "affine")
    emitError(loc, "Unsupported activation");
  else if (activation.name == "leakyrelu")
    res = rewriter.create<ONNXLeakyReluOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name == "thresholdedrelu")
    res = rewriter.create<ONNXThresholdedReluOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name == "scaledtanh")
    emitError(loc, "Unsupported activation");
  else if (activation.name == "hardsigmoid")
    res = rewriter.create<ONNXHardSigmoidOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name == "elu")
    res = rewriter.create<ONNXEluOp>(loc, scalarMemRefType, alloc, attributes);
  else if (activation.name == "softsign")
    res = rewriter.create<ONNXSoftsignOp>(loc, scalarMemRefType, alloc);
  else if (activation.name == "softplus")
    res = rewriter.create<ONNXSoftplusOp>(loc, scalarMemRefType, alloc);
  else
    return res;

  Value result = rewriter.create<LoadOp>(loc, res, IVs);
  return result;
}
