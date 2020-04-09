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
    auto alphaAttr = rewriter.getF32FloatAttr(activation.alpha.getValue());
    attributes.emplace_back(rewriter.getNamedAttr("alpha", alphaAttr));
  }
  if (activation.beta) {
    auto betaAttr = rewriter.getF32FloatAttr(activation.beta.getValue());
    attributes.emplace_back(rewriter.getNamedAttr("beta", betaAttr));
  }

  if (activation.name == "relu")
    res = rewriter.create<ONNXReluOp>(loc, scalarMemRefType, alloc);
  else if (activation.name == "tanh")
    res = rewriter.create<ONNXTanhOp>(loc, scalarMemRefType, alloc);
  else if (activation.name == "sigmoid")
    res = rewriter.create<ONNXSigmoidOp>(loc, scalarMemRefType, alloc);
  else if (activation.name == "affine")
    emitError(loc, "Unsupported activation");
  else if (activation.name == "LeakyRelu")
    res = rewriter.create<ONNXLeakyReluOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name == "ThresholdedRelu")
    res = rewriter.create<ONNXThresholdedReluOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name == "ScaledTanh")
    emitError(loc, "Unsupported activation");
  else if (activation.name == "HardSigmoid")
    res = rewriter.create<ONNXHardSigmoidOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name == "Elu")
    res = rewriter.create<ONNXEluOp>(loc, scalarMemRefType, alloc, attributes);
  else if (activation.name == "Softsign")
    res = rewriter.create<ONNXSoftsignOp>(loc, scalarMemRefType, alloc);
  else if (activation.name == "Softplus")
    res = rewriter.create<ONNXSoftplusOp>(loc, scalarMemRefType, alloc);
  else
    return res;

  Value result = rewriter.create<LoadOp>(loc, res, IVs);
  return result;
}

Value activation_g(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value input, Type elementType) {
  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto two = emitConstantOp(rewriter, loc, elementType, 2);
  auto neg = rewriter.create<SubFOp>(loc, zero, input);
  auto exp = rewriter.create<ExpOp>(loc, input);
  auto negExp = rewriter.create<ExpOp>(loc, neg);

  auto sinh = rewriter.create<DivFOp>(
      loc, rewriter.create<SubFOp>(loc, exp, negExp), two);
  auto cosh = rewriter.create<DivFOp>(
      loc, rewriter.create<AddFOp>(loc, exp, negExp), two);

  return rewriter.create<DivFOp>(loc, sinh, cosh);
}

Value activation_h(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value input, Type elementType) {
  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto two = emitConstantOp(rewriter, loc, elementType, 2);
  auto neg = rewriter.create<SubFOp>(loc, zero, input);
  auto exp = rewriter.create<ExpOp>(loc, input);
  auto negExp = rewriter.create<ExpOp>(loc, neg);

  auto sinh = rewriter.create<DivFOp>(
      loc, rewriter.create<SubFOp>(loc, exp, negExp), two);
  auto cosh = rewriter.create<DivFOp>(
      loc, rewriter.create<AddFOp>(loc, exp, negExp), two);

  return rewriter.create<DivFOp>(loc, sinh, cosh);
}
