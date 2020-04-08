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
    std::string activation, Value input, Type elementType,
    Optional<float> alpha, Optional<float> beta) {
  Value res;

  MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
  Value alloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
  rewriter.create<StoreOp>(loc, input, alloc);

  std::vector<mlir::NamedAttribute> attributes;
  if (alpha) {
    auto alphaAttr = rewriter.getF32FloatAttr(alpha.getValue());
    attributes.emplace_back(rewriter.getNamedAttr("alpha", alphaAttr));
  }
  if (beta) {
    auto betaAttr = rewriter.getF32FloatAttr(beta.getValue());
    attributes.emplace_back(rewriter.getNamedAttr("beta", betaAttr));
  }

  if (activation == "relu")
    res = rewriter.create<ONNXReluOp>(loc, scalarMemRefType, alloc);
  else if (activation == "tanh")
    res = rewriter.create<ONNXTanhOp>(loc, scalarMemRefType, alloc);
  else if (activation == "sigmoid")
    res = rewriter.create<ONNXSigmoidOp>(loc, scalarMemRefType, alloc);
  else if (activation == "affine")
    emitError(loc, "Unsupported activation");
  else if (activation == "LeakyRelu")
    res = rewriter.create<ONNXLeakyReluOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation == "ThresholdedRelu")
    res = rewriter.create<ONNXThresholdedReluOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation == "ScaledTanh")
    emitError(loc, "Unsupported activation");
  else if (activation == "HardSigmoid")
    res = rewriter.create<ONNXHardSigmoidOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation == "Elu")
    res = rewriter.create<ONNXEluOp>(loc, scalarMemRefType, alloc, attributes);
  else if (activation == "Softsign")
    res = rewriter.create<ONNXSoftsignOp>(loc, scalarMemRefType, alloc);
  else if (activation == "Softplus")
    res = rewriter.create<ONNXSoftplusOp>(loc, scalarMemRefType, alloc);
  else
    return res;

  Value result = rewriter.create<LoadOp>(loc, res);
  return result;
}

Value activation_f(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value input, Type elementType) {
  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto neg = rewriter.create<SubFOp>(loc, zero, input);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, one, rewriter.create<AddFOp>(loc, one, negExp));
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
