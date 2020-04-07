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
