/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- ONNXToMhloCommon.cpp - ONNX dialects to Mhlo lowering ---------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the MHLO dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"

namespace onnx_mlir {

Value getShapedZero(Location loc, ConversionPatternRewriter &rewriter,
    const ShapedType &inpType, Value &inp, const Type &resultType) {
  Value broadcastedZero;
  if (inpType.hasStaticShape())
    broadcastedZero =
        rewriter.create<mhlo::ConstantOp>(loc, rewriter.getZeroAttr(inpType));
  else {
    Type elemType = inpType.getElementType();
    Value zero =
        rewriter.create<mhlo::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, inp);
    broadcastedZero = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, resultType, zero, shape, rewriter.getI64TensorAttr({}));
  }
  return broadcastedZero;
}

} // namespace onnx_mlir
