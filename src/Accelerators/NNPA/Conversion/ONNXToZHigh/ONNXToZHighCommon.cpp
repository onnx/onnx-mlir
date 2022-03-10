/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- ONNXToZHighCommon.cpp - Common functions to ZHigh lowering ----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file includes utility functions for lowering ONNX operations to a
// combination of ONNX and ZHigh operations.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"

using namespace mlir;

/// Get transposed tensor by using a permutation array.
/// TODO: migrate this to onnx-mlir.
Value emitONNXTranspose(
    Location loc, PatternRewriter &rewriter, Value x, ArrayRef<int64_t> perms) {
  ShapedType inputType = x.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  Type transposedType;
  if (inputType.hasRank()) {
    assert((uint64_t)inputType.getRank() == perms.size() &&
           "Permutation array size is different from the input rank");
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<int64_t, 4> transposedShape;
    for (uint64_t i = 0; i < perms.size(); ++i)
      transposedShape.emplace_back(inputShape[perms[i]]);
    transposedType = RankedTensorType::get(transposedShape, elementType);
  } else {
    transposedType = UnrankedTensorType::get(elementType);
  }

  ONNXTransposeOp transposedInput = rewriter.create<ONNXTransposeOp>(
      loc, transposedType, x, rewriter.getI64ArrayAttr(perms));
  return transposedInput.getResult();
}

/// Get transposed tensor by using a permutation array and a result type.
/// TODO: migrate this to onnx-mlir.
Value emitONNXTransposeWithType(Location loc, PatternRewriter &rewriter,
    Type transposedType, Value x, ArrayRef<int64_t> perms) {
  ONNXTransposeOp transposedInput = rewriter.create<ONNXTransposeOp>(
      loc, transposedType, x, rewriter.getI64ArrayAttr(perms));
  return transposedInput.getResult();
}
