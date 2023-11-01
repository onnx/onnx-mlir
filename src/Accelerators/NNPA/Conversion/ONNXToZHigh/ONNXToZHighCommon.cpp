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
#include "src/Dialect/ONNX/DialectBuilder.hpp"

using namespace mlir;
namespace onnx_mlir {

/// Get transposed tensor by using a permutation array.
Value emitONNXTranspose(
    Location loc, PatternRewriter &rewriter, Value x, ArrayRef<int64_t> perms) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  Value result = create.onnx.transposeInt64(x, perms);
  return result;
}

/// Get transposed tensor by using a permutation array and a result type.
Value emitONNXTransposeWithType(Location loc, PatternRewriter &rewriter,
    Type transposedType, Value x, ArrayRef<int64_t> perms) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  Value result =
      create.onnx.transpose(transposedType, x, rewriter.getI64ArrayAttr(perms));
  return result;
}

/// Split a tensor along an axis by chunkSize. The last chunk becomes smaller
/// than it. The default chunkSize is NNPA_MAXIMUM_DIMENSION_INDEX_SIZE.
ValueRange splitAlongAxis(MultiDialectBuilder<OnnxBuilder> &create, Value X,
    int64_t axis, int64_t chunkSize) {
  Type xType = X.getType();
  ArrayRef<int64_t> xShape = getShape(xType);
  Type elementTy = getElementType(xType);

  // Compute split sizes.
  SmallVector<Type> splitTy;
  SmallVector<int64_t> splitSizesI64;
  SmallVector<int64_t> splitShape(xShape);
  int64_t dimSize = xShape[axis];
  // First splits have the same size of chunkSize.
  while (dimSize > chunkSize) {
    splitShape[axis] = chunkSize;
    auto ty = RankedTensorType::get(splitShape, elementTy);
    splitTy.emplace_back(ty);
    splitSizesI64.emplace_back(chunkSize);
    dimSize -= chunkSize;
  }
  // The last split.
  splitShape[axis] = dimSize;
  auto ty = RankedTensorType::get(splitShape, elementTy);
  splitTy.emplace_back(ty);
  splitSizesI64.emplace_back(dimSize);

  Value splitSizes = create.onnx.constantInt64(splitSizesI64);
  ValueRange splits = create.onnx.split(splitTy, X, splitSizes, axis);
  return splits;
}

} // namespace onnx_mlir
