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

bool isF32ScalarConstantTensor(mlir::Value v) {
  if (!isScalarConstantTensor(v))
    return false;
  auto t = dyn_cast<ShapedType>(v.getType());
  return t.getElementType().isF32();
}

FloatAttr getScalarF32AttrFromConstant(Value v) {
  if (!isF32ScalarConstantTensor(v))
    return nullptr;
  DenseElementsAttr constElements = ElementsAttrBuilder::toDenseElementsAttr(
      getElementAttributeFromONNXValue(v));
  return constElements.getSplatValue<FloatAttr>();
}

Value getDynShape(Location loc, PatternRewriter &rewriter, Value x) {
  if (!hasShapeAndRank(x))
    llvm_unreachable("The input must have shape and rank");

  OnnxBuilder create(rewriter, loc);
  auto t = dyn_cast<ShapedType>(x.getType());
  int64_t r = t.getRank();
  SmallVector<Value> dims;
  for (int64_t i = 0; i < r; ++i) {
    Value d = create.dim(x, i);
    dims.emplace_back(d);
  }
  return create.concat(
      RankedTensorType::get({r}, rewriter.getI64Type()), dims, 0);
}

SmallVector<int64_t, 2> getParallelOpt(std::string nnpaMatMulParallelOpt) {
  SmallVector<int64_t, 2> opts;
  if (!nnpaMatMulParallelOpt.empty()) {
    size_t pos = nnpaMatMulParallelOpt.find(':');
    std::string nDevString = nnpaMatMulParallelOpt.substr(0, pos);
    std::string thresholdString = nnpaMatMulParallelOpt.substr(pos + 1);
    if (nDevString.empty())
      opts.emplace_back(1);
    else
      opts.emplace_back(std::stoi(nDevString));
    if (thresholdString.empty())
      opts.emplace_back(32);
    else
      opts.emplace_back(std::stoi(thresholdString));
  } else {
    // Default
    opts.emplace_back(1);
    opts.emplace_back(32);
  }
  return opts;
}

} // namespace onnx_mlir
