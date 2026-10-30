/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------------ DialectBuilder.cpp - Linalg dialect builder ------------===//
//
// Copyright 2022-2026
//
// =============================================================================
//
// This file contains dialect builder for Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToLinalg/DialectBuilder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// =============================================================================
// IndexExpr Builder for Lowering using Arith/Tensor Dialect.
// =============================================================================

ElementsAttr IndexExprBuilderForTensor::getConst(Value value) {
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>())
    return dyn_cast<ElementsAttr>(constantOp.getValue());
  return getElementAttributeFromONNXValue(value);
}

Value IndexExprBuilderForTensor::getVal(Value array, uint64_t index) {
  // If the shape comes from an onnx.Concat of mixed constant/runtime values,
  // then we can potentially get a constant value at index.
  if (areDimsFromConcat(array)) {
    SmallVector<Value, 4> dims;
    getDims(array, dims);
    if (isDenseONNXConstant(dims[index]))
      return dims[index];
  }

  auto arrayType = cast<RankedTensorType>(array.getType());
  SmallVector<Value> indices;
  if (arrayType.getRank() != 0)
    indices.push_back(arith::ConstantIndexOp::create(b(), loc(), index));
  return tensor::ExtractOp::create(b(), loc(), array, indices);
}

Value IndexExprBuilderForTensor::getShapeVal(Value tensor, uint64_t index) {
  return tensor::DimOp::create(b(), loc(), tensor, index);
}

} // namespace onnx_mlir
