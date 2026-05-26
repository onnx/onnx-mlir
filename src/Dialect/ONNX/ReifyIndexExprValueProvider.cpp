/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ReifyIndexExprValueProvider.cpp - shape IR for reifyResultShapes ---===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements IndexExprValueProvider for reifyResultShapes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ReifyIndexExprValueProvider.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

// Return null if none is found.
ElementsAttr ReifyIndexExprValueProvider::getConst(Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto castOp = dyn_cast_or_null<arith::IndexCastOp>(definingOp)) {
    Value input = castOp.getIn();
    definingOp = input.getDefiningOp();
  }
  if (auto constOp = dyn_cast_or_null<ONNXConstantOp>(definingOp)) {
    if (constOp.getValue().has_value())
      return mlir::dyn_cast<ElementsAttr>(constOp.getValueAttr());
  }
  return nullptr;
}

Value ReifyIndexExprValueProvider::getVal(Value intArrayVal, uint64_t i) {
  Type elemType = getElementType(intArrayVal.getType());
  if (!mlir::isa<IndexType>(elemType)) {
    Type indexTensorType = RankedTensorType::get(
        mlir::cast<ShapedType>(intArrayVal.getType()).getShape(),
        db.getBuilder().getIndexType());
    intArrayVal = arith::IndexCastOp::create(
        db.getBuilder(), db.getLoc(), indexTensorType, intArrayVal);
  }
  ShapeBuilder createShape(db);
  return createShape.getExtent(intArrayVal, i);
}

Value ReifyIndexExprValueProvider::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  ShapeBuilder createShape(db);
  return createShape.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
