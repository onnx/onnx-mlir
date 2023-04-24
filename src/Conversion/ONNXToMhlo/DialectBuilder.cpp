/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ DialectBuilder.cpp - Mhlo dialect builder --------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file contains dialect builder for Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "src/Conversion/ONNXToMhlo/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

// =============================================================================
// IndexExpr Builder for Lowering using Shape/MHLO Dialect.
// =============================================================================

// Return null if none is found.
ElementsAttr IndexExprBuilderForMhlo::getConst(Value value) {
  auto definingOp = value.getDefiningOp();
  // If we have a cast between index/integer, skip it, i.e. get the defining op
  // that is the input to the cast.
  if (auto castOp = dyn_cast_or_null<arith::IndexCastOp>(definingOp)) {
    Value input = castOp.getIn();
    definingOp = input.getDefiningOp();
  }
  if (auto constOp = dyn_cast_or_null<mhlo::ConstantOp>(definingOp)) {
    if (constOp.getValueAttr())
      return constOp.getValueAttr().dyn_cast<ElementsAttr>();
  } else if (auto constOp = dyn_cast_or_null<ONNXConstantOp>(definingOp)) {
    if (constOp.getValue().has_value())
      return constOp.getValueAttr().dyn_cast<ElementsAttr>();
  }
  return nullptr;
}

Value IndexExprBuilderForMhlo::getVal(Value intArrayVal, uint64_t i) {
  Type elemType = getElementType(intArrayVal.getType());
  if (!elemType.isa<IndexType>()) {
    Type indexTensorType = RankedTensorType::get(
        intArrayVal.getType().cast<ShapedType>().getShape(),
        b().getIndexType());
    intArrayVal =
        b().create<arith::IndexCastOp>(loc(), indexTensorType, intArrayVal);
  }
  ShapeBuilder createShape(*this);
  return createShape.getExtent(intArrayVal, i);
}

Value IndexExprBuilderForMhlo::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  ShapeBuilder createShape(*this);
  return createShape.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
