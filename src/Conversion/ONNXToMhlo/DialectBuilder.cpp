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

#include "src/Conversion/ONNXToMhlo/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {


// =============================================================================
// IndexExpr Builder for Lowering using Shape/MHLO Dialect.
// =============================================================================

// Return null if none is found.
DenseElementsAttr IndexExprBuilderForMhlo::getConst(Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto globalOp = dyn_cast_or_null<mhlo::ConstantOp>(definingOp)) {
    if (globalOp.getValueAttr())
      return globalOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  } else if (auto globalOp = dyn_cast_or_null<ONNXConstantOp>(definingOp)) {
    if (globalOp.value().has_value())
      return globalOp.valueAttr().dyn_cast<DenseElementsAttr>();
  }
  return nullptr;
}

Value IndexExprBuilderForMhlo::getVal(Value intArrayVal, uint64_t i) {
  MultiDialectBuilder<AffineBuilder, MathBuilder> create(*this);
  llvm_unreachable("unimplemented (see IndexExprBuilderForKrnl for functionality).");
}

Value IndexExprBuilderForMhlo::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  ShapeBuilder createShape(*this);
  return createShape.dim(tensorOrMemrefValue, i);
}


} // namespace onnx_mlir
