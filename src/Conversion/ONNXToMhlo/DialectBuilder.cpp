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
  MultiDialectBuilder<AffineBuilder, MathBuilder> create(*this);
  // Need to add some acceptable dialects to MHLO conversion.
  llvm_unreachable(
      "unimplemented getVal (see IndexExprBuilderForKrnl for functionality).");
}

Value IndexExprBuilderForMhlo::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  ShapeBuilder createShape(*this);
  return createShape.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
