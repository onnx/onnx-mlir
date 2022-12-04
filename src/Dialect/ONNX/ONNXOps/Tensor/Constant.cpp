/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Constant.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Constant operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Fold
//===----------------------------------------------------------------------===//

OpFoldResult ONNXConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constants take no operands");
  if (Attribute sparse = sparse_valueAttr())
    return sparse;
  if (Attribute dense = valueAttr())
    return dense;

  // A handful of funny attributes that appear in a lit test:
  if (FloatAttr floatAttr = value_floatAttr()) {
    // rank 1 seems wrong, but if we set rank to 0 a constant with rank 1 is
    // created somehow and we shouldn't have type mismatch with the attribute
    return DenseElementsAttr::get(
        RankedTensorType::get({1}, FloatType::getF32(getContext())), floatAttr);
  }
  if (ArrayAttr floatsAttr = value_floatsAttr()) {
    return DenseElementsAttr::get(
        RankedTensorType::get(
            {int64_t(floatsAttr.size())}, FloatType::getF32(getContext())),
        floatsAttr.getValue());
  }
  if (IntegerAttr intAttr = value_intAttr()) {
    // rank 1 seems wrong, but if we set rank to 0 a constant with rank 1 is
    // created somehow and we shouldn't have type mismatch with the attribute
    return DenseElementsAttr::get(
        RankedTensorType::get({1}, IntegerType::get(getContext(), 64)),
        intAttr.getSInt());
  }
  if (ArrayAttr intsAttr = value_intsAttr()) {
    return DenseElementsAttr::get(
        RankedTensorType::get(
            {int64_t(intsAttr.size())}, IntegerType::get(getContext(), 64)),
        intsAttr.getValue());
  }

  // Only sparse and dense constants are used in practice so we don't bother
  // implementing all the others.
  llvm_unreachable("only sparse and dense constants can be folded");
}

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXConstantOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if ((sparse_value().has_value() && value().has_value()) ||
      (!sparse_value().has_value() && !value().has_value()))
    return emitError("Require exactly one of the two attributes, "
                     "either value or sparse_value");
  ElementsAttr valAttr;
  if (sparse_value().has_value())
    valAttr = sparse_valueAttr().cast<SparseElementsAttr>();
  else
    valAttr = valueAttr().cast<ElementsAttr>();
  getResult().setType(valAttr.getType());
  return success();
}
