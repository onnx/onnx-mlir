/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Constant.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
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
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXConstantOpShapeHelper::computeShape() {
  ONNXConstantOpAdaptor operandAdaptor(operands, op->getAttrDictionary());

  ElementsAttr valAttr;
  if (operandAdaptor.getSparseValue().has_value())
    valAttr = operandAdaptor.getSparseValueAttr().cast<SparseElementsAttr>();
  else
    valAttr = operandAdaptor.getValueAttr().cast<ElementsAttr>();
  return setOutputDimsFromTypeWithConstantShape(valAttr.getType());
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXConstantOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if ((getSparseValue().has_value() && getValue().has_value()) ||
      (!getSparseValue().has_value() && !getValue().has_value()))
    return emitError("Require exactly one of the two attributes, "
                     "either value or sparse_value");
  ElementsAttr valAttr;
  if (getSparseValue().has_value())
    valAttr = getSparseValueAttr().cast<SparseElementsAttr>();
  else
    valAttr = getValueAttr().cast<ElementsAttr>();
  Type elementType =
      valAttr.getType().cast<RankedTensorType>().getElementType();
  ONNXConstantOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXConstantOp>;
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Folder
//===----------------------------------------------------------------------===//

OpFoldResult ONNXConstantOp::fold(FoldAdaptor adaptor) {
  if (getValueAttr())
    return getValueAttr();
  else if (getValueFloatAttr())
    return getValueFloatAttr();
  else if (getValueIntAttr())
    return getValueIntAttr();
  else if (getValueIntsAttr())
    return getValueIntsAttr();
  else if (getValueFloatsAttr())
    return getValueFloatsAttr();
  else {
    assert(getValueStringAttr() &&
           "ONNXConstantOp does not have a valid attribute");
    return getValueStringAttr();
  }
}
