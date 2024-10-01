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
    valAttr =
        mlir::cast<SparseElementsAttr>(operandAdaptor.getSparseValueAttr());
  else
    valAttr = mlir::cast<ElementsAttr>(operandAdaptor.getValueAttr());
  return setOutputDimsFromTypeWithConstantShape(valAttr.getType());
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXConstantOp::resultTypeInference() {
  ShapedType type;
  if (auto attr = getValueAttr()) {
    type = mlir::cast<ElementsAttr>(attr).getShapedType();
  } else if (auto attr = getSparseValueAttr()) {
    type = mlir::cast<ElementsAttr>(attr).getShapedType();
  } else if (auto attr = getValueFloatAttr()) {
    type = RankedTensorType::get({}, FloatType::getF32(getContext()));
  } else if (auto attr = getValueFloatsAttr()) {
    int64_t size = attr.size();
    type = RankedTensorType::get({size}, FloatType::getF32(getContext()));
  } else if (auto attr = getValueIntAttr()) {
    type = RankedTensorType::get({}, IntegerType::get(getContext(), 64));
  } else if (auto attr = getValueIntsAttr()) {
    int64_t size = attr.size();
    type = RankedTensorType::get({size}, IntegerType::get(getContext(), 64));
  } else if (auto attr = getValueStringAttr()) {
    type = RankedTensorType::get({}, ONNXStringType::get(getContext()));
  } else if (auto attr = getValueStringsAttr()) {
    int64_t size = attr.size();
    type = RankedTensorType::get({size}, ONNXStringType::get(getContext()));
  } else {
    llvm_unreachable("Unexpected attributes for Constant Op");
  }
  return {type};
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXConstantOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if ((getSparseValue().has_value() && getValue().has_value()) ||
      (!getSparseValue().has_value() && !getValue().has_value())) {
    // This can happen in ONNXHybridTransformPass where shape inference is run
    // before canonicalization, which normalizes constant ops with other
    // attributes (see ONNXConstantOpNormalize in Rewrite.td).
    // We could implement shape inference here for all the attributes but it's
    // simpler to do nothing and punt it to a subsequent canonicalization pass.
    return success();
  }
  ElementsAttr valAttr;
  if (getSparseValue().has_value())
    valAttr = mlir::cast<SparseElementsAttr>(getSparseValueAttr());
  else
    valAttr = mlir::cast<ElementsAttr>(getValueAttr());
  Type elementType =
      mlir::cast<RankedTensorType>(valAttr.getType()).getElementType();
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
  if (auto sparseValue = getSparseValueAttr())
    return sparseValue;
  if (auto value = getValueAttr())
    return value;
  else if (auto valueFloat = getValueFloatAttr())
    return valueFloat;
  else if (auto valueInt = getValueIntAttr())
    return valueInt;
  else if (auto valueInts = getValueIntsAttr())
    return valueInts;
  else if (auto valueFloats = getValueFloatsAttr())
    return valueFloats;
  else if (auto valueString = getValueStringAttr())
    return valueString;
  else if (auto valueStrings = getValueStringsAttr())
    return valueStrings;
  else
    llvm_unreachable("ONNXConstantOp does not have a valid attribute");
}
