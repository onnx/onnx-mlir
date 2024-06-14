/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ CategoryMapper.cpp - ONNX Operations --------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect CategoryMapper operation.
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
LogicalResult ONNXCategoryMapperOpShapeHelper::computeShape() {
  ONNXCategoryMapperOpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getX());
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXCategoryMapperOp::verify() {
  ONNXCategoryMapperOpAdaptor operandAdaptor(*this);

  // Check input.
  const Value X = operandAdaptor.getX();
  if (!hasShapeAndRank(X)) {
    // Won't be able to do any checking at this stage.
    return success();
  }

  ShapedType inputType = mlir::cast<ShapedType>(X.getType());
  Type elementType = inputType.getElementType();
  if (!elementType.isInteger(64) && !mlir::isa<ONNXStringType>(elementType))
    return emitOpError("input must be a tensor of int64 or string");

  // Check attributes.
  if (!getCatsInt64s())
    return emitOpError("cats_int64 attribute must be present");
  if (!getCatsStrings())
    return emitOpError("cats_strings attribute must be present");
  if (ArrayAttrSize(getCatsInt64s()) != ArrayAttrSize(getCatsStrings()))
    return emitOpError("cats_int64 and cats_strings should have the same size");

  if (elementType.isInteger(64) && !getDefaultStringAttr())
    return emitOpError("'default_string' attribute is missing.");
  if (mlir::isa<ONNXStringType>(elementType) && !getDefaultInt64Attr())
    return emitOpError("'default_int64' attribute is missing.");

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXCategoryMapperOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getX()))
    return success();

  Type inputElementType =
      mlir::cast<ShapedType>(getX().getType()).getElementType();
  assert((inputElementType.isInteger(64) ||
             mlir::isa<ONNXStringType>(inputElementType)) &&
         "Input tensor must have int64 or string element type.");

  Type outputElementType;
  if (inputElementType.isInteger(64))
    outputElementType = ONNXStringType::get(getContext());
  else
    outputElementType = IntegerType::get(getContext(), /*width=*/64);

  ONNXCategoryMapperOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(outputElementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXCategoryMapperOp>;
} // namespace onnx_mlir
