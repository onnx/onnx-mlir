/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ConstantOfShape.cpp - ONNX Operations -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect ConstantOfShape operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXConstantOfShapeOpShapeHelper::computeShape() {
  ONNXConstantOfShapeOpAdaptor operandAdaptor(operands);
  Value input = operandAdaptor.getInput();
  DimsExpr outputDims;

  auto inputShape = mlir::cast<RankedTensorType>(input.getType()).getShape();
  if (inputShape[0] == 0) {
    // If 'input' is an empty tensor, the output would be a scalar.
    // Represent this by an empty outputDims.
    outputDims.clear();
  } else {
    // Calculate output dimensions.
    createIE->getIntFromArrayAsDims(input, outputDims);
  }
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXConstantOfShapeOp::verify() {
  ONNXConstantOfShapeOpAdaptor operandAdaptor(*this);
  auto input = operandAdaptor.getInput();
  if (!hasShapeAndRank(input))
    return success();

  auto inputShape = mlir::cast<RankedTensorType>(input.getType()).getShape();
  if (inputShape.size() != 1)
    return emitOpError("Input tensor must be a 1D tensor");

  if (ShapedType::isDynamic(inputShape[0]))
    return success();

  // Calculate output dimensions.
  SmallVector<int64_t, 4> outputDims(inputShape[0], ShapedType::kDynamic);
  // If 'input' is a constant, check whether its values are valid or not.
  // If the values are valid, it is possible to infer shape.
  if (auto constantOp = getONNXConstantOp(input)) {
    ElementsAttr valueAttribute =
        mlir::cast<ElementsAttr>(constantOp.getValueAttr());
    if (isElementAttrUninitializedDenseResource(valueAttribute)) {
      return success(); // Return success to allow the parsing of MLIR with
                        // elided attributes
    }
    // Get repeat values from valueAttribute.
    auto valueIt = valueAttribute.getValues<IntegerAttr>().begin();
    for (int i = 0; i < inputShape[0]; ++i) {
      auto dim = mlir::cast<IntegerAttr>((*valueIt++)).getInt();
      if (dim < 0)
        return emitOpError("All values of the input tensor must be >=0");
    }
    // Unreachable error: Type error will trigger before this occurs
    // No test needed for this error -----
    if (valueIt != valueAttribute.getValues<IntegerAttr>().end())
      return emitOpError(
          "Constant value must have same length as output's rank");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXConstantOfShapeOp::resultTypeInference() {
  Type elementType;
  if (auto attr = getValueAttr()) {
    elementType = mlir::cast<ElementsAttr>(attr).getElementType();
  } else {
    elementType = FloatType::getF32(getContext());
  }
  return {UnrankedTensorType::get(elementType)};
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXConstantOfShapeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ShapedType inputType = mlir::cast<ShapedType>(getInput().getType());
  if (!inputType.hasStaticShape())
    return success();

  Type elementType;

  // 'value' attribute is a one-element tensor whose value and datatype are
  // used to set the output tensor value and datatype.
  if (getValue().has_value()) {
    elementType = mlir::cast<ElementsAttr>(getValueAttr())
                      .getShapedType()
                      .getElementType();
  } else {
    // If 'value' attribute is not specified, it defaults to a tensor of
    // value 0 and datatype float32.
    elementType = FloatType::getF32(getContext());

    llvm::SmallVector<int64_t, 2> dims(1, 1);
    auto tensorType = RankedTensorType::get(dims, elementType);

    llvm::SmallVector<float, 1> values(1, 0.);
    setValueAttr(DenseElementsAttr::get(tensorType, llvm::ArrayRef(values)));
  }

  ONNXConstantOfShapeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXConstantOfShapeOp>;
} // namespace onnx_mlir
