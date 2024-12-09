/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ SqueezeUnsqueeze.cpp - ONNX Operations ------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Squeeze and Unsqueeze
// operation.
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

// This function will either take the dims to squeeze from the the shape (when
// axlesFromShape is true) or from the squeezedDims.
// When reading the dims from shape, the shape must be compile time constant.
// WHen reading the dims from squeezedDims, these also need to be compile time
// constant and within range of the data rank.
// Regardless of the source, the axles to squeeze will be stored in the shape
// helper data structure "squeezedAxes" as a vector of ints.
// If there is any new values, or modified value, the new values will be
// reflected in the operation, either as a ONNX const (v13 or more) or as an
// attribute (v11). This saving is performed using the specialized "saveAxes()"
// function.
template <typename OP_TYPE>
LogicalResult ONNXCommonSqueezeOpShapeHelper<OP_TYPE>::customComputeShape(
    DimsExpr &squeezedDims, bool axesFromShape) {
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  DimsExpr outputDims;
  Value data = operandAdaptor.getData();
  if (!hasShapeAndRank(data)) {
    return failure();
  }
  int64_t dataRank = createIE->getShapedTypeRank(data);

  // Init state.
  bool modified = false;
  squeezedAxes.clear();

  if (axesFromShape) {
    // Extract dimensions to squeeze from the shape.
    // Get the shape as symbols, so that we may detect if some are dynamic.
    DimsExpr dataShape;
    createIE->getShapeAsSymbols(data, dataShape);
    for (int i = 0; i < dataRank; ++i) {
      // Check if the dimension to squeeze is a literal and in range.
      if (!dataShape[i].isLiteral())
        return op->emitError(
            "Can not squeeze from dynamic dimensions at this time");
      int64_t shape = dataShape[i].getLiteral();
      assert(shape != ShapedType::kDynamic &&
             "Compile time shape should be nonnegative");
      if (shape == 1) {
        // We will squeeze dim i as its shape is 1.
        squeezedAxes.emplace_back(i);
        modified = true;
      }
    }
  } else {
    // Normalize the axis values, record modified values in squeezedDims.
    for (uint64_t i = 0; i < squeezedDims.size(); ++i) {
      // Check if the dimension to squeeze is a literal and in range.
      if (!squeezedDims[i].isLiteral())
        return op->emitError(
            "Can not squeeze from dynamic dimensions at this time");
      int64_t a = squeezedDims[i].getLiteral();
      if (a < -dataRank || a >= dataRank)
        return op->emitError("Invalid axis value");
      // Handle negative axis.
      if (a < 0) {
        a += dataRank;
        modified = true;
      }
      squeezedAxes.emplace_back(a);
    }
  }

  // Keep all of the dims that are not squeezed.
  for (int i = 0; i < dataRank; ++i)
    if (std::find(squeezedAxes.begin(), squeezedAxes.end(), i) ==
        squeezedAxes.end())
      // Did not find it, add shape.
      outputDims.emplace_back(createIE->getShapeAsDim(data, i));

  // Save the final result.
  setOutputDims(outputDims);
  // Save the modified state
  if (modified)
    saveAxes();
  return success();
}

template <>
void ONNXSqueezeOpShapeHelper::saveAxes() {
  // Create a ConstantOp associated with this Squeeze Op
  // There could be an issue if we were to generate a constant Op late in
  // lowering, but since we normalize them during the first shape inference, we
  // should never encounter a "saveAxles" situation during lowering.

  ONNXSqueezeOp squeezeOp = llvm::cast<ONNXSqueezeOp>(op);
  SaveOnnxConstInOp(op, squeezeOp.getAxesMutable(), squeezedAxes);
}

template <>
void ONNXSqueezeV11OpShapeHelper::saveAxes() {
  SaveOnnxAttrInOp<ONNXSqueezeV11Op>(op, squeezedAxes,
      [](ONNXSqueezeV11Op op, ArrayAttr attr) { op.setAxesAttr(attr); });
}

template <>
LogicalResult ONNXSqueezeOpShapeHelper::computeShape() {
  ONNXSqueezeOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  Value axes = operandAdaptor.getAxes();
  SmallVector<IndexExpr, 4> squeezedDims;
  bool squeezeFromShape = false;
  if (isNoneValue(axes))
    squeezeFromShape = true;
  else
    createIE->getIntFromArrayAsSymbols(axes, squeezedDims);
  return customComputeShape(squeezedDims, squeezeFromShape);
}

template <>
LogicalResult ONNXSqueezeV11OpShapeHelper::computeShape() {
  ONNXSqueezeV11OpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  auto axesAttr = operandAdaptor.getAxesAttr();
  SmallVector<IndexExpr, 4> squeezedDims;
  bool squeezeFromShape = false;
  if (axesAttr)
    createIE->getIntFromArrayAsLiterals(axesAttr, squeezedDims);
  else
    squeezeFromShape = true;
  return customComputeShape(squeezedDims, squeezeFromShape);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXSqueezeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto dataType = mlir::dyn_cast<RankedTensorType>(getData().getType());
  if (!dataType)
    return success();

  Type elementType = dataType.getElementType();
  ONNXSqueezeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

LogicalResult ONNXSqueezeV11Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto dataType = mlir::dyn_cast<RankedTensorType>(getData().getType());
  if (!dataType)
    return success();

  Type elementType = dataType.getElementType();
  ONNXSqueezeV11OpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXCommonSqueezeOpShapeHelper<ONNXSqueezeOp>;
template struct ONNXCommonSqueezeOpShapeHelper<ONNXSqueezeV11Op>;
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Folder
//===----------------------------------------------------------------------===//
OpFoldResult ONNXSqueezeOp::fold(FoldAdaptor adaptor) {
  // Fold type
  if (failed(inferShapes(nullptr)))
    return nullptr;

  // Fold value
  if (!adaptor.getData() || !adaptor.getAxes()) {
    // Use original Op if Data or Axes is not constant
    return nullptr;
  }

  assert(hasStaticShape(getSqueezed().getType()) &&
         "Shape should be static when the inputs are constant");

  OnnxElementsAttrBuilder elementsBuilder(getContext());
  return elementsBuilder.reshape(mlir::cast<ElementsAttr>(adaptor.getData()),
      getShape(getSqueezed().getType()));
}

OpFoldResult ONNXSqueezeV11Op::fold(FoldAdaptor adaptor) {
  // Fold the type of tensor
  if (failed(inferShapes(nullptr)))
    return nullptr;

  // Fold the value in tensor
  if (!adaptor.getData()) {
    // Use original Op if Data is not constant
    return nullptr;
  }

  assert(hasStaticShape(getSqueezed().getType()) &&
         "Shape should be static when the inputs are constant");

  OnnxElementsAttrBuilder elementsBuilder(getContext());
  return elementsBuilder.reshape(mlir::cast<ElementsAttr>(adaptor.getData()),
      getShape(getSqueezed().getType()));
}
