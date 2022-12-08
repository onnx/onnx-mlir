/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ SqueezeUnsqueeze.cpp - ONNX Operations ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Squeeze and Unsqueeze
// operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
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
LogicalResult NewONNXCommonSqueezeOpShapeHelper<OP_TYPE>::customComputeShape(
    DimsExpr &squeezedDims, bool axesFromShape) {
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  DimsExpr outputDims;
  Value data = operandAdaptor.data();
  int64_t dataRank = createIE->getTypeRank(data);

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
      assert(shape >= 0 && "Compile time shape should be nonnegative");
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
void NewONNXSqueezeOpShapeHelper::saveAxes() {
  // Create a ConstantOp associated with this Squeeze Op
  // There could be an issue if we were to generate a constant Op late in
  // lowering, but since we normalize them during the first shape inference, we
  // should never encounter a "saveAxles" situation during lowering.

  ONNXSqueezeOp squeezeOp = llvm::cast<ONNXSqueezeOp>(op);
  SaveOnnxConstInOp(op, squeezeOp.axesMutable(), squeezedAxes);
}

template <>
void NewONNXSqueezeV11OpShapeHelper::saveAxes() {
  SaveOnnxAttrInOp<ONNXSqueezeV11Op>(op, squeezedAxes,
      [](ONNXSqueezeV11Op op, ArrayAttr attr) { op.axesAttr(attr); });
}

template <>
LogicalResult NewONNXSqueezeOpShapeHelper::computeShape() {
  ONNXSqueezeOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  Value axes = operandAdaptor.axes();
  SmallVector<IndexExpr, 4> squeezedDims;
  bool squeezeFromShape = false;
  if (axes.getType().template isa<NoneType>())
    squeezeFromShape = true;
  else
    createIE->getIntFromArrayAsSymbols(axes, squeezedDims);
  return customComputeShape(squeezedDims, squeezeFromShape);
}

template <>
LogicalResult NewONNXSqueezeV11OpShapeHelper::computeShape() {
  ONNXSqueezeV11OpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  auto axesAttr = operandAdaptor.axesAttr();
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
    std::function<void(mlir::Region &)> doShapeInference) {
  auto dataType = data().getType().dyn_cast<RankedTensorType>();
  if (!dataType)
    return success();

  Type elementType = dataType.getElementType();
  NewONNXSqueezeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

LogicalResult ONNXSqueezeV11Op::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto dataType = data().getType().dyn_cast<RankedTensorType>();
  if (!dataType)
    return success();

  Type elementType = dataType.getElementType();
  NewONNXSqueezeV11OpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct NewONNXCommonSqueezeOpShapeHelper<ONNXSqueezeOp>;
template struct NewONNXCommonSqueezeOpShapeHelper<ONNXSqueezeV11Op>;
} // namespace onnx_mlir
