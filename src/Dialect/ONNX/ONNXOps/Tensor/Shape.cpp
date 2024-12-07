/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Shape.cpp - ONNX Operations -----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Shape operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

namespace {

// The Shape op spec says:
//
// "Note that axes will be clipped to the range [0, r-1], where r is the
// rank of the input tensor if they are out-of-range (after adding r in the case
// of negative axis). Thus, specifying any end value > r is equivalent to
// specifying an end value of r, and specifying any start value < -r is
// equivalent to specifying a start value of 0."
static int64_t normalizeClampedPerSpec(int64_t axis, int64_t rank) {
  if (axis < 0)
    axis += rank;
  if (axis < 0)
    axis = 0;
  if (axis > rank)
    axis = rank;
  return axis;
}

} // namespace

LogicalResult ONNXShapeOpShapeHelper::computeShape() {
  ONNXShapeOp shapeOp = llvm::cast<ONNXShapeOp>(op);
  ONNXShapeOpAdaptor operandAdaptor(operands);
  Value data = operandAdaptor.getData();

  // Compute and store start/end in ONNXShapeOpShapeHelper object.
  if (!hasShapeAndRank(data)) {
    return failure();
  }
  int64_t rank = createIE->getShapedTypeRank(data);
  start = shapeOp.getStart();
  start = normalizeClampedPerSpec(start, rank);
  end = shapeOp.getEnd().has_value() ? shapeOp.getEnd().value() : rank;
  end = normalizeClampedPerSpec(end, rank);
  if (start > end)
    return op->emitError("Start must not be greater than end");

  // Output shape is a 1D vector with "end-start" values
  DimsExpr outputDims(1, LitIE(end - start));
  setOutputDims(outputDims);
  return success();
}

void ONNXShapeOpShapeHelper::computeSelectedDataShape(
    DimsExpr &selectedDataShape) {
  assert(start != -1 && end != -1 && "must compute shape first");
  ONNXShapeOpAdaptor operandAdaptor(operands);
  Value data = operandAdaptor.getData();

  selectedDataShape.clear();
  for (int64_t i = start; i < end; ++i)
    selectedDataShape.emplace_back(createIE->getShapeAsDim(data, i));
}

/* static */ void ONNXShapeOpShapeHelper::getStartEndValues(
    ONNXShapeOp shapeOp, int64_t &startVal, int64_t &endVal) {
  // Get rank of data operand.
  ONNXShapeOpAdaptor operandAdaptor(shapeOp);
  Value data = operandAdaptor.getData();
  ShapedType shapedType = mlir::dyn_cast_or_null<ShapedType>(data.getType());
  assert(shapedType && shapedType.hasRank() && "need shaped type with rank");
  int64_t rank = shapedType.getRank();
  // Compute the normalized start/end. Negative value means counting
  // dimensions from the back.
  startVal = operandAdaptor.getStart();
  startVal = normalizeClampedPerSpec(startVal, rank);
  endVal = operandAdaptor.getEnd().has_value() ? operandAdaptor.getEnd().value()
                                               : rank;
  endVal = normalizeClampedPerSpec(endVal, rank);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXShapeOp::verify() {
  if (!hasShapeAndRank(getData()))
    return success();
  int64_t start, end;
  ONNXShapeOpShapeHelper::getStartEndValues(*this, start, end);
  if (start > end)
    return emitOpError() << "Start: " << start << " is after End: " << end;
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXShapeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getData()))
    return success();

  // Output is an 1D int64 tensor containing the shape of the input tensor.
  Type elementType = IntegerType::get(getContext(), 64);
  ONNXShapeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
