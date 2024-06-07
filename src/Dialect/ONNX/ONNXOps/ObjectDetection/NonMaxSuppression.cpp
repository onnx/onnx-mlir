/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- NonMaxSuppression.cpp - ONNX Operations  -------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect NonMaxSuppression operation.
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
LogicalResult ONNXNonMaxSuppressionOpShapeHelper::computeShape() {
  // Three is a backend test where the result of ONNXNonMaxSuppressionOp is set
  // to 1 (first dim), where in fact it the size must be ShapedType::kDynamic
  // since its data dependent. Thus disable the refineDims because of this test
  // case.
  return setOutputDimsFromLiterals(
      {ShapedType::kDynamic, 3}, 0, /*refineDims*/ false);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXNonMaxSuppressionOp::verify() {
  ONNXNonMaxSuppressionOpAdaptor operandAdaptor =
      ONNXNonMaxSuppressionOpAdaptor(*this);
  // Get operands.
  auto boxes = operandAdaptor.getBoxes();
  auto scores = operandAdaptor.getScores();
  auto MOPC = operandAdaptor.getMaxOutputBoxesPerClass();
  auto scoreThreshold = operandAdaptor.getScoreThreshold();
  auto iouThreshold = operandAdaptor.getIouThreshold();

  // Check operands.
  if (hasShapeAndRank(boxes)) {
    auto shape = mlir::cast<ShapedType>(boxes.getType()).getShape();
    if (shape.size() != 3)
      return emitOpError("boxes should have a rank of three");
    if (!ShapedType::isDynamic(shape[2]) && shape[2] != 4)
      return emitOpError("The last dim of Boxes should be four");
  }

  if (hasShapeAndRank(scores))
    if (mlir::cast<ShapedType>(scores.getType()).getRank() != 3)
      return emitOpError("scores should have a rank of three");

  if (hasShapeAndRank(MOPC))
    if (mlir::cast<ShapedType>(MOPC.getType()).getRank() > 1)
      return emitOpError(
          "max_output_boxex_per_class should have a rank of zero or one");

  if (hasShapeAndRank(scoreThreshold))
    if (mlir::cast<ShapedType>(scoreThreshold.getType()).getRank() > 1)
      return emitOpError("score_threshold should have a rank of zero or one");

  if (hasShapeAndRank(iouThreshold))
    if (mlir::cast<ShapedType>(iouThreshold.getType()).getRank() > 1)
      return emitOpError("iou_threshold should have a rank of zero or one");

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXNonMaxSuppressionOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Builder b = Builder(getContext());
  Type elementType = b.getI64Type();
  ONNXNonMaxSuppressionOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXNonMaxSuppressionOp>;
} // namespace onnx_mlir
