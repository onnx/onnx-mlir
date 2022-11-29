/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Expand.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Expand operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

using namespace mlir;

namespace onnx_mlir {


LogicalResult NewONNXExpandOpShapeHelper::computeShape() {
  // Get info about input operands.
  ONNXExpandOpAdaptor operandAdaptor(operands);
  Value input = operandAdaptor.input();
  Value shape = operandAdaptor.shape();

  Operation *shapeDefOp = shape.getDefiningOp();
  ShapedType shapeType = shape.getType().dyn_cast_or_null<ShapedType>();
  if (!shapeType)
    return op->emitError("expected shape parameter to be defined");
  if (ShapedType::isDynamic(shapeType.getShape()[0]))
    return op->emitError("expected size of shape parameter to be defined");

  if (ONNXShapeOp shapeOp = dyn_cast_or_null<ONNXShapeOp>(shapeDefOp)) {
    assert(shapeOp.data().getType().isa<ShapedType>() && "expected");
    // Consider a first case where the expand.shape is produced by a shape op.
    // Infer its shape and use it as the requested shape.
    // Compute the output of the shape operation. We have to use its shape
    // helper as we need to connect to the actual expressions used to compute
    // it, not just a shape, in presence of runtime dimensions.

    // Use the full constructor as this is called from shape helper which may
    // be used in either shape inference or lowering to ONNX context. We also
    // pass here the scope of the ExpandOp shape helper so that the
    // computations performed in the ShapeOp shape helper can be used in the
    // context of the ExpandOp.
    NewONNXShapeOpShapeHelper shapeOpShapeHelper(
        shapeOp.getOperation(), {}, createIE);
    ONNXShapeOpAdaptor shapeOpOperandAdaptor(shapeOp);
    if (failed(shapeOpShapeHelper.computeShape()))
      return op->emitError("failed to get shape op shape");

    // Compute the data selected by the Shape operator.
    DimsExpr selectedData = computeSelectedData(shapeOpOperandAdaptor);

    // Now that we have the shape's actual computation
    if (failed(NewONNXOpBroadcastedShapeHelper::customComputeShape(
            {input}, &selectedData)))
      return op->emitError("failed to broadcast 3");

    return success();
  }

  if (!shape.getType().isa<ShapedType>())
    return op->emitError("Expecting a shaped type");
  SmallVector<IndexExpr, 4> constVals;
  createIE->getIntArrayAsSymbols(shape, constVals);
  if (failed(NewONNXOpBroadcastedShapeHelper::customComputeShape(
          {input}, &constVals)))
    return op->emitError("failed to broadcast 4");

  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXExpandOp::verify() {
  ONNXExpandOpAdaptor operandAdaptor = ONNXExpandOpAdaptor(*this);
  // Get operands.
  auto shape = operandAdaptor.shape();
  // Check input.
  auto shapeType = shape.getType().dyn_cast_or_null<ShapedType>();
  if (shapeType && shapeType.hasRank()) {
    if (shapeType.getRank() != 1)
      return emitOpError("Shape has a rank of 1");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXExpandOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!input().getType().isa<RankedTensorType>())
    return success();
  if (!shape().getType().isa<RankedTensorType>())
    return success();

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  IndexExprBuilderForAnalysis createIE(getLoc());
  NewONNXExpandOpShapeHelper shapeHelper(getOperation(), {}, &createIE);
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
