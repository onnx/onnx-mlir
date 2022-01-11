/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Expand.cpp - Shape Inference for Expand Op -------------===//
//
// This file implements shape inference for the ONNX Expand Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

ONNXExpandOpShapeHelper::ONNXExpandOpShapeHelper(ONNXExpandOp *newOp)
    : ONNXOpBroadcastedShapeHelper<ONNXExpandOp>(newOp), expandOp(newOp) {}

ONNXExpandOpShapeHelper::ONNXExpandOpShapeHelper(ONNXExpandOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpBroadcastedShapeHelper<ONNXExpandOp>(
          newOp, rewriter, fGetDenseVal, fLoadVal),
      expandOp(newOp) {}

LogicalResult ONNXExpandOpShapeHelper::computeShape(
    ONNXExpandOpAdaptor operandAdaptor) {
  // Get info about input operands.
  Value input = operandAdaptor.input();
  Value shape = operandAdaptor.shape();
  Operation *shapeDefOp = shape.getDefiningOp();

  ShapedType shapeType = shape.getType().dyn_cast_or_null<ShapedType>();
  if (!shapeType)
    return op->emitError("expected shape parameter to be defined");
  if (shapeType.getShape()[0] == -1)
    return op->emitError("expected size of shape parameter to be defined");
  if (mlir::ONNXShapeOp shapeOp =
          dyn_cast_or_null<mlir::ONNXShapeOp>(shapeDefOp)) {
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
    ONNXShapeOpShapeHelper shapeOpShapeHelper(
        &shapeOp, scope->getRewriterPtr(), fGetDenseVal, fLoadVal, scope);
    ONNXShapeOpAdaptor shapeOpOperandAdaptor(shapeOp);
    if (failed(shapeOpShapeHelper.computeShape(shapeOpOperandAdaptor)))
      return op->emitError("failed to get shape op shape");
    // Now that we have the shape's actual computation in
    if (failed(ONNXOpBroadcastedShapeHelper::computeShape(
            {input}, shapeOpShapeHelper.selectedData)))
      return op->emitError("failed to broadcast");

  } else {
    assert(shape.getType().isa<ShapedType>());
    SmallVector<IndexExpr, 4> constVals;
    ArrayValueIndexCapture arrayCapture(shape, fGetDenseVal, fLoadVal);
    if (!arrayCapture.getSymbolList(constVals)) {
      return op->emitError(
          "Shape argument of Expand is the output of an unexpected "
          "operation. Supported operations are: onnx.Constant and "
          "onnx.Shape");
    }
    if (failed(ONNXOpBroadcastedShapeHelper::computeShape({input}, constVals)))
      return op->emitError("failed to broadcast");
  }

  return success();
}
