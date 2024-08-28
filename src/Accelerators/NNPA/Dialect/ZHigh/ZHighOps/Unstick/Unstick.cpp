/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Unstick.cpp - ZHigh Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/Unstick/ONNXZHighUnstick.inc"
} // end anonymous namespace

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Custom builders
//===----------------------------------------------------------------------===//

void ZHighUnstickOp::build(
    OpBuilder &builder, OperationState &state, Value input) {
  Type resType;
  Type resElementType = builder.getF32Type();
  ShapedType inputType = mlir::cast<ShapedType>(input.getType());
  if (hasRankedType(input)) {
    // Compute shape.
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<int64_t, 4> resShape(inputShape.begin(), inputShape.end());
    // Direct unstickify from NHWC to NCHW.
    StringAttr layout = convertZTensorDataLayoutToStringAttr(
        builder, getZTensorLayout(input.getType()));
    if (isNHWCLayout(layout)) {
      assert((inputShape.size() == 4) && "Input must have rank 4");
      // NHWC -> NCHW
      resShape[0] = inputShape[0];
      resShape[1] = inputShape[3];
      resShape[2] = inputShape[1];
      resShape[3] = inputShape[2];
    }
    resType = RankedTensorType::get(resShape, resElementType);
  } else
    resType = UnrankedTensorType::get(resElementType);
  build(builder, state, resType, input);
}

//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

LogicalResult ZHighUnstickOpShapeHelper::computeShape() {
  ZHighUnstickOp::Adaptor operandAdaptor(operands);
  Value input = operandAdaptor.getIn();

  // Output dims of result.
  DimsExpr outputDims;

  // Get layout attribute. Do not get it from the input in OpAdaptor since
  // that input is the converted type, i.e. MemRefType. Get directly from
  // Operation instead where the type is TensorType that has the layout
  // encoding attribute.
  OpBuilder b(op);
  StringAttr layout = getZTensorLayoutAttr(b, op->getOperand(0).getType());

  // Get operands and bounds.
  SmallVector<IndexExpr, 4> inputDims;
  createIE->getShapeAsDims(input, inputDims);
  int64_t rank = inputDims.size();

  for (int64_t i = 0; i < rank; ++i)
    outputDims.emplace_back(inputDims[i]);

  // Direct unstickify from NHWC to NCHW.
  if (isNHWCLayout(layout)) {
    assert((rank == 4) && "Unstickify input must have rank 4");
    // NHWC -> NCHW
    outputDims[0] = inputDims[0];
    outputDims[1] = inputDims[3];
    outputDims[2] = inputDims[1];
    outputDims[3] = inputDims[2];
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighUnstickOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasRankedType(getIn()))
    return success();

  ZHighUnstickOpShapeHelper shapeHelper(getOperation());
  Type elementType =
      mlir::cast<ShapedType>(getResult().getType()).getElementType();
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

void ZHighUnstickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<UnstickStickRemovalPattern>(context);
  results.insert<DimUnstickRemovalPattern>(context);
  results.insert<DimUnstickNHWCRemovalPattern>(context);
}

} // namespace zhigh
} // namespace onnx_mlir
