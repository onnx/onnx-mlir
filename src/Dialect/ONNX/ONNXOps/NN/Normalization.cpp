/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Normalization.cpp - ONNX Operations ---------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Normalization operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// BatchNormalizationInferenceMode
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXBatchNormalizationInferenceModeOpShapeHelper::computeShape() {
  // Single output in inference mode, Y same shape as X.
  ONNXBatchNormalizationInferenceModeOpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getX());
}

} // namespace onnx_mlir

LogicalResult ONNXBatchNormalizationInferenceModeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getScale()) ||
      !hasShapeAndRank(getB()) || !hasShapeAndRank(getMean()) ||
      !hasShapeAndRank(getVar()))
    return success();

  // Verifier code.
  auto inputTensorTy = getX().getType().cast<RankedTensorType>();
  auto scaleTensorTy = getScale().getType().cast<RankedTensorType>();
  auto biasTensorTy = getB().getType().cast<RankedTensorType>();
  auto meanTensorTy = getMean().getType().cast<RankedTensorType>();
  auto varianceTensorTy = getVar().getType().cast<RankedTensorType>();

  // Check whether the shapes of scale, bias, mean and variance are valid.
  // Operand's dimensions can be in the form of NxCxD1xD2x...xDn or N.
  // In case of N, C is assumed to be 1.
  // 2-D tensors are assumed to be of shape NxC
  // Shapes of scale, bias, mean and variance must be C.
  int64_t c = ShapedType::kDynamic;
  if (inputTensorTy.getShape().size() == 1) {
    c = 1;
  } else if (inputTensorTy.getShape().size() >= 2) {
    c = (!inputTensorTy.isDynamicDim(1)) ? inputTensorTy.getShape()[1]
                                         : ShapedType::kDynamic;
  }

  if (!ShapedType::isDynamic(c)) {
    auto s = scaleTensorTy.getShape();
    auto b = biasTensorTy.getShape();
    auto m = meanTensorTy.getShape();
    auto v = varianceTensorTy.getShape();

    if ((s.size() != 1) || (!ShapedType::isDynamic(s[0]) && s[0] != c))
      return emitError("Wrong rank for the scale");
    if ((b.size() != 1) || (!ShapedType::isDynamic(b[0]) && b[0] != c))
      return emitError("Wrong rank for the bias");
    if ((m.size() != 1) || (!ShapedType::isDynamic(m[0]) && m[0] != c))
      return emitError("Wrong rank for the mean");
    if ((v.size() != 1) || (!ShapedType::isDynamic(v[0]) && v[0] != c))
      return emitError("Wrong rank for the variance");
  }

  // The output tensor of the same shape as the input.
  Type elementType = getX().getType().cast<RankedTensorType>().getElementType();
  ONNXBatchNormalizationInferenceModeOpShapeHelper shapeHelper(
      getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<
    ONNXBatchNormalizationInferenceModeOp>;
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// InstanceNormalization
//===----------------------------------------------------------------------===//

LogicalResult ONNXInstanceNormalizationOp::verify() {
  ONNXInstanceNormalizationOpAdaptor operandAdaptor =
      ONNXInstanceNormalizationOpAdaptor(*this);
  // Get operands.
  auto input = operandAdaptor.getInput();
  auto scale = operandAdaptor.getScale();
  auto B = operandAdaptor.getB();

  // Check input.
  if (!hasShapeAndRank(input)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  auto inputType = input.getType().cast<ShapedType>();
  auto inputShape = inputType.getShape();
  auto inputElementType = inputType.getElementType();
  int64_t spatialRank = inputShape.size() - 2;
  // If ranked, verify ranks of inputs.
  if (spatialRank < 1)
    return emitOpError("Spatial rank must be strictly positive");

  // Check bias B.
  if (hasShapeAndRank(B)) {
    // Can check at this stage.
    auto bType = B.getType().cast<ShapedType>();
    auto bShape = bType.getShape();
    if (bShape.size() != 1)
      return emitOpError("Bias should have a rank of one");
    if (bShape[0] != ShapedType::kDynamic &&
        inputShape[1] != ShapedType::kDynamic && bShape[0] != inputShape[1])
      return emitOpError(
          "Bias should have same dimension as the second dimension of input");
    if (bType.getElementType() != inputElementType)
      return emitOpError("Bias should have same element type as input");
  }

  // Check scale.
  if (hasShapeAndRank(scale)) {
    // Can check at this stage.
    auto scaleType = scale.getType().cast<ShapedType>();
    auto scaleShape = scaleType.getShape();
    if (scaleShape.size() != 1)
      return emitOpError("Scale should have a rank of one");
    if (scaleShape[0] != ShapedType::kDynamic &&
        inputShape[1] != ShapedType::kDynamic && scaleShape[0] != inputShape[1])
      return emitOpError(
          "Scale should have same dimension as the second dimension of input");
    if (scaleType.getElementType() != inputElementType)
      return emitOpError("Scale should have same element type as input");
  }

  return success();
}

// TODO: should there be a shape inference for this one?

//===----------------------------------------------------------------------===//
// LayerNormalization
//===----------------------------------------------------------------------===//

LogicalResult ONNXLayerNormalizationOp::verify() {
  auto operandAdaptor = ONNXLayerNormalizationOpAdaptor(*this);

  // Get attributes.
  int64_t axis = getAxis();

  // Get operands.
  Value X = operandAdaptor.getX();
  Value scale = operandAdaptor.getScale();
  Value B = operandAdaptor.getB();

  // Check X.
  if (!hasShapeAndRank(X)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  ShapedType XType = X.getType().cast<ShapedType>();
  ArrayRef<int64_t> XShape = XType.getShape();
  int64_t XRank = XShape.size();
  Type XElementType = XType.getElementType();

  // Axis attribute (if specified) must be in the range [-r,r), where r =
  // rank(input).
  if (!isAxisInRange(axis, XRank))
    return emitOpError("axis must be in [-r, r) range]");

  // Check bias B.
  if (hasShapeAndRank(B)) {
    // Can check at this stage.
    ShapedType bType = B.getType().cast<ShapedType>();
    ArrayRef<int64_t> bShape = bType.getShape();
    SmallVector<int64_t> BBroadcastShape;
    if (!OpTrait::util::getBroadcastedShape(XShape, bShape, BBroadcastShape))
      emitOpError(
          "LayerNormalization op with incompatible B shapes (broadcast)");
    if ((int64_t)BBroadcastShape.size() != XRank)
      emitOpError("LayerNormalization op with incompatible B shapes "
                  "(unidirectional broadcast)");
    if (bType.getElementType() != XElementType)
      emitOpError("LayerNormalization op with incompatible B type");
  }

  // Check scale.
  if (hasShapeAndRank(scale)) {
    // Can check at this stage.
    ShapedType scaleType = scale.getType().cast<ShapedType>();
    ArrayRef<int64_t> scaleShape = scaleType.getShape();
    SmallVector<int64_t> scaleBroadcastShape;
    if (!OpTrait::util::getBroadcastedShape(
            XShape, scaleShape, scaleBroadcastShape))
      emitOpError(
          "LayerNormalization op with incompatible scale shapes (broadcast)");
    if ((int64_t)scaleBroadcastShape.size() != XRank)
      emitOpError("LayerNormalization op with incompatible scale shapes "
                  "(unidirectional broadcast)");
    if (scaleType.getElementType() != XElementType)
      emitOpError("LayerNormalization op with incompatible scale type");
  }

  return success();
}

namespace onnx_mlir {

mlir::LogicalResult ONNXLayerNormalizationOpShapeHelper::computeShape() {

  ONNXLayerNormalizationOpAdaptor operandAdaptor(operands);
  ONNXLayerNormalizationOp lnOp = llvm::cast<ONNXLayerNormalizationOp>(op);

  // Get rank and axis attribute.
  Value X = operandAdaptor.getX();
  int64_t XRank = X.getType().cast<ShapedType>().getRank();
  int64_t axis = getAxisInRange(lnOp.getAxis(), XRank);

  // Compute the shape of the first output and all the inputs.
  llvm::SmallVector<Value, 3> operandsForBroadcast;
  operandsForBroadcast.emplace_back(X);
  operandsForBroadcast.emplace_back(operandAdaptor.getScale());
  if (!isNoneValue(operandAdaptor.getB()))
    operandsForBroadcast.emplace_back(operandAdaptor.getB());
  if (failed(ONNXBroadcastOpShapeHelper::customComputeShape(
          operandsForBroadcast, nullptr)))
    return failure();

  // Compute mean output shape if requested.
  if (!isNoneValue(lnOp.getMean())) {
    DimsExpr meanShape(getOutputDims(0));
    for (int64_t r = axis; r < XRank; ++r)
      meanShape[r] = LiteralIndexExpr(1);
    setOutputDims(meanShape, 1, false);
  }
  // Compute invStdDev output shape if requested.
  if (!isNoneValue(lnOp.getInvStdDev())) {
    DimsExpr invStdDevShape(getOutputDims(0));
    for (int64_t r = axis; r < XRank; ++r)
      invStdDevShape[r] = LiteralIndexExpr(1);
    setOutputDims(invStdDevShape, 2, false);
  }
  return success();
}
} // namespace onnx_mlir

LogicalResult ONNXLayerNormalizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // If any input is not ranked tensor, do nothing. Account for possibly null
  // inputs (B).
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getScale()) ||
      (!isNoneValue(getB()) && !hasShapeAndRank(getB())))
    return success();
  Type commonType = getX().getType().cast<RankedTensorType>().getElementType();
  ONNXLayerNormalizationOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(commonType);
}
