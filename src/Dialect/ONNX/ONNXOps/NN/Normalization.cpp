/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Normalization.cpp - ONNX Operations ---------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
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
  auto inputTensorTy = mlir::cast<RankedTensorType>(getX().getType());
  auto scaleTensorTy = mlir::cast<RankedTensorType>(getScale().getType());
  auto biasTensorTy = mlir::cast<RankedTensorType>(getB().getType());
  auto meanTensorTy = mlir::cast<RankedTensorType>(getMean().getType());
  auto varianceTensorTy = mlir::cast<RankedTensorType>(getVar().getType());

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
  Type elementType =
      mlir::cast<RankedTensorType>(getX().getType()).getElementType();
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
  auto inputType = mlir::cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();
  auto inputElementType = inputType.getElementType();
  int64_t spatialRank = inputShape.size() - 2;
  // If ranked, verify ranks of inputs.
  if (spatialRank < 1)
    return emitOpError("Spatial rank must be strictly positive");

  // Check bias B.
  if (hasShapeAndRank(B)) {
    // Can check at this stage.
    auto bType = mlir::cast<ShapedType>(B.getType());
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
    auto scaleType = mlir::cast<ShapedType>(scale.getType());
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

//===----------------------------------------------------------------------===//
// GroupNormalizationV18
//===----------------------------------------------------------------------===//
LogicalResult ONNXGroupNormalizationV18Op::verify() {
  ONNXGroupNormalizationV18OpAdaptor(*this);
  llvm::outs()
      << "\nWarning: The previous understanding of Opset 18 for "
         "GroupNormalization "
         "is incorrect. As shown in the following issue: "
         "https://github.com/onnx/onnx/issues/5466.Rather, use Opset 21 for "
         "GroupNormalization instead."
      << "\n\n";
  return success();
}

// TODO: should there be a shape inference for this one?

//===----------------------------------------------------------------------===//
// Generic LayerNormalization (LN)
//===----------------------------------------------------------------------===//

template <class OP_TYPE>
LogicalResult verifyShapeForLayerNorm(OP_TYPE *op) {
  typename OP_TYPE::Adaptor operandAdaptor(*op);

  // Get attributes.
  int64_t axis = op->getAxis();

  // Get operands.
  Value X = operandAdaptor.getX();
  Value scale = operandAdaptor.getScale();
  Value B = operandAdaptor.getB();

  // Check X.
  if (!hasShapeAndRank(X)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  ShapedType XType = mlir::cast<ShapedType>(X.getType());
  ArrayRef<int64_t> XShape = XType.getShape();
  int64_t XRank = XShape.size();
  Type XElementType = XType.getElementType();

  // Axis attribute (if specified) must be in the range [-r,r), where r =
  // rank(input).
  if (!isAxisInRange(axis, XRank))
    return op->emitOpError("axis must be in [-r, r) range]");

  // Check bias B.
  if (hasShapeAndRank(B)) {
    // Can check at this stage.
    ShapedType bType = mlir::cast<ShapedType>(B.getType());
    ArrayRef<int64_t> bShape = bType.getShape();
    SmallVector<int64_t> BBroadcastShape;
    if (!OpTrait::util::getBroadcastedShape(XShape, bShape, BBroadcastShape))
      op->emitOpError(
          "LayerNormalization op with incompatible B shapes (broadcast)");
    if (static_cast<int64_t>(BBroadcastShape.size()) != XRank)
      op->emitOpError("LayerNormalization op with incompatible B shapes "
                      "(unidirectional broadcast)");
    if (bType.getElementType() != XElementType)
      op->emitOpError("LayerNormalization op with incompatible B type");
  }

  // Check scale.
  if (hasShapeAndRank(scale)) {
    // Can check at this stage.
    ShapedType scaleType = mlir::cast<ShapedType>(scale.getType());
    ArrayRef<int64_t> scaleShape = scaleType.getShape();
    SmallVector<int64_t> scaleBroadcastShape;
    if (!OpTrait::util::getBroadcastedShape(
            XShape, scaleShape, scaleBroadcastShape))
      op->emitOpError(
          "LayerNormalization op with incompatible scale shapes (broadcast)");
    if (static_cast<int64_t>(scaleBroadcastShape.size()) != XRank)
      op->emitOpError("LayerNormalization op with incompatible scale shapes "
                      "(unidirectional broadcast)");
    if (scaleType.getElementType() != XElementType)
      op->emitOpError("LayerNormalization op with incompatible scale type");
  }

  return success();
}

namespace onnx_mlir {

template <typename OP_TYPE>
mlir::LogicalResult ONNXLNOpShapeHelper<OP_TYPE>::computeShape() {
  typename OP_TYPE::Adaptor operandAdaptor(operands);
  OP_TYPE lnOp = llvm::cast<OP_TYPE>(op);

  // Get rank and axis attribute.
  Value X = operandAdaptor.getX();
  int64_t XRank = mlir::cast<ShapedType>(X.getType()).getRank();
  int64_t axis = getAxisInRange(lnOp.getAxis(), XRank);

  // Check optional outputs, with specialization for ONNXLayerNormalizationOp
  // and ONNXRMSLayerNormalizationOp.
  bool hasMean, hasInvStdDev;
  int64_t invStdDevIndex;
  if constexpr (std::is_same<OP_TYPE, ONNXLayerNormalizationOp>::value) {
    hasMean = !isNoneValue(lnOp.getMean());
    invStdDevIndex = 2;
  } else if constexpr (std::is_same<OP_TYPE,
                           ONNXRMSLayerNormalizationOp>::value) {
    hasMean = false;
    invStdDevIndex = 1;
  } else {
    llvm_unreachable("unknown type");
  }
  hasInvStdDev = !isNoneValue(lnOp.getInvStdDev());

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
  if (hasMean) {
    DimsExpr meanShape(getOutputDims(0));
    for (int64_t r = axis; r < XRank; ++r)
      meanShape[r] = LitIE(1);
    setOutputDims(meanShape, 1, false);
  }

  // Compute invStdDev output shape if requested.
  if (hasInvStdDev) {
    DimsExpr invStdDevShape(getOutputDims(0));
    for (int64_t r = axis; r < XRank; ++r)
      invStdDevShape[r] = LitIE(1);
    setOutputDims(invStdDevShape, invStdDevIndex, false);
  }
  return success();
}
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// LayerNormalization
//===----------------------------------------------------------------------===//

LogicalResult ONNXLayerNormalizationOp::verify() {
  return verifyShapeForLayerNorm<ONNXLayerNormalizationOp>(this);
}

LogicalResult ONNXLayerNormalizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // If any input is not ranked tensor, do nothing. Account for possibly null
  // inputs (B).
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getScale()) ||
      (!isNoneValue(getB()) && !hasShapeAndRank(getB())))
    return success();
  Type commonType =
      mlir::cast<RankedTensorType>(getX().getType()).getElementType();
  ONNXLayerNormalizationOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(commonType);
}

//===----------------------------------------------------------------------===//
// RMSLayerNormalization (Additional ONNX Op)
//===----------------------------------------------------------------------===//

LogicalResult ONNXRMSLayerNormalizationOp::verify() {
  return verifyShapeForLayerNorm<ONNXRMSLayerNormalizationOp>(this);
}

LogicalResult ONNXRMSLayerNormalizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // If any input is not ranked tensor, do nothing. Account for possibly null
  // inputs (B).
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getScale()) ||
      (!isNoneValue(getB()) && !hasShapeAndRank(getB())))
    return success();
  Type commonType =
      mlir::cast<RankedTensorType>(getX().getType()).getElementType();
  ONNXRMSLayerNormalizationOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(commonType);
}
