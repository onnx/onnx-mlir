/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ElementwiseUnary.cpp - ONNX Operations ------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Elementwise Unary operation.
//
// Please add operations in alphabetical order.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/Mlir/IndexExprBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

/// Handle shape inference for unary element-wise operators.
LogicalResult inferShapeForUnaryOps(Operation *op) {
  Value input = op->getOperand(0);
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  return inferShapeForUnaryOps(
      op, inputType.getElementType(), inputType.getEncoding());
}

/// Handle shape inference for unary element-wise operators with specific output
/// type.
LogicalResult inferShapeForUnaryOps(Operation *op, Type elementType) {
  Value input = op->getOperand(0);
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  return inferShapeForUnaryOps(op, elementType, inputType.getEncoding());
}

/// Handle shape inference for unary element-wise operators with specific output
/// type and encoding.
LogicalResult inferShapeForUnaryOps(
    Operation *op, Type elementType, Attribute encoding) {
  Value input = op->getOperand(0);
  if (!hasShapeAndRank(input))
    return success();

  ONNXUnaryOpShapeHelper shapeHelper(op, {});
  return shapeHelper.computeShapeAndUpdateType(elementType, encoding);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXAbsOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Acos
//===----------------------------------------------------------------------===//

LogicalResult ONNXAcosOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Acosh
//===----------------------------------------------------------------------===//

LogicalResult ONNXAcoshOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Asin
//===----------------------------------------------------------------------===//

LogicalResult ONNXAsinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Asinh
//===----------------------------------------------------------------------===//

LogicalResult ONNXAsinhOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Atan
//===----------------------------------------------------------------------===//

LogicalResult ONNXAtanOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Atanh
//===----------------------------------------------------------------------===//

LogicalResult ONNXAtanhOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// BitwiseNot
//===----------------------------------------------------------------------===//

LogicalResult ONNXBitwiseNotOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Cast
//===----------------------------------------------------------------------===//

LogicalResult ONNXCastOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();

  Type elementType = (*this)->getAttr("to").cast<::TypeAttr>().getValue();
  ONNXCastOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// CastLike
//===----------------------------------------------------------------------===//

LogicalResult ONNXCastLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();

  Type elementType = getElementType(getTargetType().getType());
  ONNXCastLikeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Ceil
//===----------------------------------------------------------------------===//

LogicalResult ONNXCeilOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Celu
//===----------------------------------------------------------------------===//

LogicalResult ONNXCeluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Cos
//===----------------------------------------------------------------------===//

LogicalResult ONNXCosOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Cosh
//===----------------------------------------------------------------------===//

LogicalResult ONNXCoshOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// CumSum
//===----------------------------------------------------------------------===//

LogicalResult ONNXCumSumOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Elu
//===----------------------------------------------------------------------===//

LogicalResult ONNXEluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ErfOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXErfOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Exp
//===----------------------------------------------------------------------===//

LogicalResult ONNXExpOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Floor
//===----------------------------------------------------------------------===//

LogicalResult ONNXFloorOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// HardSigmoid
//===----------------------------------------------------------------------===//

LogicalResult ONNXHardSigmoidOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// HardSwishOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXHardSwishOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// InstanceNormalization
//===----------------------------------------------------------------------===//

LogicalResult ONNXInstanceNormalizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// LeakyRelu
//===----------------------------------------------------------------------===//

LogicalResult ONNXLeakyReluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Log
//===----------------------------------------------------------------------===//

LogicalResult ONNXLogOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// LogSoftmax
//===----------------------------------------------------------------------===//

LogicalResult ONNXLogSoftmaxOp::verify() {
  ONNXLogSoftmaxOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(operandAdaptor.getInput()))
    return success(); // Won't be able to do any checking at this stage.

  int64_t inputRank =
      operandAdaptor.getInput().getType().cast<ShapedType>().getRank();
  int64_t axisIndex = getAxis();

  // axis attribute must be in the range [-r,r-1], where r = rank(input).
  if (axisIndex < -inputRank || axisIndex >= inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));

  return success();
}

LogicalResult ONNXLogSoftmaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// LpNormalization
//===----------------------------------------------------------------------===//

LogicalResult ONNXLpNormalizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// MeanVarianceNormalization
//===----------------------------------------------------------------------===//

LogicalResult ONNXMeanVarianceNormalizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXNegOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Not
//===----------------------------------------------------------------------===//

LogicalResult ONNXNotOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ReciprocalOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXReciprocalOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Relu
//===----------------------------------------------------------------------===//

LogicalResult ONNXReluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Round
//===----------------------------------------------------------------------===//

LogicalResult ONNXRoundOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Scaler
//===----------------------------------------------------------------------===//

LogicalResult ONNXScalerOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()))
    return success();

  ONNXUnaryOpShapeHelper shapeHelper(getOperation(), {});
  RankedTensorType xType = getX().getType().dyn_cast<RankedTensorType>();
  return shapeHelper.computeShapeAndUpdateType(
      FloatType::getF32(getContext()), xType.getEncoding());
}

//===----------------------------------------------------------------------===//
// Selu
//===----------------------------------------------------------------------===//

LogicalResult ONNXSeluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Shrink
//===----------------------------------------------------------------------===//

LogicalResult ONNXShrinkOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Sigmoid
//===----------------------------------------------------------------------===//

LogicalResult ONNXSigmoidOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SignOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSignOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Sin
//===----------------------------------------------------------------------===//

LogicalResult ONNXSinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Sinh
//===----------------------------------------------------------------------===//

LogicalResult ONNXSinhOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSoftmaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SoftmaxV11Op
//===----------------------------------------------------------------------===//

LogicalResult ONNXSoftmaxV11Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SoftplusOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSoftplusOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SoftsignOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSoftsignOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSqrtOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Tan
//===----------------------------------------------------------------------===//

LogicalResult ONNXTanOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Tanh
//===----------------------------------------------------------------------===//

LogicalResult ONNXTanhOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ThresholdedRelu
//===----------------------------------------------------------------------===//

LogicalResult ONNXThresholdedReluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// TriluOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXTriluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}
