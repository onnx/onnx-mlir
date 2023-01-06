/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ElementwiseUnary.cpp - ONNX Operations ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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

ONNXOpShapeHelper *ONNXAbsOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXAbsOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Acos
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXAcosOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXAcosOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Acosh
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXAcoshOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXAcoshOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Asin
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXAsinOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXAsinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Asinh
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXAsinhOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXAsinhOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Atan
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXAtanOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXAtanOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Atanh
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXAtanhOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXAtanhOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// BitwiseNot
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXBitwiseNotOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXBitwiseNotOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Cast
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXCastOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXCastOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ShapedType inputType = input().getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    return success();
  }

  auto getOutputType = [&inputType](Type elementType) -> Type {
    if (inputType.hasRank()) {
      return RankedTensorType::get(inputType.getShape(), elementType);
    }
    return UnrankedTensorType::get(elementType);
  };

  Type targetType = (*this)->getAttr("to").cast<::TypeAttr>().getValue();
  OpBuilder builder(getContext());
  getResult().setType(getOutputType(targetType));
  return success();
}

//===----------------------------------------------------------------------===//
// CastLike
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXCastLikeOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXCastLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ShapedType inputType = input().getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    return success();
  }

  TensorType targetType = target_type().getType().dyn_cast<TensorType>();
  if (!inputType) {
    return success();
  }
  auto targetElementType = targetType.getElementType();

  auto getOutputType = [&inputType](Type elementType) -> Type {
    if (inputType.hasRank()) {
      return RankedTensorType::get(inputType.getShape(), elementType);
    }
    return UnrankedTensorType::get(elementType);
  };

  getResult().setType(getOutputType(targetElementType));
  return success();
}

//===----------------------------------------------------------------------===//
// Ceil
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXCeilOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXCeilOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Celu
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXCeluOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXCeluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Cos
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXCosOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXCosOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Cosh
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXCoshOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXCoshOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// CumSum
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXCumSumOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXCumSumOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Elu
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXEluOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXEluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ErfOp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXErfOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXErfOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Exp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXExpOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXExpOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Floor
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXFloorOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXFloorOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// HardSigmoid
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXHardSigmoidOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXHardSigmoidOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// HardSwishOp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXHardSwishOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXHardSwishOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// InstanceNormalization
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXInstanceNormalizationOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXInstanceNormalizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// LeakyRelu
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXLeakyReluOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXLeakyReluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Log
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXLogOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXLogOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// LogSoftmax
//===----------------------------------------------------------------------===//

LogicalResult ONNXLogSoftmaxOp::verify() {
  ONNXLogSoftmaxOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(operandAdaptor.input()))
    return success(); // Won't be able to do any checking at this stage.

  int64_t inputRank =
      operandAdaptor.input().getType().cast<ShapedType>().getRank();
  int64_t axisIndex = axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(input).
  if (axisIndex < -inputRank || axisIndex >= inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));

  return success();
}

ONNXOpShapeHelper *ONNXLogSoftmaxOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXLogSoftmaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// LpNormalization
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXLpNormalizationOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXLpNormalizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// MeanVarianceNormalization
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXMeanVarianceNormalizationOp::getShapeHelper(
    Operation *op, ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb,
    IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXMeanVarianceNormalizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXNegOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXNegOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Not
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXNotOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXNotOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// PRelu
//===----------------------------------------------------------------------===//

LogicalResult ONNXPReluOp::verify() {
  if (!hasShapeAndRank(X())) {
    return success();
  }
  if (!hasShapeAndRank(slope())) {
    return success();
  }
  ArrayRef<int64_t> xShape = X().getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> slopeShape =
      slope().getType().cast<ShapedType>().getShape();
  // PRelu supports unidirectional broadcasting, that is slope should be
  // unidirectional broadcast to input X.
  if (slopeShape.size() > xShape.size())
    return emitError("Slope tensor has a wrong shape");
  return success();
}

ONNXOpShapeHelper *ONNXPReluOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXPReluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ONNXPReluOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success();

  auto xShape = X().getType().cast<ShapedType>().getShape();
  auto slopeShape = slope().getType().cast<ShapedType>().getShape();

  // To do unidirectional broadcasting, we first apply bidirectional
  // broadcasting. Then, fine-tune by getting constant dimensions from X.
  SmallVector<int64_t, 4> shape;
  // Bidirectional broadcasting rules.
  getBroadcastedShape(xShape, slopeShape, shape);
  // Fine-tune.
  for (unsigned int i = 0; i < shape.size(); ++i)
    if (!ShapedType::isDynamic(xShape[i]))
      shape[i] = xShape[i];

  getResult().setType(RankedTensorType::get(
      shape, X().getType().cast<ShapedType>().getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ReciprocalOp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXReciprocalOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXReciprocalOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Relu
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXReluOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXReluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Round
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXRoundOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXRoundOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Scaler
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXScalerOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXScalerOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(X()))
    return success();

  ONNXUnaryOpShapeHelper shapeHelper(getOperation(), {});
  RankedTensorType xType = X().getType().dyn_cast<RankedTensorType>();
  return shapeHelper.computeShapeAndUpdateType(
      FloatType::getF32(getContext()), xType.getEncoding());
}

//===----------------------------------------------------------------------===//
// Selu
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXSeluOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXSeluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Shrink
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXShrinkOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXShrinkOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Sigmoid
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXSigmoidOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXSigmoidOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SignOp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXSignOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXSignOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Sin
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXSinOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXSinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Sinh
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXSinhOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXSinhOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXSoftmaxOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXSoftmaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SoftmaxV11Op
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXSoftmaxV11Op::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXSoftmaxV11Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SoftplusOp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXSoftplusOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXSoftplusOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SoftsignOp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXSoftsignOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXSoftsignOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXSqrtOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXSqrtOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Tan
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXTanOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXTanOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Tanh
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXTanhOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXTanhOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ThresholdedRelu
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXThresholdedReluOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXThresholdedReluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// TriluOp
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXTriluOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXTriluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}
