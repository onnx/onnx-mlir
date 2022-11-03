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

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

/// Handle shape inference for unary element-wise operators.
LogicalResult inferShapeForUnaryElementwiseOps(Operation *op) {
  Value input = op->getOperand(0);
  Value output = op->getResult(0);

  if (!hasShapeAndRank(input))
    return success();

  ONNXGenericOpUnaryElementwiseShapeHelper shapeHelper(op);
  if (failed(shapeHelper.computeShape(input)))
    return op->emitError("Failed to scan parameters successfully");
  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(), outputDims);

  // Inferred shape is getting from the input's shape.
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  updateType(
      output, outputDims, inputType.getElementType(), inputType.getEncoding());
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXAbsOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Acos
//===----------------------------------------------------------------------===//

LogicalResult ONNXAcosOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Acosh
//===----------------------------------------------------------------------===//

LogicalResult ONNXAcoshOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Asin
//===----------------------------------------------------------------------===//

LogicalResult ONNXAsinOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Asinh
//===----------------------------------------------------------------------===//

LogicalResult ONNXAsinhOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Atan
//===----------------------------------------------------------------------===//

LogicalResult ONNXAtanOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Atanh
//===----------------------------------------------------------------------===//

LogicalResult ONNXAtanhOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Cast
//===----------------------------------------------------------------------===//

LogicalResult ONNXCastOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
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

  mlir::Type targetType =
      (*this)->getAttr("to").cast<::mlir::TypeAttr>().getValue();
  OpBuilder builder(getContext());
  getResult().setType(getOutputType(targetType));
  return success();
}

//===----------------------------------------------------------------------===//
// CastLike
//===----------------------------------------------------------------------===//

LogicalResult ONNXCastLikeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
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

LogicalResult ONNXCeilOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Celu
//===----------------------------------------------------------------------===//

LogicalResult ONNXCeluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Cos
//===----------------------------------------------------------------------===//

LogicalResult ONNXCosOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Cosh
//===----------------------------------------------------------------------===//

LogicalResult ONNXCoshOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// CumSum
//===----------------------------------------------------------------------===//

LogicalResult ONNXCumSumOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Elu
//===----------------------------------------------------------------------===//

LogicalResult ONNXEluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ErfOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXErfOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Exp
//===----------------------------------------------------------------------===//

LogicalResult ONNXExpOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Floor
//===----------------------------------------------------------------------===//

LogicalResult ONNXFloorOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// HardSigmoid
//===----------------------------------------------------------------------===//

LogicalResult ONNXHardSigmoidOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// HardmaxOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXHardmaxOp::verify() {
  ONNXHardmaxOpAdaptor operandAdaptor(*this);
  Value input = operandAdaptor.input();
  if (!hasShapeAndRank(input))
    return success(); // Won't be able to do any checking at this stage.

  // axis attribute must be in the range [-r,r-1], where r = rank(input).
  int64_t axisValue = axis();
  int64_t inputRank = input.getType().cast<ShapedType>().getRank();
  if (axisValue < -inputRank || axisValue >= inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisValue,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));

  return success();
}

LogicalResult ONNXHardmaxOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasShapeAndRank(input()))
    return success();

  auto inputType = input().getType().cast<ShapedType>();
  int64_t inputRank = inputType.getRank();
  int64_t axisValue = axis();

  // axis attribute must be in the range [-r,r], where r = rank(input).
  if (axisValue < -inputRank || axisValue > inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisValue,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));

  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// HardSwishOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXHardSwishOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// InstanceNormalization
//===----------------------------------------------------------------------===//

LogicalResult ONNXInstanceNormalizationOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// LeakyRelu
//===----------------------------------------------------------------------===//

LogicalResult ONNXLeakyReluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Log
//===----------------------------------------------------------------------===//

LogicalResult ONNXLogOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
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

LogicalResult ONNXLogSoftmaxOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// LpNormalization
//===----------------------------------------------------------------------===//

LogicalResult ONNXLpNormalizationOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// MeanVarianceNormalization
//===----------------------------------------------------------------------===//

LogicalResult ONNXMeanVarianceNormalizationOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXNegOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Not
//===----------------------------------------------------------------------===//

LogicalResult ONNXNotOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
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

LogicalResult ONNXPReluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
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
    if (xShape[i] != -1)
      shape[i] = xShape[i];

  getResult().setType(RankedTensorType::get(
      shape, X().getType().cast<ShapedType>().getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ReciprocalOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXReciprocalOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Relu
//===----------------------------------------------------------------------===//

LogicalResult ONNXReluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Round
//===----------------------------------------------------------------------===//

LogicalResult ONNXRoundOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Scaler
//===----------------------------------------------------------------------===//

LogicalResult ONNXScalerOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto inputType = X().getType().dyn_cast<RankedTensorType>();

  if (!inputType)
    return success();

  updateType(
      getResult(), inputType.getShape(), FloatType::getF32(getContext()));
  return success();
}

//===----------------------------------------------------------------------===//
// Scatter
//===----------------------------------------------------------------------===//

LogicalResult ONNXScatterOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ScatterElements
//===----------------------------------------------------------------------===//

LogicalResult ONNXScatterElementsOp::verify() {
  ONNXScatterElementsOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  // Get operands and attributes.
  Value data = operandAdaptor.data();
  Value indices = operandAdaptor.indices();
  Value updates = operandAdaptor.updates();
  auto dataType = data.getType().cast<ShapedType>();
  auto indicesType = indices.getType().cast<ShapedType>();
  auto updatesType = updates.getType().cast<ShapedType>();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  int64_t updatesRank = updatesType.getRank();
  int64_t axis = this->axis();

  // All inputs must have the same rank, and the rank must be strictly greater
  // than zero.
  if (dataRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), data, dataRank, "> 0");
  if (indicesRank != dataRank)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), indices, indicesRank, std::to_string(dataRank));
  if (updatesRank != dataRank)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), updates, updatesRank, std::to_string(dataRank));

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  if (axis < -dataRank || axis >= dataRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axis,
        onnx_mlir::Diagnostic::Range<int64_t>(-dataRank, dataRank - 1));

  if (axis < 0)
    axis += dataRank;

  // All index values in 'indices' are expected to be within bounds [-s, s-1]
  // along axis of size s.
  ArrayRef<int64_t> dataShape = dataType.getShape();
  const int64_t dataDimAtAxis = dataShape[axis];
  if (dataDimAtAxis >= 0) {
    if (DenseElementsAttr valueAttribute =
            getDenseElementAttributeFromONNXValue(indices)) {
      for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
        int64_t index = value.getInt();
        if (index >= -dataDimAtAxis && index < dataDimAtAxis)
          continue;

        return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
            *this->getOperation(), "indices", index,
            onnx_mlir::Diagnostic::Range<int64_t>(
                -dataDimAtAxis, dataDimAtAxis - 1));
      }
    }
  }

  return success();
}

LogicalResult ONNXScatterElementsOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ScatterND
//===----------------------------------------------------------------------===//

LogicalResult ONNXScatterNDOp::verify() {
  ONNXScatterNDOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  // Get operands and attributes.
  Value data = operandAdaptor.data();
  Value indices = operandAdaptor.indices();
  Value updates = operandAdaptor.updates();
  auto dataType = data.getType().cast<ShapedType>();
  auto indicesType = indices.getType().cast<ShapedType>();
  auto updatesType = updates.getType().cast<ShapedType>();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  int64_t updatesRank = updatesType.getRank();

  // 'data' and 'indices' must have rank strictly greater than zero.
  if (dataRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), data, dataRank, "> 0");
  if (indicesRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), indices, indicesRank, "> 0");

  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  ArrayRef<int64_t> updatesShape = updatesType.getShape();
  int64_t indicesLastDim = indicesShape[indicesRank - 1];

  // The rank of 'updates' must be equal to:
  //    rank(data) + rank(indices) - indices.shape[-1] - 1.
  if (indicesLastDim > 0) {
    int64_t expectedUpdatesRank = dataRank + indicesRank - indicesLastDim - 1;
    if (updatesRank != expectedUpdatesRank)
      return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
          *this->getOperation(), updates, updatesRank,
          std::to_string(expectedUpdatesRank));

    // The last dimension of the 'indices' shape can be at most equal to the
    // rank of 'data'.
    if (indicesLastDim > dataRank)
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), indices, indicesRank - 1, indicesLastDim,
          "<= " + std::to_string(dataRank));
  }

  // The constraints check following this point requires the input tensors shape
  // dimensions to be known, if they aren't delay the checks.
  if (llvm::any_of(indicesShape, [](int64_t idx) { return (idx < 0); }))
    return success();
  if (llvm::any_of(updatesShape, [](int64_t idx) { return (idx < 0); }))
    return success();

  // Let q = rank(indices). The first (q-1) dimensions of the 'updates' shape
  // must match the first (q-1) dimensions of the 'indices' shape.
  for (int64_t i = 0; i < indicesRank - 1; ++i) {
    assert(i < updatesRank && "i is out of bounds");
    if (updatesShape[i] != indicesShape[i])
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), updates, i, updatesShape[i],
          std::to_string(indicesShape[i]));
  }

  if (llvm::any_of(dataShape, [](int64_t idx) { return (idx < 0); }))
    return success();

  // Let k = indices.shape[-1], r = rank(data), q = rank(indices). Check that
  // updates.shape[q:] matches data.shape[k:r-1].
  for (int64_t i = indicesLastDim, j = indicesRank - 1; i < dataRank;
       ++i, ++j) {
    assert(j < updatesRank && "j is out of bounds");
    if (updatesShape[j] != dataShape[i])
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), updates, j, updatesShape[j],
          std::to_string(dataShape[i]));
  }

  return success();
}

LogicalResult ONNXScatterNDOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Selu
//===----------------------------------------------------------------------===//

LogicalResult ONNXSeluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Shrink
//===----------------------------------------------------------------------===//

LogicalResult ONNXShrinkOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Sigmoid
//===----------------------------------------------------------------------===//

LogicalResult ONNXSigmoidOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SignOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSignOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Sin
//===----------------------------------------------------------------------===//

LogicalResult ONNXSinOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Sinh
//===----------------------------------------------------------------------===//

LogicalResult ONNXSinhOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSoftmaxOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}
//===----------------------------------------------------------------------===//
// SoftplusOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSoftplusOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SoftsignOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSoftsignOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSqrtOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Tan
//===----------------------------------------------------------------------===//

LogicalResult ONNXTanOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Tanh
//===----------------------------------------------------------------------===//

LogicalResult ONNXTanhOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ThresholdedRelu
//===----------------------------------------------------------------------===//

LogicalResult ONNXThresholdedReluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// TriluOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXTriluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryElementwiseOps(this->getOperation());
}
