/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ElementwiseBroadcast.cpp - ONNX Operations --------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Elementwise operation with
// broadcast.
//
// Please add operations in alphabetical order.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace {

static LogicalResult verifyShapeForBroadcastingOps(
    Operation *op, Type elementType = nullptr) {
  if (!hasShapeAndRank(op))
    return success();

  auto resultTy = op->getOperand(0).getType().template cast<ShapedType>();
  for (unsigned i = 1; i < op->getNumOperands(); ++i) {
    auto nextTy = op->getOperand(i).getType().template cast<ShapedType>();
    resultTy = getBroadcastedType(resultTy, nextTy, elementType);
    if (resultTy == nullptr)
      op->emitError("Broadcast op with incompatible dimensions");
  }
  return success();
}

// Handle shape inference for numpy style broadcasting operators.
template <class OP_TYPE>
static LogicalResult inferShapeForBroadcastingOps(
    OP_TYPE &op, Type elementType = nullptr) {
  typename OP_TYPE::Adaptor operandAdaptor(op);
  if (!hasShapeAndRank(op.getOperation()))
    return success();

  if (!elementType)
    elementType =
        op.getOperand(0).getType().template cast<ShapedType>().getElementType();
  ONNXBroadcastOpShapeHelper shapeHelper(op.getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

} // namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXAddOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXAddOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXAddOp>(*this);
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXAndOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXAndOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXAndOp>(*this);
}

//===----------------------------------------------------------------------===//
// BitwiseAndOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXBitwiseAndOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXBitwiseAndOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXBitwiseAndOp>(*this);
}

//===----------------------------------------------------------------------===//
// BitwiseAndOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXBitwiseOrOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXBitwiseOrOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXBitwiseOrOp>(*this);
}

//===----------------------------------------------------------------------===//
// BitwiseAndOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXBitwiseXorOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXBitwiseXorOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXBitwiseXorOp>(*this);
}

//===----------------------------------------------------------------------===//
// BitShiftOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXBitShiftOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXBitShiftOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXBitShiftOp>(*this);
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXDivOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXDivOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXDivOp>(*this);
}

//===----------------------------------------------------------------------===//
// EqualOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXEqualOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXEqualOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Builder b(getContext());
  return inferShapeForBroadcastingOps<ONNXEqualOp>(*this, b.getI1Type());
}

//===----------------------------------------------------------------------===//
// GreaterOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXGreaterOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXGreaterOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Builder b(getContext());
  return inferShapeForBroadcastingOps<ONNXGreaterOp>(*this, b.getI1Type());
}

//===----------------------------------------------------------------------===//
// GreaterOrEqualOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXGreaterOrEqualOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXGreaterOrEqualOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Builder b(getContext());
  return inferShapeForBroadcastingOps<ONNXGreaterOrEqualOp>(
      *this, b.getI1Type());
}

//===----------------------------------------------------------------------===//
// Less
//===----------------------------------------------------------------------===//

LogicalResult ONNXLessOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXLessOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Builder b(getContext());
  return inferShapeForBroadcastingOps<ONNXLessOp>(*this, b.getI1Type());
}

//===----------------------------------------------------------------------===//
// LessOrEqualOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXLessOrEqualOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXLessOrEqualOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Builder b(getContext());
  return inferShapeForBroadcastingOps<ONNXLessOrEqualOp>(*this, b.getI1Type());
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMaxOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXMaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMaxOp>(*this);
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMeanOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXMeanOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMeanOp>(*this);
}

//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMinOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXMinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMinOp>(*this);
}

//===----------------------------------------------------------------------===//
// ModOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXModOp::verify() {
  Type elementType;
  if (getA().getType().isa<ShapedType>())
    elementType = getA().getType().cast<ShapedType>().getElementType();
  else
    return emitOpError("Input type must be TensorType or MemRefType");

  // Verify that when the input type is floating point, then `fmod` attribute
  // must be set to 1.
  if (elementType.isa<FloatType>() && (getFmod() != 1))
    return emitOpError("fmod must be 1 when the input type is floating point");

  return success();
}

LogicalResult ONNXModOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXModOp>(*this);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMulOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXMulOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMulOp>(*this);
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXOrOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXOrOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXOrOp>(*this);
}

//===----------------------------------------------------------------------===//
// PowOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXPowOp::verify() {
  ShapedType lhsTy = getX().getType().cast<ShapedType>();
  ShapedType rhsTy = getY().getType().cast<ShapedType>();
  Type rhsETy = rhsTy.getElementType();
  Type lhsETy = lhsTy.getElementType();
  if (rhsETy != lhsETy)
    return emitOpError("Pow with different input type not implemented yet");
  if (lhsETy.isa<IntegerType>() || lhsETy.isa<IntegerType>())
    return emitOpError("Integer power not implemented yet");
  return success();
}

LogicalResult ONNXPowOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXPowOp>(*this);
}

//===----------------------------------------------------------------------===//
// PRelu
//===----------------------------------------------------------------------===//

LogicalResult ONNXPReluOp::verify() {
  if (!hasShapeAndRank(getX()))
    return success();
  if (!hasShapeAndRank(getSlope()))
    return success();

  ArrayRef<int64_t> xShape = getX().getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> slopeShape =
      getSlope().getType().cast<ShapedType>().getShape();
  // PRelu supports unidirectional broadcasting, that is slope should be
  // unidirectional broadcast to input X.
  if (slopeShape.size() > xShape.size())
    return emitError("Slope tensor has a wrong shape");
  return success();
}

LogicalResult ONNXPReluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getOperation()))
    return success();

  Type elementType = getX().getType().cast<ShapedType>().getElementType();
  ONNXPReluOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSubOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXSubOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXSubOp>(*this);
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSumOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXSumOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXSumOp>(*this);
}

//===----------------------------------------------------------------------===//
// WhereOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXWhereOp::verify() {
  Type resultElementType = getX().getType().cast<ShapedType>().getElementType();
  return verifyShapeForBroadcastingOps(getOperation(), resultElementType);
}

LogicalResult ONNXWhereOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type resultElementType = getX().getType().cast<ShapedType>().getElementType();
  return inferShapeForBroadcastingOps<ONNXWhereOp>(*this, resultElementType);
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXXorOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXXorOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXXorOp>(*this);
}
