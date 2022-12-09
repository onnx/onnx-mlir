/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ElementwiseBroadcast.cpp - ONNX Operations --------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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

// Verify shape for numpy style broadcasting operators.
template <class OP_TYPE>
static LogicalResult verifyShapeForBroadcastingOps(
    OP_TYPE &op, Type elementType = nullptr) {
  typename OP_TYPE::Adaptor operandAdaptor(op);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // cannot infer when the operands shape is not yet known.

  auto resultTy = op.getOperand(0).getType().template cast<ShapedType>();
  for (unsigned i = 1; i < op->getNumOperands(); ++i) {
    auto nextTy = op.getOperand(i).getType().template cast<ShapedType>();
    resultTy = getBroadcastedType(resultTy, nextTy, elementType);
    if (resultTy == nullptr)
      op.emitError("Broadcast op with incompatible dimensions");
  }
  return success();
}

// Handle shape inference for numpy style broadcasting operators.
template <class OP_TYPE>
static LogicalResult inferShapeForBroadcastingOps(
    OP_TYPE &op, Type elementType = nullptr) {
  typename OP_TYPE::Adaptor operandAdaptor(op);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // cannot infer when the operands shape is not yet known.

  auto resultTy = op.getOperand(0).getType().template cast<ShapedType>();
  for (unsigned i = 1; i < op->getNumOperands(); ++i) {
    auto nextTy = op.getOperand(i).getType().template cast<ShapedType>();
    resultTy = getBroadcastedType(resultTy, nextTy, elementType);
  }

  updateType(op.getResult(), getShape(resultTy), resultTy.getElementType());
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXAddOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXAddOp>(*this);
}

LogicalResult ONNXAddOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXAddOp>(*this);
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXAndOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXAndOp>(*this);
}

LogicalResult ONNXAndOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXAndOp>(*this);
}

//===----------------------------------------------------------------------===//
// BitShiftOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXBitShiftOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXBitShiftOp>(*this);
}

LogicalResult ONNXBitShiftOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXBitShiftOp>(*this);
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXDivOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXDivOp>(*this);
}

LogicalResult ONNXDivOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXDivOp>(*this);
}

//===----------------------------------------------------------------------===//
// EqualOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXEqualOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXEqualOp>(*this);
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
  return verifyShapeForBroadcastingOps<ONNXGreaterOp>(*this);
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
  return verifyShapeForBroadcastingOps<ONNXGreaterOrEqualOp>(*this);
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
  return verifyShapeForBroadcastingOps<ONNXLessOp>(*this);
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
  return verifyShapeForBroadcastingOps<ONNXLessOrEqualOp>(*this);
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
  return verifyShapeForBroadcastingOps<ONNXMaxOp>(*this);
}

LogicalResult ONNXMaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMaxOp>(*this);
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMeanOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXMeanOp>(*this);
}

LogicalResult ONNXMeanOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMeanOp>(*this);
}

//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMinOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXMinOp>(*this);
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
  if (A().getType().isa<ShapedType>())
    elementType = A().getType().cast<ShapedType>().getElementType();
  else
    return emitOpError("Input type must be TensorType or MemRefType");

  // Verify that when the input type is floating point, then `fmod` attribute
  // must be set to 1.
  if (elementType.isa<FloatType>() && (fmod() != 1))
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
  return verifyShapeForBroadcastingOps<ONNXMulOp>(*this);
}

LogicalResult ONNXMulOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMulOp>(*this);
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXOrOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXOrOp>(*this);
}

LogicalResult ONNXOrOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXOrOp>(*this);
}

//===----------------------------------------------------------------------===//
// PowOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXPowOp::verify() {
  ShapedType lhsTy = X().getType().cast<ShapedType>();
  ShapedType rhsTy = Y().getType().cast<ShapedType>();
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
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSubOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXSubOp>(*this);
}

LogicalResult ONNXSubOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXSubOp>(*this);
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSumOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXSumOp>(*this);
}

LogicalResult ONNXSumOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXSumOp>(*this);
}

//===----------------------------------------------------------------------===//
// WhereOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXWhereOp::verify() {
  Type resultElementType = X().getType().cast<ShapedType>().getElementType();
  return verifyShapeForBroadcastingOps<ONNXWhereOp>(*this, resultElementType);
}

LogicalResult ONNXWhereOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type resultElementType = X().getType().cast<ShapedType>().getElementType();
  return inferShapeForBroadcastingOps<ONNXWhereOp>(*this, resultElementType);
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXXorOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXXorOp>(*this);
}

LogicalResult ONNXXorOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXXorOp>(*this);
}
