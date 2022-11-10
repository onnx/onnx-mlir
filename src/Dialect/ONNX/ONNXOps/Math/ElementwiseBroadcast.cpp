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
template <class OP, class ADAPTOR>
static LogicalResult verifyShapeForBroadcastingOps(
    OP &op, Type elementType = nullptr) {
  ADAPTOR operandAdaptor(op);
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
template <class OP, class ADAPTOR>
static LogicalResult inferShapeForBroadcastingOps(
    OP &op, Type elementType = nullptr) {
  ADAPTOR operandAdaptor(op);
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
  return verifyShapeForBroadcastingOps<ONNXAddOp, ONNXAddOpAdaptor>(*this);
}

LogicalResult ONNXAddOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXAddOp, ONNXAddOpAdaptor>(*this);
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXAndOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXAndOp, ONNXAndOpAdaptor>(*this);
}

LogicalResult ONNXAndOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXAndOp, ONNXAndOpAdaptor>(*this);
}

//===----------------------------------------------------------------------===//
// BitShiftOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXBitShiftOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXBitShiftOp, ONNXBitShiftOpAdaptor>(
      *this);
}

LogicalResult ONNXBitShiftOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXBitShiftOp, ONNXBitShiftOpAdaptor>(
      *this);
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXDivOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXDivOp, ONNXDivOpAdaptor>(*this);
}

LogicalResult ONNXDivOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXDivOp, ONNXDivOpAdaptor>(*this);
}

//===----------------------------------------------------------------------===//
// EqualOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXEqualOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXEqualOp, ONNXEqualOpAdaptor>(*this);
}

LogicalResult ONNXEqualOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Builder b(getContext());
  return inferShapeForBroadcastingOps<ONNXEqualOp, ONNXEqualOpAdaptor>(
      *this, b.getI1Type());
}

//===----------------------------------------------------------------------===//
// GreaterOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXGreaterOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXGreaterOp, ONNXGreaterOpAdaptor>(
      *this);
}

LogicalResult ONNXGreaterOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Builder b(getContext());
  return inferShapeForBroadcastingOps<ONNXGreaterOp, ONNXGreaterOpAdaptor>(
      *this, b.getI1Type());
}

//===----------------------------------------------------------------------===//
// GreaterOrEqualOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXGreaterOrEqualOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXGreaterOrEqualOp,
      ONNXGreaterOrEqualOpAdaptor>(*this);
}

LogicalResult ONNXGreaterOrEqualOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Builder b(getContext());
  return inferShapeForBroadcastingOps<ONNXGreaterOrEqualOp,
      ONNXGreaterOrEqualOpAdaptor>(*this, b.getI1Type());
}

//===----------------------------------------------------------------------===//
// Less
//===----------------------------------------------------------------------===//

LogicalResult ONNXLessOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXLessOp, ONNXLessOpAdaptor>(*this);
}

LogicalResult ONNXLessOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Builder b(getContext());
  return inferShapeForBroadcastingOps<ONNXLessOp, ONNXLessOpAdaptor>(
      *this, b.getI1Type());
}

//===----------------------------------------------------------------------===//
// LessOrEqualOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXLessOrEqualOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXLessOrEqualOp,
      ONNXLessOrEqualOpAdaptor>(*this);
}

LogicalResult ONNXLessOrEqualOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Builder b(getContext());
  return inferShapeForBroadcastingOps<ONNXLessOrEqualOp,
      ONNXLessOrEqualOpAdaptor>(*this, b.getI1Type());
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMaxOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXMaxOp, ONNXMaxOpAdaptor>(*this);
}

LogicalResult ONNXMaxOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMaxOp, ONNXMaxOpAdaptor>(*this);
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMeanOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXMeanOp, ONNXMeanOpAdaptor>(*this);
}

LogicalResult ONNXMeanOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMeanOp, ONNXMeanOpAdaptor>(*this);
}

//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMinOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXMinOp, ONNXMinOpAdaptor>(*this);
}

LogicalResult ONNXMinOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMinOp, ONNXMinOpAdaptor>(*this);
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
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXModOp, ONNXModOpAdaptor>(*this);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMulOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXMulOp, ONNXMulOpAdaptor>(*this);
}

LogicalResult ONNXMulOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXMulOp, ONNXMulOpAdaptor>(*this);
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXOrOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXOrOp, ONNXOrOpAdaptor>(*this);
}

LogicalResult ONNXOrOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXOrOp, ONNXOrOpAdaptor>(*this);
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
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXPowOp, ONNXPowOpAdaptor>(*this);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSubOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXSubOp, ONNXSubOpAdaptor>(*this);
}

LogicalResult ONNXSubOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXSubOp, ONNXSubOpAdaptor>(*this);
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSumOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXSumOp, ONNXSumOpAdaptor>(*this);
}

LogicalResult ONNXSumOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXSumOp, ONNXSumOpAdaptor>(*this);
}

//===----------------------------------------------------------------------===//
// WhereOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXWhereOp::verify() {
  Type resultElementType = X().getType().cast<ShapedType>().getElementType();
  return verifyShapeForBroadcastingOps<ONNXWhereOp, ONNXWhereOpAdaptor>(
      *this, resultElementType);
}

LogicalResult ONNXWhereOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Type resultElementType = X().getType().cast<ShapedType>().getElementType();
  return inferShapeForBroadcastingOps<ONNXWhereOp, ONNXWhereOpAdaptor>(
      *this, resultElementType);
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXXorOp::verify() {
  return verifyShapeForBroadcastingOps<ONNXXorOp, ONNXXorOpAdaptor>(*this);
}

LogicalResult ONNXXorOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXXorOp, ONNXXorOpAdaptor>(*this);
}
