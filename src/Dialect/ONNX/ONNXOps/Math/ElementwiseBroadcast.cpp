/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ElementwiseBroadcast.cpp - ONNX Operations --------===//
//
// Copyright 2019-2024 The IBM Research Authors.
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

// Returns true if op is a v1-v6 binary op with legacy axis and
// broadcast attributes set.
static bool hasBroadcastAxisAttribute(Operation *op) {
  IntegerAttr bcast = op->getAttrOfType<IntegerAttr>("broadcast");
  return bcast && bcast.getValue().getSExtValue() == 1 &&
         op->getAttrOfType<IntegerAttr>("axis");
}

static LogicalResult verifyShapeForBroadcastingOps(Operation *op) {
  if (!hasShapeAndRank(op))
    return success();

  if (hasBroadcastAxisAttribute(op)) {
    // Leave it to BinaryOpBroadcastAxisPattern to process and remove the axis
    // attribute and fix up the shapes instead of trying to verify them here.
    return success();
  }

  auto operands = op->getOperands();
  auto it = operands.begin();
  SmallVector<int64_t> resultShape(getShape((*it).getType()));
  while (++it != operands.end()) {
    ArrayRef<int64_t> nextShape = getShape((*it).getType());
    SmallVector<int64_t> bcastShape;
    if (!OpTrait::util::getBroadcastedShape(
            resultShape, nextShape, bcastShape)) {
      op->emitOpError("Broadcast op with incompatible shapes: ") << *op;
    }
    resultShape = bcastShape;
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

  if (hasBroadcastAxisAttribute(op.getOperation())) {
    // Leave it to BinaryOpBroadcastAxisPattern to process and remove the axis
    // attribute and fix up the shapes instead of trying to infer shapes here.
    return success();
  }

  if (!elementType)
    elementType =
        mlir::cast<ShapedType>(op.getOperand(0).getType()).getElementType();
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
// BitwiseOrOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXBitwiseOrOp::verify() {
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXBitwiseOrOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForBroadcastingOps<ONNXBitwiseOrOp>(*this);
}

//===----------------------------------------------------------------------===//
// BitwiseXorOp
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
  if (mlir::isa<ShapedType>(getA().getType()))
    elementType = mlir::cast<ShapedType>(getA().getType()).getElementType();
  else
    return emitOpError("Input type must be TensorType or MemRefType");

  // Verify that when the input type is floating point, then `fmod` attribute
  // must be set to 1.
  if (mlir::isa<FloatType>(elementType) && (getFmod() != 1))
    return emitOpError("fmod must be 1 when the input type is floating point");
  // Verify that when the input type is integer, then `fmod` attribute
  // must be set to 0.
  if (mlir::isa<IntegerType>(elementType) && (getFmod() != 0))
    return emitOpError("fmod must be 0 when the input type is an integer");

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
  ShapedType lhsTy = mlir::cast<ShapedType>(getX().getType());
  ShapedType rhsTy = mlir::cast<ShapedType>(getY().getType());
  Type rhsETy = rhsTy.getElementType();
  Type lhsETy = lhsTy.getElementType();
  if (rhsETy != lhsETy)
    return emitOpError("Pow with different input type not implemented yet");
  if (mlir::isa<IntegerType>(lhsETy) || mlir::isa<IntegerType>(lhsETy))
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

  ArrayRef<int64_t> xShape =
      mlir::cast<ShapedType>(getX().getType()).getShape();
  ArrayRef<int64_t> slopeShape =
      mlir::cast<ShapedType>(getSlope().getType()).getShape();
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

  Type elementType = mlir::cast<ShapedType>(getX().getType()).getElementType();
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
  return verifyShapeForBroadcastingOps(getOperation());
}

LogicalResult ONNXWhereOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type resultElementType =
      mlir::cast<ShapedType>(getX().getType()).getElementType();
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
