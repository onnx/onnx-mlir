/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Elementwise.cpp - ZHigh Operations ----------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// AddOp
LogicalResult ZHighAddOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// SubOp

LogicalResult ZHighSubOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// MulOp

LogicalResult ZHighMulOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// DivOp

LogicalResult ZHighDivOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// MinOp

LogicalResult ZHighMinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// MaxOp

LogicalResult ZHighMaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// LogOp

LogicalResult ZHighLogOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// ExpOp

LogicalResult ZHighExpOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// InvSqrtOp

LogicalResult ZHighInvSqrtOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// LeakyReluOp

LogicalResult ZHighLeakyReluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// ReluOp

LogicalResult ZHighReluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// GeluOp

LogicalResult ZHighGeluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// TanhOp

LogicalResult ZHighTanhOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// SigmoiOp

LogicalResult ZHighSigmoidOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

//===----------------------------------------------------------------------===//
// SqrtOp

LogicalResult ZHighSqrtOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation(),
      getZTensorEncoding(this->getOperation()->getOperand(0).getType()));
}

} // namespace zhigh
} // namespace onnx_mlir
