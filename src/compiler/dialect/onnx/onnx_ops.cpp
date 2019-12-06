//===- onnx_ops.cpp - MLIR ONNX Operations --------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines ONNX operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "onnx_ops.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ONNXOpsDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
ONNXOpsDialect::ONNXOpsDialect(mlir::MLIRContext* ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "src/compiler/onnx.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ONNX Operations
//===----------------------------------------------------------------------===//
// Exp
/// Infer the output shape of the ONNXExpOp. This method is required by the
/// shape inference interface.
void ONNXExpOp::inferShapes() {
  getResult()->setType(getOperand()->getType());
}

//===----------------------------------------------------------------------===//
// Tanh
/// Infer the output shape of the ONNXTanhOp. This method is required by the
/// shape inference interface.
void ONNXTanhOp::inferShapes() {
  getResult()->setType(getOperand()->getType());
}

//===----------------------------------------------------------------------===//
// Sinh
/// Infer the output shape of the ONNXSinhOp. This method is required by the
/// shape inference interface.
void ONNXSinhOp::inferShapes() {
  getResult()->setType(getOperand()->getType());
}

//===----------------------------------------------------------------------===//
// Cosh
/// Infer the output shape of the ONNXCoshOp. This method is required by the
/// shape inference interface.
void ONNXCoshOp::inferShapes() {
  getResult()->setType(getOperand()->getType());
}

//===----------------------------------------------------------------------===//
// Sigmoid
/// Infer the output shape of the ONNXSigmoidOp. This method is required by the
/// shape inference interface.
void ONNXSigmoidOp::inferShapes() {
  getResult()->setType(getOperand()->getType());
}

//===----------------------------------------------------------------------===//
// Add
/// Infer the output shape of the ONNXAddOp. This method is required by the
/// shape inference interface.
void ONNXAddOp::inferShapes() {
  getResult()->setType(getOperand(0)->getType());
}

//===----------------------------------------------------------------------===//
// Mul
/// Infer the output shape of the ONNXMulOp. This method is required by the
/// shape inference interface.
void ONNXMulOp::inferShapes() {
  getResult()->setType(getOperand(0)->getType());
}

//===----------------------------------------------------------------------===//
// Div
/// Infer the output shape of the ONNXDivOp. This method is required by the
/// shape inference interface.
void ONNXDivOp::inferShapes() {
  getResult()->setType(getOperand(0)->getType());
}

//===----------------------------------------------------------------------===//
// Sub
/// Infer the output shape of the ONNXSubOp. This method is required by the
/// shape inference interface.
void ONNXSubOp::inferShapes() {
  getResult()->setType(getOperand(0)->getType());
}

//===----------------------------------------------------------------------===//
// And
/// Infer the output shape of the ONNXAndOp. This method is required by the
/// shape inference interface.
void ONNXAndOp::inferShapes() {
  getResult()->setType(getOperand(0)->getType());
}

//===----------------------------------------------------------------------===//
// Or
/// Infer the output shape of the ONNXOrOp. This method is required by the
/// shape inference interface.
void ONNXOrOp::inferShapes() {
  getResult()->setType(getOperand(0)->getType());
}

//===----------------------------------------------------------------------===//
// Xor
/// Infer the output shape of the ONNXXorOp. This method is required by the
/// shape inference interface.
void ONNXXorOp::inferShapes() {
  getResult()->setType(getOperand(0)->getType());
}

//===----------------------------------------------------------------------===//

// MatMul

void ONNXMatMulOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand(0)->getType().isa<RankedTensorType>() ||
      !getOperand(1)->getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0)->getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1)->getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims;
  dims.emplace_back(lhsTy.getShape()[0]);
  dims.emplace_back(rhsTy.getShape()[1]);
  getResult()->setType(RankedTensorType::get(dims, lhsTy.getElementType()));
}

// TODO:
//   Verify that matrix sizes are valid.
//   Take into account the dimensionality of the matrix.

//===----------------------------------------------------------------------===//

// Gemm

void ONNXGemmOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand(0)->getType().isa<RankedTensorType>() ||
      !getOperand(1)->getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0)->getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1)->getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims;
  dims.emplace_back(lhsTy.getShape()[0]);
  dims.emplace_back(rhsTy.getShape()[1]);
  getResult()->setType(RankedTensorType::get(dims, lhsTy.getElementType()));
}

// FullGemm

void ONNXFullGemmOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand(0)->getType().isa<RankedTensorType>() ||
      !getOperand(1)->getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0)->getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1)->getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims;
  dims.emplace_back(lhsTy.getShape()[0]);
  dims.emplace_back(rhsTy.getShape()[1]);
  getResult()->setType(RankedTensorType::get(dims, lhsTy.getElementType()));
}

// TODO:
//   Verify that matrix sizes are valid for multiplication and addition.
//   Take into account the dimensionality of the matrix.

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/compiler/onnx.cpp.inc"
