//===----------------ONNXShapeHelper.hpp - help for shapes----------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/BitVector.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

/// When defining support for a new op, add one such stuct which mÍÎust
/// minimally compute the outputDims present in the parent class. Computation
/// should be performed using a `Compute` function. Return success on successful
/// computation of all the IndexExpr. During shape inference, object is built
/// using a null-ptr rewriter; during lowering, the rewriter is nonnull and will
/// be used to generate code.
template <class OP>
struct ONNXOpShapeHelper {
  ONNXOpShapeHelper(OP *newOp, ConversionPatternRewriter *rewriter);

  // Define in every children. Use op to get attributes, and operandAdaptor
  // to get the input/output parameters.
  LogicalResult Compute(ONNXSliceOpAdaptor operandAdaptor) {
    llvm_unreachable("implement in child structs");
  }

  // Data that must be present for every ShapeHelper operation. Op and context
  // are initialized in the constructor, and outputDims is computed by the
  // child's struct `Compute` function.
  OP *op;
  IndexExprContext context;
  SmallVector<IndexExpr, 4> outputDims;
};

// Shape for SliceOp.
struct ONNXSliceOpShapeHelper : public ONNXOpShapeHelper<ONNXSliceOp> {
  ONNXSliceOpShapeHelper(
      ONNXSliceOp *newOp, ConversionPatternRewriter *rewriter);

  LogicalResult Compute(ONNXSliceOpAdaptor operandAdaptor);

  // Additional data for SliceOp.
  SmallVector<IndexExpr, 4> starts;
  SmallVector<IndexExpr, 4> ends;
  SmallVector<IndexExpr, 4> steps;
};

// Shape for GemmOp.
struct ONNXGemmOpShapeHelper : public ONNXOpShapeHelper<ONNXGemmOp> {
  ONNXGemmOpShapeHelper(ONNXGemmOp *newOp, ConversionPatternRewriter *rewriter);

  LogicalResult Compute(ONNXGemmOpAdaptor operandAdaptor);

  // Additional data for GemmOp: output = a * b.
  SmallVector<IndexExpr, 4> aDims; // Dim of A, after applying transpose.
  SmallVector<IndexExpr, 4> bDims; // Dim of B, after applying transpose.
  SmallVector<IndexExpr, 4> cDims; // Dim of C, padding "1" when broadcast.
  bool hasBias;                    // Whether ther eis a bias (aka C exists).
  int cRank; // Dim of the original C (not padding dims by 1).
};

// Shape for MatMulOp.
struct ONNXMatMulOpShapeHelper : public ONNXOpShapeHelper<ONNXMatMulOp> {
  ONNXMatMulOpShapeHelper(ONNXMatMulOp *newOp, ConversionPatternRewriter *rewriter);

  LogicalResult Compute(ONNXMatMulOpAdaptor operandAdaptor);

  // Additional data for MatMulOp: output = a * b.
  SmallVector<IndexExpr, 4> aDims; // Dim of A, after applying padding.
  SmallVector<IndexExpr, 4> bDims; // Dim of B, after applying padding.
  llvm::BitVector aPadDims; // When true, that dim was padded.
  llvm::BitVector bPadDims; // When true, that dim was padded.
};

//===----------------------------------------------------------------------===//
// Low Level Helpers
//===----------------------------------------------------------------------===//

size_t ArrayAttrSize(ArrayAttr a);
size_t ArrayAttrSize(Optional<ArrayAttr> a);
int64_t ArrayAttrIntVal(ArrayAttr a, int i);
int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i);
// Returns the ConstantOp which defines an MLIR Value or null.
ONNXConstantOp getONNXConstantOp(Value value);

DenseElementsAttr getDenseElementAttributeFromValue(Value value);

bool getIntegerLiteralFromValue(Value value, int64_t &intLit);
