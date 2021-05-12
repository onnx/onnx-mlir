/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

typedef SmallVector<IndexExpr, 4> DimsExpr;

/// When defining support for a new op, add one such stuct which must
/// minimally compute the outputDims present in the parent class. Computation
/// should be performed using a `Compute` function. Return success on successful
/// computation of all the IndexExpr. During shape inference, object is built
/// using a null-ptr rewriter; during lowering, the rewriter is nonnull and will
/// be used to generate code.
template <class OP>
struct ONNXOpShapeHelper {
  // Constructor for shape inference.
  ONNXOpShapeHelper(OP *newOp);
  // Constructor when code can be generated.
  ONNXOpShapeHelper(OP *newOp, ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  // Define in every children. Use op to get attributes, and operandAdaptor
  // to get the input/output parameters.
  LogicalResult Compute(ONNXSliceOpAdaptor operandAdaptor) {
    llvm_unreachable("implement in child structs");
  }

  // Return output dims for the N-th output.
  DimsExpr &dimsForOutput(int n = 0) { return outputsDims[n]; }

  // Set the number of outputs.
  void setNumberOfOutputs(int n) { outputsDims.resize(n); }

  // Data that must be present for every ShapeHelper operation. Op and scope
  // are initialized in the constructor, and outputsDims is computed by the
  // child's struct `Compute` function.
  OP *op;
  IndexExprScope scope;

protected:
  // Function to get a dense value from an attribute.
  ArrayValueIndexCapture::GetDenseVal fGetDenseVal;
  // Function to load a value from an array.
  ArrayValueIndexCapture::LoadVal fLoadVal;

private:
  SmallVector<DimsExpr, 1> outputsDims;
};

/// Compute a broadcasted shape from the shapes of given operands. Operands must
/// be ranked in advance.
struct ONNXOpBroadcastedShapeHelper {
  ONNXOpBroadcastedShapeHelper(ConversionPatternRewriter *rewriter,
      Location loc, bool uniBroadcasting = false);

  // Compute a vector of IndexExprs to represent the output shape. Results are
  // stored in 'outputDims'.
  // Used in shape inference and memory allocation for the output.
  // Parameters:
  //   - operands: a list of input tensors.
  LogicalResult Compute(ArrayRef<Value> operands);

  // Compute access indices to load/store value from/to a given 'operand'.
  // Used in a loop to access the operand.
  // Parameters:
  //   - operand: operand to access
  //   - operandIndex: index of the operand in 'inputsDims'
  //   - loopAccessExprs: IndexExprs for the loop's IVs
  //   - operandAccessExprs: access indices to access the operand.
  LogicalResult GetAccessExprs(Value operand, unsigned operandIndex,
      const SmallVectorImpl<IndexExpr> &outputAccessExprs,
      SmallVectorImpl<IndexExpr> &operandAccessExprs);

  IndexExprScope scope;
  // A vector of input shapes where dimensions are padded with 1 if necessary,
  // so that all inputs have the same rank.
  SmallVector<DimsExpr, 4> inputsDims;
  // A vector of IndexExprs representing the output shape.
  DimsExpr outputDims;
  int64_t outputRank = -1;

private:
  // If unidirectional broadcasting, the other operands are always
  // unidirectional broadcastable to the first operand.
  bool isUniBroadcasting;
};

// Shape for ArgMax
struct ONNXArgMaxOpShapeHelper : public ONNXOpShapeHelper<ONNXArgMaxOp> {
  ONNXArgMaxOpShapeHelper(ONNXArgMaxOp *newOp);
  ONNXArgMaxOpShapeHelper(ONNXArgMaxOp *newOp,
      ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult Compute(ONNXArgMaxOpAdaptor operandAdaptor);
};

// Shape for concat
struct ONNXConcatOpShapeHelper : public ONNXOpShapeHelper<ONNXConcatOp> {
  ONNXConcatOpShapeHelper(ONNXConcatOp *newOp);
  ONNXConcatOpShapeHelper(ONNXConcatOp *newOp,
      ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult Compute(ONNXConcatOpAdaptor operandAdaptor);
};

// Shape for SliceOp.
struct ONNXSliceOpShapeHelper : public ONNXOpShapeHelper<ONNXSliceOp> {
  ONNXSliceOpShapeHelper(ONNXSliceOp *newOp);
  ONNXSliceOpShapeHelper(ONNXSliceOp *newOp,
      ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult Compute(ONNXSliceOpAdaptor operandAdaptor);

  // Additional data for SliceOp.
  SmallVector<IndexExpr, 4> starts;
  SmallVector<IndexExpr, 4> ends;
  SmallVector<IndexExpr, 4> steps;
};

// Shape for Tile.
struct ONNXTileOpShapeHelper : public ONNXOpShapeHelper<ONNXTileOp> {
  ONNXTileOpShapeHelper(ONNXTileOp *newOp);
  ONNXTileOpShapeHelper(ONNXTileOp *newOp, ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult Compute(ONNXTileOpAdaptor operandAdaptor);
};

// Shape for GemmOp. Rank of C is known, and its rank can be 0, 1, or 2. Each
// of the dimensions of C can have 1 (broadcast) or many (same size as position
// requires).
struct ONNXGemmOpShapeHelper : public ONNXOpShapeHelper<ONNXGemmOp> {
  ONNXGemmOpShapeHelper(ONNXGemmOp *newOp);
  ONNXGemmOpShapeHelper(ONNXGemmOp *newOp, ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult Compute(ONNXGemmOpAdaptor operandAdaptor);

  // Additional data for GemmOp: output = a * b.
  SmallVector<IndexExpr, 4> aDims; // Dim of A, after applying transpose.
  SmallVector<IndexExpr, 4> bDims; // Dim of B, after applying transpose.
  SmallVector<IndexExpr, 4> cDims; // Dim of C, padding "1" when broadcast.
  bool hasBias;                    // Whether there is a bias (aka C exists).
  int cRank; // Dim of the original C (not padding dims by 1).
};

// Shape for MatMulOp.
struct ONNXMatMulOpShapeHelper : public ONNXOpShapeHelper<ONNXMatMulOp> {
  ONNXMatMulOpShapeHelper(ONNXMatMulOp *newOp);
  ONNXMatMulOpShapeHelper(ONNXMatMulOp *newOp,
      ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult Compute(ONNXMatMulOpAdaptor operandAdaptor);

  // Additional data for MatMulOp: output = a & b.
  SmallVector<IndexExpr, 4> aDims; // Dim of A, after applying padding.
  SmallVector<IndexExpr, 4> bDims; // Dim of B, after applying padding.
  llvm::BitVector aPadDims;        // When true, that dim was padded.
  llvm::BitVector bPadDims;        // When true, that dim was padded.
};

// Shape for Gather.
struct ONNXGatherOpShapeHelper : public ONNXOpShapeHelper<ONNXGatherOp> {
  ONNXGatherOpShapeHelper(ONNXGatherOp *newOp);
  ONNXGatherOpShapeHelper(ONNXGatherOp *newOp,
      ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult Compute(ONNXGatherOpAdaptor operandAdaptor);

  SmallVector<IndexExpr, 4> dataDims;    // Dim of data.
  SmallVector<IndexExpr, 4> indicesDims; // Dim of indices.
  bool positiveConstantIndices; // True when all indices are positive consants.
};

// Shape for SplitOp.
struct ONNXSplitOpShapeHelper : public ONNXOpShapeHelper<ONNXSplitOp> {
  ONNXSplitOpShapeHelper(ONNXSplitOp *newOp);
  ONNXSplitOpShapeHelper(ONNXSplitOp *newOp,
      ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult Compute(ONNXSplitOpAdaptor operandAdaptor);
};

// Shape for TransposeOp.
struct ONNXTransposeOpShapeHelper : public ONNXOpShapeHelper<ONNXTransposeOp> {
  ONNXTransposeOpShapeHelper(ONNXTransposeOp *newOp);
  ONNXTransposeOpShapeHelper(ONNXTransposeOp *newOp,
      ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult Compute(ONNXTransposeOpAdaptor operandAdaptor);
};

// Shape for LRN.
struct ONNXLRNOpShapeHelper : public ONNXOpShapeHelper<ONNXLRNOp> {
  ONNXLRNOpShapeHelper(ONNXLRNOp *newOp);
  ONNXLRNOpShapeHelper(ONNXLRNOp *newOp, ConversionPatternRewriter &rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult Compute(ONNXLRNOpAdaptor operandAdaptor);
};
