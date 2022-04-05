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

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

// Steps to add a new op XXX:
// 1) Create a new shape inference type inside this file, ONNXShapeHelper.hpp.
// 2) Create new shape inference file, say XXX.cpp and implement.
// 3) Add template instantiation at bottom of ONNXShapeHelper.cpp.
// 4) Add new file name XXX.cpp to ../CMakeLists.txt
// 5) Use the new object in ONNXOps.cpp and ONNXToKrnl lowering for XXX.

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

using DimsExpr = llvm::SmallVector<IndexExpr, 4>;

/// When defining support for a new op, add one such stuct which must
/// minimally compute the outputDims present in the parent class. Computation
/// should be performed using a `computeShape` function. Return success on
/// successful computation of all the IndexExpr. During shape inference, object
/// is built using a null-ptr rewriter; during lowering, the rewriter is nonnull
/// and will be used to generate code.
///
/// By adding here the ability of a ShapeHelper to be created in the
/// IndexExprScope of another ShapeHelper, this enables us to nest ShapeHelper.
/// For example, there is a case where ExpandOp needs to find out specific
/// details of an ShapeOp that provides info to the ExpandOp. We can now invoke
/// the ShapeOp shape helper in the context of the ExpandOp shape helper while
/// having all of the IndexExpr info in the same context and thus be generally
/// usable. Support is here to provide an IndexExprScope, which can be added to
/// any subclasses of ONNXOpShapeHelper when this nesting becomes useful to
/// other ops as well.

template <class OP>
struct ONNXOpShapeHelper {
  // Constructor for shape inference. Reuse scope if given, otherwise create one
  // now and free it in destructor.
  ONNXOpShapeHelper(
      OP *newOp, int numResults, IndexExprScope *inScope = nullptr);
  // Constructor when code can be generated. Reuse scope if given, otherwise
  // create one now and free it in destructor.
  ONNXOpShapeHelper(OP *newOp, int numResults, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);
  ~ONNXOpShapeHelper() {
    if (ownScope)
      delete scope;
  }

  // Every child class is expected to create a computeShape with the following
  // signature. This method is responsible to compute at a minimum the output
  // dims.
  //
  // LogicalResult computeShape(<<OpAdaptor>> operandAdaptor);
  //
  // Use the op to get attributes, and operandAdaptor to get the input/output
  // tensors.

  // Return output dims for the N-th output.
  DimsExpr &dimsForOutput(int n = 0) { return outputsDims[n]; }

  // Set the number of outputs.
  void setNumberOfOutputs(int n) { outputsDims.resize(n); }

  // Data that must be present for every ShapeHelper operation. Op and scope
  // are initialized in the constructor, and outputsDims is computed by the
  // child's struct `computeShape` function.
  OP *op;
  IndexExprScope *scope;

protected:
  // Function to get a dense value from an attribute.
  ArrayValueIndexCapture::GetDenseVal fGetDenseVal;
  // Function to load a value from an array.
  ArrayValueIndexCapture::LoadVal fLoadVal;

private:
  llvm::SmallVector<DimsExpr, 1> outputsDims;
  bool ownScope;
};

/// Compute a broadcasted shape from the shapes of given operands. Operands must
/// be ranked in advance.
template <class OP>
struct ONNXOpBroadcastedShapeHelper : public ONNXOpShapeHelper<OP> {
  ONNXOpBroadcastedShapeHelper(OP *newOp, IndexExprScope *inScope = nullptr,
      bool uniBroadcasting = false, bool noBroadcasting = false);

  ONNXOpBroadcastedShapeHelper(OP *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr, bool uniBroadcasting = false,
      bool noBroadcasting = false);

  // computeShape a vector of IndexExprs to represent the output shape. Results
  // are stored in 'outputDims'. Used in shape inference and memory allocation
  // for the output. Parameters:
  //   - operands: a list of input tensors.
  //   - additional operand: one additional input that comes from as a vector
  //     of IndexExpr (used for example for ONNXExpandOp). Ignored when empty.
  mlir::LogicalResult computeShape(
      mlir::ArrayRef<mlir::Value> operands, DimsExpr &additionalOperand);

  // Compute access indices to load/store value from/to a given 'operand'.
  // Used in a loop to access the operand.
  // Parameters:
  //   - operand: operand to access.
  //   - operandIndex: index of the operand in 'this->inputsDims'.
  //   - loopAccessExprs: IndexExprs for the loop's IVs.
  //   - operandAccessExprs: access indices to access the operand.
  //     This is the output of this function. Use it in subsequent load/stores.
  mlir::LogicalResult GetAccessExprs(mlir::Value operand, unsigned operandIndex,
      const llvm::SmallVectorImpl<IndexExpr> &outputAccessExprs,
      llvm::SmallVectorImpl<IndexExpr> &operandAccessExprs);

  // A vector of input shapes where dimensions are padded with 1 if necessary,
  // so that all inputs have the same rank.
  llvm::SmallVector<DimsExpr, 4> inputsDims;
  // A vector of IndexExprs representing the output shape.
  // in upper DimsExpr outputDims;
  int64_t outputRank;

protected:
  // If unidirectional broadcasting, the other operands are always
  // unidirectional broadcastable to the first operand.
  bool isUniBroadcasting;

  // If isNoBroadcasting is true, the shape of all input is assumed to be same
  // This flag is used to test dynamic shape
  // There is no impact on static shape
  bool isNoBroadcasting;
};

struct ONNXGenericOpBroadcastedShapeHelper
    : public ONNXOpBroadcastedShapeHelper<mlir::Operation> {
  ONNXGenericOpBroadcastedShapeHelper(mlir::Operation *newOp,
      onnx_mlir::IndexExprScope *inScope = nullptr,
      bool uniBroadcasting = false, bool noBroadcasting = false);
  ONNXGenericOpBroadcastedShapeHelper(mlir::Operation *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      onnx_mlir::IndexExprScope *inScope = nullptr,
      bool uniBroadcasting = false, bool noBroadcasting = false);
};

// Shape for ArgMax
struct ONNXArgMaxOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXArgMaxOp> {
  ONNXArgMaxOpShapeHelper(mlir::ONNXArgMaxOp *newOp);
  ONNXArgMaxOpShapeHelper(mlir::ONNXArgMaxOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXArgMaxOpAdaptor operandAdaptor);
};

// Shape for Clip.
struct ONNXClipOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXClipOp> {
  ONNXClipOpShapeHelper(mlir::ONNXClipOp *newOp);
  ONNXClipOpShapeHelper(mlir::ONNXClipOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXClipOpAdaptor operandAdaptor);
};

// Shape for concat
struct ONNXConcatOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXConcatOp> {
  ONNXConcatOpShapeHelper(mlir::ONNXConcatOp *newOp);
  ONNXConcatOpShapeHelper(mlir::ONNXConcatOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXConcatOpAdaptor operandAdaptor);
};

// Shape for DepthToSpace.
struct ONNXDepthToSpaceOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXDepthToSpaceOp> {
  ONNXDepthToSpaceOpShapeHelper(mlir::ONNXDepthToSpaceOp *newOp);
  ONNXDepthToSpaceOpShapeHelper(mlir::ONNXDepthToSpaceOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(
      mlir::ONNXDepthToSpaceOpAdaptor operandAdaptor);
};

// Shape for SliceOp.
struct ONNXSliceOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXSliceOp> {
  ONNXSliceOpShapeHelper(mlir::ONNXSliceOp *newOp);
  ONNXSliceOpShapeHelper(mlir::ONNXSliceOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXSliceOpAdaptor operandAdaptor);
  // Additional data for SliceOp.
  llvm::SmallVector<IndexExpr, 4> starts;
  llvm::SmallVector<IndexExpr, 4> ends;
  llvm::SmallVector<IndexExpr, 4> steps;
};

// Shape for Tile.
struct ONNXTileOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXTileOp> {
  ONNXTileOpShapeHelper(mlir::ONNXTileOp *newOp);
  ONNXTileOpShapeHelper(mlir::ONNXTileOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXTileOpAdaptor operandAdaptor);
};

// Shape for GemmOp. Rank of C is known, and its rank can be 0, 1, or 2. Each
// of the dimensions of C can have 1 (broadcast) or many (same size as position
// requires).
struct ONNXGemmOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXGemmOp> {
  ONNXGemmOpShapeHelper(mlir::ONNXGemmOp *newOp);
  ONNXGemmOpShapeHelper(mlir::ONNXGemmOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXGemmOpAdaptor operandAdaptor);
  // Additional data for GemmOp: output = a * b.
  llvm::SmallVector<IndexExpr, 4> aDims; // Dim of A, after applying transpose.
  llvm::SmallVector<IndexExpr, 4> bDims; // Dim of B, after applying transpose.
  llvm::SmallVector<IndexExpr, 4>
      cDims;    // Dim of C, padding "1" when broadcast.
  bool hasBias; // Whether there is a bias (aka C exists).
  int cRank;    // Dim of the original C (not padding dims by 1).
};

// Shape for MatMulOp.
struct ONNXMatMulOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXMatMulOp> {
  ONNXMatMulOpShapeHelper(mlir::ONNXMatMulOp *newOp);
  ONNXMatMulOpShapeHelper(mlir::ONNXMatMulOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXMatMulOpAdaptor operandAdaptor);
  // Additional data for MatMulOp: output = a & b.
  llvm::SmallVector<IndexExpr, 4> aDims; // Dim of A, after applying padding.
  llvm::SmallVector<IndexExpr, 4> bDims; // Dim of B, after applying padding.
  llvm::BitVector aPadDims;              // When true, that dim was padded.
  llvm::BitVector bPadDims;              // When true, that dim was padded.
};

// Shape for Gather.
struct ONNXGatherOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXGatherOp> {
  ONNXGatherOpShapeHelper(mlir::ONNXGatherOp *newOp);
  ONNXGatherOpShapeHelper(mlir::ONNXGatherOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXGatherOpAdaptor operandAdaptor);
  // Additional data for GatherOp.
  llvm::SmallVector<IndexExpr, 4> dataDims;    // Dim of data.
  llvm::SmallVector<IndexExpr, 4> indicesDims; // Dim of indices.
  bool positiveConstantIndices; // True when all indices are positive consants.
};

// Shape for SpaceToDepth.
struct ONNXSpaceToDepthOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXSpaceToDepthOp> {
  ONNXSpaceToDepthOpShapeHelper(mlir::ONNXSpaceToDepthOp *newOp);
  ONNXSpaceToDepthOpShapeHelper(mlir::ONNXSpaceToDepthOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(
      mlir::ONNXSpaceToDepthOpAdaptor operandAdaptor);
};

// Shape for SplitOp.
struct ONNXSplitOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXSplitOp> {
  ONNXSplitOpShapeHelper(mlir::ONNXSplitOp *newOp);
  ONNXSplitOpShapeHelper(mlir::ONNXSplitOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXSplitOpAdaptor operandAdaptor);
};

// Shape for SplitV11Op.
struct ONNXSplitV11OpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXSplitV11Op> {
  ONNXSplitV11OpShapeHelper(mlir::ONNXSplitV11Op *newOp);
  ONNXSplitV11OpShapeHelper(mlir::ONNXSplitV11Op *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXSplitV11OpAdaptor operandAdaptor);
};

// Shape for TransposeOp.
struct ONNXTransposeOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXTransposeOp> {
  ONNXTransposeOpShapeHelper(mlir::ONNXTransposeOp *newOp);
  ONNXTransposeOpShapeHelper(mlir::ONNXTransposeOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXTransposeOpAdaptor operandAdaptor);
};

// Shape for LRN.
struct ONNXLRNOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXLRNOp> {
  ONNXLRNOpShapeHelper(mlir::ONNXLRNOp *newOp);
  ONNXLRNOpShapeHelper(mlir::ONNXLRNOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXLRNOpAdaptor operandAdaptor);
};

// Shape for generic pooling/conv ops.
template <typename OP_TYPE, typename OP_ADAPTOR>
struct ONNXGenericPoolShapeHelper : public ONNXOpShapeHelper<OP_TYPE> {
  ONNXGenericPoolShapeHelper(OP_TYPE *newOp, bool hasFilter, bool ceilMode)
      : ONNXOpShapeHelper<OP_TYPE>(
            newOp, newOp->getOperation()->getNumResults()),
        hasFilter(hasFilter), ceilMode(ceilMode) {}

  ONNXGenericPoolShapeHelper(OP_TYPE *newOp, bool hasFilter, bool ceilMode,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal)
      : ONNXOpShapeHelper<OP_TYPE>(newOp,
            newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
            fLoadVal),
        hasFilter(hasFilter), ceilMode(ceilMode) {}

  mlir::LogicalResult computeShape(OP_ADAPTOR operandAdaptor,
      mlir::Value filterValue, mlir::Optional<mlir::ArrayAttr> kernelShapeOpt,
      mlir::Optional<mlir::ArrayAttr> padOpt,
      mlir::Optional<mlir::ArrayAttr> strideOpt,
      mlir::Optional<mlir::ArrayAttr> dilationOpt);
  // Additional data for Pool operations.
  bool hasFilter; // If has filter, it also has CO and optional kernel.
  bool ceilMode;  // Use ceil or floor for auto_pad=NOTSET policy.
  // Values set by Compute.
  llvm::SmallVector<IndexExpr, 2> kernelShape;
  llvm::SmallVector<IndexExpr, 4> pads;
  llvm::SmallVector<int64_t, 2> strides;
  llvm::SmallVector<int64_t, 2> dilations;
};

// Shape for Conv.
struct ONNXConvOpShapeHelper
    : public ONNXGenericPoolShapeHelper<mlir::ONNXConvOp,
          mlir::ONNXConvOpAdaptor> {
  ONNXConvOpShapeHelper(mlir::ONNXConvOp *newOp);
  ONNXConvOpShapeHelper(mlir::ONNXConvOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXConvOpAdaptor operandAdaptor);
};

// Shape for MaxPoolSingleOut.
struct ONNXMaxPoolSingleOutOpShapeHelper
    : public ONNXGenericPoolShapeHelper<mlir::ONNXMaxPoolSingleOutOp,
          mlir::ONNXMaxPoolSingleOutOpAdaptor> {
  ONNXMaxPoolSingleOutOpShapeHelper(mlir::ONNXMaxPoolSingleOutOp *newOp)
      : ONNXGenericPoolShapeHelper<mlir::ONNXMaxPoolSingleOutOp,
            mlir::ONNXMaxPoolSingleOutOpAdaptor>(
            newOp, false /*hasFilter*/, newOp->ceil_mode()) {}

  ONNXMaxPoolSingleOutOpShapeHelper(mlir::ONNXMaxPoolSingleOutOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal)
      : ONNXGenericPoolShapeHelper<mlir::ONNXMaxPoolSingleOutOp,
            mlir::ONNXMaxPoolSingleOutOpAdaptor>(newOp, false /*hasFilter*/,
            newOp->ceil_mode(), rewriter, fGetDenseVal, fLoadVal) {}

  mlir::LogicalResult computeShape(
      mlir::ONNXMaxPoolSingleOutOpAdaptor operandAdaptor);
};

// Shape for ONNXAveragePoolOp
struct ONNXAveragePoolOpShapeHelper
    : public ONNXGenericPoolShapeHelper<mlir::ONNXAveragePoolOp,
          mlir::ONNXAveragePoolOpAdaptor> {
  ONNXAveragePoolOpShapeHelper(mlir::ONNXAveragePoolOp *newOp)
      : ONNXGenericPoolShapeHelper<mlir::ONNXAveragePoolOp,
            mlir::ONNXAveragePoolOpAdaptor>(
            newOp, false /*hasFilter*/, newOp->ceil_mode()) {}

  ONNXAveragePoolOpShapeHelper(mlir::ONNXAveragePoolOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal)
      : ONNXGenericPoolShapeHelper<mlir::ONNXAveragePoolOp,
            mlir::ONNXAveragePoolOpAdaptor>(newOp, false /*hasFilter*/,
            newOp->ceil_mode(), rewriter, fGetDenseVal, fLoadVal) {}

  mlir::LogicalResult computeShape(
      mlir::ONNXAveragePoolOpAdaptor operandAdaptor);
};

// Shape for Reduction.
struct ONNXReduceSumOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXReduceSumOp> {
  ONNXReduceSumOpShapeHelper(mlir::ONNXReduceSumOp *newOp);
  ONNXReduceSumOpShapeHelper(mlir::ONNXReduceSumOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXReduceSumOpAdaptor operandAdaptor);
};

// Shape for ReshapeOp.
struct ONNXReshapeOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXReshapeOp> {
  ONNXReshapeOpShapeHelper(mlir::ONNXReshapeOp *newOp);
  ONNXReshapeOpShapeHelper(mlir::ONNXReshapeOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXReshapeOpAdaptor operandAdaptor);
};

// Shape for ONNXReverseSequence.
struct ONNXReverseSequenceOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXReverseSequenceOp> {
  ONNXReverseSequenceOpShapeHelper(
      mlir::ONNXReverseSequenceOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXReverseSequenceOpShapeHelper(mlir::ONNXReverseSequenceOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);

  mlir::LogicalResult Compute(
      mlir::ONNXReverseSequenceOpAdaptor operandAdaptor);
};

// Shape for SqueezeOp.
struct ONNXSqueezeOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXSqueezeOp> {
  ONNXSqueezeOpShapeHelper(mlir::ONNXSqueezeOp *newOp);
  ONNXSqueezeOpShapeHelper(mlir::ONNXSqueezeOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXSqueezeOpAdaptor operandAdaptor);
};

// Shape for SqueezeV11Op.
struct ONNXSqueezeV11OpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXSqueezeV11Op> {
  ONNXSqueezeV11OpShapeHelper(mlir::ONNXSqueezeV11Op *newOp);
  ONNXSqueezeV11OpShapeHelper(mlir::ONNXSqueezeV11Op *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(
      mlir::ONNXSqueezeV11OpAdaptor operandAdaptor);
};

// Shape for UnsqueezeOp.
struct ONNXUnsqueezeOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXUnsqueezeOp> {
  ONNXUnsqueezeOpShapeHelper(mlir::ONNXUnsqueezeOp *newOp);
  ONNXUnsqueezeOpShapeHelper(mlir::ONNXUnsqueezeOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXUnsqueezeOpAdaptor operandAdaptor);
};

// Shape for UnsqueezeV11Op.
struct ONNXUnsqueezeV11OpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXUnsqueezeV11Op> {
  ONNXUnsqueezeV11OpShapeHelper(mlir::ONNXUnsqueezeV11Op *newOp);
  ONNXUnsqueezeV11OpShapeHelper(mlir::ONNXUnsqueezeV11Op *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(
      mlir::ONNXUnsqueezeV11OpAdaptor operandAdaptor);
};

// Shape for ONNXShapeOp.
struct ONNXShapeOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXShapeOp> {
  ONNXShapeOpShapeHelper(
      mlir::ONNXShapeOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXShapeOpShapeHelper(mlir::ONNXShapeOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);
  mlir::LogicalResult computeShape(mlir::ONNXShapeOpAdaptor operandAdaptor);
  // Additional data for ShapeOp.
  DimsExpr selectedData;
};

// Shape for PadOp.
struct ONNXPadOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXPadOp> {
  ONNXPadOpShapeHelper(mlir::ONNXPadOp *newOp);
  ONNXPadOpShapeHelper(mlir::ONNXPadOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXPadOpAdaptor operandAdaptor);
  // Additional data for PadOp.
  llvm::SmallVector<IndexExpr, 4> pads;
};

// Shape for ONNXExpandOp.
struct ONNXExpandOpShapeHelper
    : public ONNXOpBroadcastedShapeHelper<mlir::ONNXExpandOp> {
  ONNXExpandOpShapeHelper(mlir::ONNXExpandOp *newOp);
  ONNXExpandOpShapeHelper(mlir::ONNXExpandOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXExpandOpAdaptor operandAdaptor);
  // Additional data for ExpandOp.
  mlir::ONNXExpandOp *expandOp;
};

// Shape for OneHotOp.
struct ONNXOneHotOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXOneHotOp> {
  ONNXOneHotOpShapeHelper(mlir::ONNXOneHotOp *newOp);
  ONNXOneHotOpShapeHelper(mlir::ONNXOneHotOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  mlir::LogicalResult computeShape(mlir::ONNXOneHotOpAdaptor operandAdaptor);
  // Additional data for ExpandOp.
  int64_t axis = -1; // Default value.
  IndexExpr depth;   // Depth which may/maynot be known at compile time.
};

// Shape for ONNXCompressOp.
struct ONNXCompressOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXCompressOp> {
  ONNXCompressOpShapeHelper(mlir::ONNXCompressOp *newOp);
  ONNXCompressOpShapeHelper(mlir::ONNXCompressOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXCompressOpAdaptor operandAdaptor);
};

// Shape for ONNXTopKOp.
struct ONNXTopKOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXTopKOp> {
  ONNXTopKOpShapeHelper(mlir::ONNXTopKOp *newOp);
  ONNXTopKOpShapeHelper(mlir::ONNXTopKOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXTopKOpAdaptor operandAdaptor);
};

// Shape for ONNXCategoryMapperOp.
struct ONNXCategoryMapperOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXCategoryMapperOp> {
  ONNXCategoryMapperOpShapeHelper(mlir::ONNXCategoryMapperOp *newOp);
  ONNXCategoryMapperOpShapeHelper(mlir::ONNXCategoryMapperOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(
      mlir::ONNXCategoryMapperOpAdaptor operandAdaptor);
};

// Shape for ONNXRoiAlignOp
struct ONNXRoiAlignOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXRoiAlignOp> {
  ONNXRoiAlignOpShapeHelper(mlir::ONNXRoiAlignOp *newOp);
  ONNXRoiAlignOpShapeHelper(mlir::ONNXRoiAlignOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  mlir::LogicalResult computeShape(mlir::ONNXRoiAlignOpAdaptor operandAdaptor);
  // Additional data for RoiAlignOp.
  llvm::SmallVector<IndexExpr, 4> xDims;            // Dim of X.
  llvm::SmallVector<IndexExpr, 1> batchIndicesDims; // Dim of batch_indices.
};

} // namespace onnx_mlir
