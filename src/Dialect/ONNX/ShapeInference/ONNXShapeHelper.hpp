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

// Steps to add a new op XXX:
// 1) Create a new shape inference type inside this file, ONNXShapeHelper.hpp.
// 2) Create new shape inference file, say XXX.cpp and implement.
// 3) Add template instantiation at bottom of ONNXShapeHelper.cpp.
// 4) Add new file name XXX.cpp to ../CMakeLists.txt
// 5) Use the new object in ONNXOps.cpp and ONNXToKrnl lowering for XXX.

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

using DimsExpr = SmallVector<IndexExpr, 4>;

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
  ONNXOpShapeHelper(OP *newOp, int numResults, OpBuilder *rewriter,
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
  SmallVector<DimsExpr, 1> outputsDims;
  bool ownScope;
};

/// Compute a broadcasted shape from the shapes of given operands. Operands must
/// be ranked in advance.
template <class OP>
struct ONNXOpBroadcastedShapeHelper : public ONNXOpShapeHelper<OP> {
  ONNXOpBroadcastedShapeHelper(OP *newOp, IndexExprScope *inScope = nullptr,
      bool uniBroadcasting = false, bool noBroadcasting = false);

  ONNXOpBroadcastedShapeHelper(OP *newOp, OpBuilder *rewriter,
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
  LogicalResult computeShape(
      ArrayRef<Value> operands, DimsExpr &additionalOperand);

  // Compute access indices to load/store value from/to a given 'operand'.
  // Used in a loop to access the operand.
  // Parameters:
  //   - operand: operand to access.
  //   - operandIndex: index of the operand in 'this->inputsDims'.
  //   - loopAccessExprs: IndexExprs for the loop's IVs.
  //   - operandAccessExprs: access indices to access the operand.
  //     This is the output of this function. Use it in subsequent load/stores.
  LogicalResult GetAccessExprs(Value operand, unsigned operandIndex,
      const SmallVectorImpl<IndexExpr> &outputAccessExprs,
      SmallVectorImpl<IndexExpr> &operandAccessExprs);

  // A vector of input shapes where dimensions are padded with 1 if necessary,
  // so that all inputs have the same rank.
  SmallVector<DimsExpr, 4> inputsDims;
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
    : public ONNXOpBroadcastedShapeHelper<Operation> {
  ONNXGenericOpBroadcastedShapeHelper(Operation *newOp,
      IndexExprScope *inScope = nullptr, bool uniBroadcasting = false,
      bool noBroadcasting = false);
  ONNXGenericOpBroadcastedShapeHelper(Operation *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr, bool uniBroadcasting = false,
      bool noBroadcasting = false);
};

// Shape for ArgMax
struct ONNXArgMaxOpShapeHelper : public ONNXOpShapeHelper<ONNXArgMaxOp> {
  ONNXArgMaxOpShapeHelper(ONNXArgMaxOp *newOp);
  ONNXArgMaxOpShapeHelper(ONNXArgMaxOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXArgMaxOpAdaptor operandAdaptor);
};

// Shape for Clip.
struct ONNXClipOpShapeHelper : public ONNXOpShapeHelper<ONNXClipOp> {
  ONNXClipOpShapeHelper(ONNXClipOp *newOp);
  ONNXClipOpShapeHelper(ONNXClipOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXClipOpAdaptor operandAdaptor);
};

// Shape for concat
struct ONNXConcatOpShapeHelper : public ONNXOpShapeHelper<ONNXConcatOp> {
  ONNXConcatOpShapeHelper(ONNXConcatOp *newOp);
  ONNXConcatOpShapeHelper(ONNXConcatOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXConcatOpAdaptor operandAdaptor);
};

// Shape for DepthToSpace.
struct ONNXDepthToSpaceOpShapeHelper
    : public ONNXOpShapeHelper<ONNXDepthToSpaceOp> {
  ONNXDepthToSpaceOpShapeHelper(ONNXDepthToSpaceOp *newOp);
  ONNXDepthToSpaceOpShapeHelper(ONNXDepthToSpaceOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXDepthToSpaceOpAdaptor operandAdaptor);
};

// Shape for SliceOp.
struct ONNXSliceOpShapeHelper : public ONNXOpShapeHelper<ONNXSliceOp> {
  ONNXSliceOpShapeHelper(ONNXSliceOp *newOp);
  ONNXSliceOpShapeHelper(ONNXSliceOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXSliceOpAdaptor operandAdaptor);
  // Additional data for SliceOp.
  SmallVector<IndexExpr, 4> starts;
  SmallVector<IndexExpr, 4> ends;
  SmallVector<IndexExpr, 4> steps;
};

// Shape for Tile.
struct ONNXTileOpShapeHelper : public ONNXOpShapeHelper<ONNXTileOp> {
  ONNXTileOpShapeHelper(ONNXTileOp *newOp);
  ONNXTileOpShapeHelper(ONNXTileOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXTileOpAdaptor operandAdaptor);
};

// Shape for GemmOp. Rank of C is known, and its rank can be 0, 1, or 2. Each
// of the dimensions of C can have 1 (broadcast) or many (same size as position
// requires).
struct ONNXGemmOpShapeHelper : public ONNXOpShapeHelper<ONNXGemmOp> {
  ONNXGemmOpShapeHelper(ONNXGemmOp *newOp);
  ONNXGemmOpShapeHelper(ONNXGemmOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXGemmOpAdaptor operandAdaptor);
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
  ONNXMatMulOpShapeHelper(ONNXMatMulOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXMatMulOpAdaptor operandAdaptor);
  // Additional data for MatMulOp: output = a & b.
  SmallVector<IndexExpr, 4> aDims; // Dim of A, after applying padding.
  SmallVector<IndexExpr, 4> bDims; // Dim of B, after applying padding.
  llvm::BitVector aPadDims;        // When true, that dim was padded.
  llvm::BitVector bPadDims;        // When true, that dim was padded.
};

// Shape for Gather.
struct ONNXGatherOpShapeHelper : public ONNXOpShapeHelper<ONNXGatherOp> {
  ONNXGatherOpShapeHelper(ONNXGatherOp *newOp);
  ONNXGatherOpShapeHelper(ONNXGatherOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXGatherOpAdaptor operandAdaptor);
  // Additional data for GatherOp.
  SmallVector<IndexExpr, 4> dataDims;    // Dim of data.
  SmallVector<IndexExpr, 4> indicesDims; // Dim of indices.
  bool positiveConstantIndices; // True when all indices are positive consants.
};

// Shape for SpaceToDepth.
struct ONNXSpaceToDepthOpShapeHelper
    : public ONNXOpShapeHelper<ONNXSpaceToDepthOp> {
  ONNXSpaceToDepthOpShapeHelper(ONNXSpaceToDepthOp *newOp);
  ONNXSpaceToDepthOpShapeHelper(ONNXSpaceToDepthOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXSpaceToDepthOpAdaptor operandAdaptor);
};

// Shape for SplitOp.
struct ONNXSplitOpShapeHelper : public ONNXOpShapeHelper<ONNXSplitOp> {
  ONNXSplitOpShapeHelper(ONNXSplitOp *newOp);
  ONNXSplitOpShapeHelper(ONNXSplitOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXSplitOpAdaptor operandAdaptor);
};

// Shape for SplitV11Op.
struct ONNXSplitV11OpShapeHelper : public ONNXOpShapeHelper<ONNXSplitV11Op> {
  ONNXSplitV11OpShapeHelper(ONNXSplitV11Op *newOp);
  ONNXSplitV11OpShapeHelper(ONNXSplitV11Op *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXSplitV11OpAdaptor operandAdaptor);
};

// Shape for TransposeOp.
struct ONNXTransposeOpShapeHelper : public ONNXOpShapeHelper<ONNXTransposeOp> {
  ONNXTransposeOpShapeHelper(ONNXTransposeOp *newOp);
  ONNXTransposeOpShapeHelper(ONNXTransposeOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXTransposeOpAdaptor operandAdaptor);
};

// Shape for LRN.
struct ONNXLRNOpShapeHelper : public ONNXOpShapeHelper<ONNXLRNOp> {
  ONNXLRNOpShapeHelper(ONNXLRNOp *newOp);
  ONNXLRNOpShapeHelper(ONNXLRNOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXLRNOpAdaptor operandAdaptor);
};

// Shape for generic pooling/conv ops.
template <typename OP_TYPE, typename OP_ADAPTOR>
struct ONNXGenericPoolShapeHelper : public ONNXOpShapeHelper<OP_TYPE> {
  ONNXGenericPoolShapeHelper(OP_TYPE *newOp, bool hasFilter, bool ceilMode)
      : ONNXOpShapeHelper<OP_TYPE>(
            newOp, newOp->getOperation()->getNumResults()),
        hasFilter(hasFilter), ceilMode(ceilMode) {}

  ONNXGenericPoolShapeHelper(OP_TYPE *newOp, bool hasFilter, bool ceilMode,
      OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal)
      : ONNXOpShapeHelper<OP_TYPE>(newOp,
            newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
            fLoadVal),
        hasFilter(hasFilter), ceilMode(ceilMode) {}

  LogicalResult computeShape(OP_ADAPTOR operandAdaptor, Value filterValue,
      Optional<ArrayAttr> kernelShapeOpt, Optional<ArrayAttr> padOpt,
      Optional<ArrayAttr> strideOpt, Optional<ArrayAttr> dilationOpt);
  // Additional data for Pool operations.
  bool hasFilter; // If has filter, it also has CO and optional kernel.
  bool ceilMode;  // Use ceil or floor for auto_pad=NOTSET policy.
  // Values set by Compute.
  SmallVector<IndexExpr, 2> kernelShape;
  SmallVector<IndexExpr, 4> pads;
  SmallVector<int64_t, 2> strides;
  SmallVector<int64_t, 2> dilations;
};

// Shape for Conv.
struct ONNXConvOpShapeHelper
    : public ONNXGenericPoolShapeHelper<ONNXConvOp, ONNXConvOpAdaptor> {
  ONNXConvOpShapeHelper(ONNXConvOp *newOp);
  ONNXConvOpShapeHelper(ONNXConvOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXConvOpAdaptor operandAdaptor);
};

// Shape for MaxPoolSingleOut.
struct ONNXMaxPoolSingleOutOpShapeHelper
    : public ONNXGenericPoolShapeHelper<ONNXMaxPoolSingleOutOp,
          ONNXMaxPoolSingleOutOpAdaptor> {
  ONNXMaxPoolSingleOutOpShapeHelper(ONNXMaxPoolSingleOutOp *newOp)
      : ONNXGenericPoolShapeHelper<ONNXMaxPoolSingleOutOp,
            ONNXMaxPoolSingleOutOpAdaptor>(
            newOp, false /*hasFilter*/, newOp->ceil_mode()) {}

  ONNXMaxPoolSingleOutOpShapeHelper(ONNXMaxPoolSingleOutOp *newOp,
      OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal)
      : ONNXGenericPoolShapeHelper<ONNXMaxPoolSingleOutOp,
            ONNXMaxPoolSingleOutOpAdaptor>(newOp, false /*hasFilter*/,
            newOp->ceil_mode(), rewriter, fGetDenseVal, fLoadVal) {}

  LogicalResult computeShape(ONNXMaxPoolSingleOutOpAdaptor operandAdaptor);
};

// Shape for ONNXAveragePoolOp
struct ONNXAveragePoolOpShapeHelper
    : public ONNXGenericPoolShapeHelper<ONNXAveragePoolOp,
          ONNXAveragePoolOpAdaptor> {
  ONNXAveragePoolOpShapeHelper(ONNXAveragePoolOp *newOp)
      : ONNXGenericPoolShapeHelper<ONNXAveragePoolOp, ONNXAveragePoolOpAdaptor>(
            newOp, false /*hasFilter*/, newOp->ceil_mode()) {}

  ONNXAveragePoolOpShapeHelper(ONNXAveragePoolOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal)
      : ONNXGenericPoolShapeHelper<ONNXAveragePoolOp, ONNXAveragePoolOpAdaptor>(
            newOp, false /*hasFilter*/, newOp->ceil_mode(), rewriter,
            fGetDenseVal, fLoadVal) {}

  LogicalResult computeShape(ONNXAveragePoolOpAdaptor operandAdaptor);
};

// Shape for ReshapeOp.
struct ONNXReshapeOpShapeHelper : public ONNXOpShapeHelper<ONNXReshapeOp> {
  ONNXReshapeOpShapeHelper(ONNXReshapeOp *newOp);
  ONNXReshapeOpShapeHelper(ONNXReshapeOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXReshapeOpAdaptor operandAdaptor);
};

// Shape for ONNXReverseSequence.
struct ONNXReverseSequenceOpShapeHelper
    : public ONNXOpShapeHelper<ONNXReverseSequenceOp> {
  ONNXReverseSequenceOpShapeHelper(
      ONNXReverseSequenceOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXReverseSequenceOpShapeHelper(ONNXReverseSequenceOp *newOp,
      OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);

  LogicalResult Compute(ONNXReverseSequenceOpAdaptor operandAdaptor);
};

// Shape for SqueezeOp.
struct ONNXSqueezeOpShapeHelper : public ONNXOpShapeHelper<ONNXSqueezeOp> {
  ONNXSqueezeOpShapeHelper(ONNXSqueezeOp *newOp);
  ONNXSqueezeOpShapeHelper(ONNXSqueezeOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXSqueezeOpAdaptor operandAdaptor);
};

// Shape for SqueezeV11Op.
struct ONNXSqueezeV11OpShapeHelper
    : public ONNXOpShapeHelper<ONNXSqueezeV11Op> {
  ONNXSqueezeV11OpShapeHelper(ONNXSqueezeV11Op *newOp);
  ONNXSqueezeV11OpShapeHelper(ONNXSqueezeV11Op *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXSqueezeV11OpAdaptor operandAdaptor);
};

// Shape for UnsqueezeOp.
struct ONNXUnsqueezeOpShapeHelper : public ONNXOpShapeHelper<ONNXUnsqueezeOp> {
  ONNXUnsqueezeOpShapeHelper(ONNXUnsqueezeOp *newOp);
  ONNXUnsqueezeOpShapeHelper(ONNXUnsqueezeOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXUnsqueezeOpAdaptor operandAdaptor);
};

// Shape for UnsqueezeV11Op.
struct ONNXUnsqueezeV11OpShapeHelper
    : public ONNXOpShapeHelper<ONNXUnsqueezeV11Op> {
  ONNXUnsqueezeV11OpShapeHelper(ONNXUnsqueezeV11Op *newOp);
  ONNXUnsqueezeV11OpShapeHelper(ONNXUnsqueezeV11Op *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXUnsqueezeV11OpAdaptor operandAdaptor);
};

// Shape for ONNXShapeOp.
struct ONNXShapeOpShapeHelper : public ONNXOpShapeHelper<ONNXShapeOp> {
  ONNXShapeOpShapeHelper(ONNXShapeOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXShapeOpShapeHelper(ONNXShapeOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);
  LogicalResult computeShape(ONNXShapeOpAdaptor operandAdaptor);
  // Additional data for ShapeOp.
  DimsExpr selectedData;
};

// Shape for PadOp.
struct ONNXPadOpShapeHelper : public ONNXOpShapeHelper<ONNXPadOp> {
  ONNXPadOpShapeHelper(ONNXPadOp *newOp);
  ONNXPadOpShapeHelper(ONNXPadOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXPadOpAdaptor operandAdaptor);
  // Additional data for PadOp.
  SmallVector<IndexExpr, 4> pads;
};

// Shape for ONNXExpandOp.
struct ONNXExpandOpShapeHelper
    : public ONNXOpBroadcastedShapeHelper<ONNXExpandOp> {
  ONNXExpandOpShapeHelper(ONNXExpandOp *newOp);
  ONNXExpandOpShapeHelper(ONNXExpandOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXExpandOpAdaptor operandAdaptor);
  // Additional data for ExpandOp.
  ONNXExpandOp *expandOp;
};

// Shape for OneHotOp.
struct ONNXOneHotOpShapeHelper : public ONNXOpShapeHelper<ONNXOneHotOp> {
  ONNXOneHotOpShapeHelper(ONNXOneHotOp *newOp);
  ONNXOneHotOpShapeHelper(ONNXOneHotOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);

  LogicalResult computeShape(ONNXOneHotOpAdaptor operandAdaptor);
  // Additional data for ExpandOp.
  int64_t axis = -1; // Default value.
  IndexExpr depth;   // Depth which may/maynot be known at compile time.
};

// Shape for ONNXCompressOp.
struct ONNXCompressOpShapeHelper : public ONNXOpShapeHelper<ONNXCompressOp> {
  ONNXCompressOpShapeHelper(ONNXCompressOp *newOp);
  ONNXCompressOpShapeHelper(ONNXCompressOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXCompressOpAdaptor operandAdaptor);
  // Additional data for CompressOp.
  int axis = -1; // Value -1 signify axis was not specified.
};

// Shape for ONNXTopKOp.
struct ONNXTopKOpShapeHelper : public ONNXOpShapeHelper<ONNXTopKOp> {
  ONNXTopKOpShapeHelper(ONNXTopKOp *newOp);
  ONNXTopKOpShapeHelper(ONNXTopKOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXTopKOpAdaptor operandAdaptor);
};

// Shape for ONNXCategoryMapperOp.
struct ONNXCategoryMapperOpShapeHelper
    : public ONNXOpShapeHelper<ONNXCategoryMapperOp> {
  ONNXCategoryMapperOpShapeHelper(ONNXCategoryMapperOp *newOp);
  ONNXCategoryMapperOpShapeHelper(ONNXCategoryMapperOp *newOp,
      OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXCategoryMapperOpAdaptor operandAdaptor);
};

// Shape for ONNXRoiAlignOp
struct ONNXRoiAlignOpShapeHelper : public ONNXOpShapeHelper<ONNXRoiAlignOp> {
  ONNXRoiAlignOpShapeHelper(ONNXRoiAlignOp *newOp);
  ONNXRoiAlignOpShapeHelper(ONNXRoiAlignOp *newOp, OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal);
  LogicalResult computeShape(ONNXRoiAlignOpAdaptor operandAdaptor);
  // Additional data for RoiAlignOp.
  SmallVector<IndexExpr, 4> xDims;            // Dim of X.
  SmallVector<IndexExpr, 1> batchIndicesDims; // Dim of batch_indices.
};
