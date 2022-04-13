/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ONNXShapeHelper.hpp - help for shapes ---------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
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
  ONNXOpShapeHelper(OP *newOp, int numResults, IndexExprScope *inScope);
  // Constructor when code can be generated. Reuse scope if given, otherwise
  // create one now and free it in destructor.
  ONNXOpShapeHelper(OP *newOp, int numResults, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope);
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
      IndexExprScope *inScope = nullptr, bool uniBroadcasting = false,
      bool noBroadcasting = false);
  ONNXGenericOpBroadcastedShapeHelper(mlir::Operation *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr, bool uniBroadcasting = false,
      bool noBroadcasting = false);
};

// Shape for generic pooling/conv ops.
template <typename OP_TYPE, typename OP_ADAPTOR>
struct ONNXGenericPoolShapeHelper : public ONNXOpShapeHelper<OP_TYPE> {
  ONNXGenericPoolShapeHelper(
      OP_TYPE *newOp, bool hasFilter, bool ceilMode, IndexExprScope *inScope)
      : ONNXOpShapeHelper<OP_TYPE>(
            newOp, newOp->getOperation()->getNumResults(), inScope),
        hasFilter(hasFilter), ceilMode(ceilMode) {}

  ONNXGenericPoolShapeHelper(OP_TYPE *newOp, bool hasFilter, bool ceilMode,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
      : ONNXOpShapeHelper<OP_TYPE>(newOp,
            newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
            fLoadVal, inScope),
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

#define DECLARE_SHAPE_HELPER(OpName)                                           \
  class OpName##ShapeHelper : public ONNXOpShapeHelper<mlir::OpName> {         \
  public:                                                                      \
    OpName##ShapeHelper(                                                       \
        mlir::OpName *newOp, IndexExprScope *inScope = nullptr)                \
        : ONNXOpShapeHelper<mlir::OpName>(                                     \
              newOp, newOp->getOperation()->getNumResults(), inScope) {}       \
    OpName##ShapeHelper(mlir::OpName *newOp, mlir::OpBuilder *rewriter,        \
        ArrayValueIndexCapture::GetDenseVal fGetDenseVal,                      \
        ArrayValueIndexCapture::LoadVal fLoadVal,                              \
        IndexExprScope *inScope = nullptr)                                     \
        : ONNXOpShapeHelper<mlir::OpName>(newOp,                               \
              newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,  \
              fLoadVal, inScope) {}                                            \
    mlir::LogicalResult computeShape(mlir::OpName##Adaptor operandAdaptor);    \
  };
DECLARE_SHAPE_HELPER(ONNXArgMaxOp)
DECLARE_SHAPE_HELPER(ONNXArgMinOp)
DECLARE_SHAPE_HELPER(ONNXCategoryMapperOp)
DECLARE_SHAPE_HELPER(ONNXClipOp)
DECLARE_SHAPE_HELPER(ONNXCompressOp)
DECLARE_SHAPE_HELPER(ONNXConcatOp)
DECLARE_SHAPE_HELPER(ONNXDepthToSpaceOp)
DECLARE_SHAPE_HELPER(ONNXLRNOp)
DECLARE_SHAPE_HELPER(ONNXReduceSumOp)
DECLARE_SHAPE_HELPER(ONNXReshapeOp)
DECLARE_SHAPE_HELPER(ONNXReverseSequenceOp)
DECLARE_SHAPE_HELPER(ONNXShapeOp)
DECLARE_SHAPE_HELPER(ONNXSpaceToDepthOp)
DECLARE_SHAPE_HELPER(ONNXSplitOp)
DECLARE_SHAPE_HELPER(ONNXSplitV11Op)
DECLARE_SHAPE_HELPER(ONNXSqueezeOp)
DECLARE_SHAPE_HELPER(ONNXSqueezeV11Op)
DECLARE_SHAPE_HELPER(ONNXTileOp)
DECLARE_SHAPE_HELPER(ONNXTopKOp)
DECLARE_SHAPE_HELPER(ONNXTransposeOp)
DECLARE_SHAPE_HELPER(ONNXUnsqueezeOp)
DECLARE_SHAPE_HELPER(ONNXUnsqueezeV11Op)
#undef DECLARE_SHAPE_HELPER

// Compute the data selected by the Shape operator.
DimsExpr computeSelectedData(mlir::ONNXShapeOpAdaptor &operandAdaptor);

// Shape for SliceOp.
struct ONNXSliceOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXSliceOp> {
  ONNXSliceOpShapeHelper(
      mlir::ONNXSliceOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXSliceOpShapeHelper(mlir::ONNXSliceOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);
  mlir::LogicalResult computeShape(mlir::ONNXSliceOpAdaptor operandAdaptor);
  // Additional data for SliceOp.
  llvm::SmallVector<IndexExpr, 4> starts, ends, steps;
};

// Shape for GemmOp. Rank of C is known, and its rank can be 0, 1,
// or 2. Each of the dimensions of C can have 1 (broadcast) or
// many (same size as position requires).
struct ONNXGemmOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXGemmOp> {
  ONNXGemmOpShapeHelper(
      mlir::ONNXGemmOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXGemmOpShapeHelper(mlir::ONNXGemmOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);
  mlir::LogicalResult computeShape(mlir::ONNXGemmOpAdaptor operandAdaptor);
  // Additional data for GemmOp: output = a * b.
  llvm::SmallVector<IndexExpr, 4> aDims,
      bDims; // Dim after applying transpose.
  llvm::SmallVector<IndexExpr, 4>
      cDims;    // Dim of C, padding "1" when broadcast.
  bool hasBias; // Whether there is a bias (aka C exists).
  int cRank;    // Dim of the original C (not padding dims by 1).
};

// Shape for MatMulOp.
struct ONNXMatMulOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXMatMulOp> {
  ONNXMatMulOpShapeHelper(
      mlir::ONNXMatMulOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXMatMulOpShapeHelper(mlir::ONNXMatMulOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);
  mlir::LogicalResult computeShape(mlir::ONNXMatMulOpAdaptor operandAdaptor);
  // Additional data for MatMulOp: output = a & b.
  llvm::SmallVector<IndexExpr, 4> aDims,
      bDims; // Dim after applying padding.
  llvm::BitVector aPadDims,
      bPadDims; // When true, that dim was padded.
};

// Shape for Gather.
struct ONNXGatherOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXGatherOp> {
  ONNXGatherOpShapeHelper(
      mlir::ONNXGatherOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXGatherOpShapeHelper(mlir::ONNXGatherOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);
  mlir::LogicalResult computeShape(mlir::ONNXGatherOpAdaptor operandAdaptor);
  // Additional data for GatherOp.
  llvm::SmallVector<IndexExpr, 4> dataDims, indicesDims;
  bool positiveConstantIndices; // True when all indices are
                                // positive consants.
};

// Shape for PadOp.
struct ONNXPadOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXPadOp> {
  ONNXPadOpShapeHelper(
      mlir::ONNXPadOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXPadOpShapeHelper(mlir::ONNXPadOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);
  mlir::LogicalResult computeShape(mlir::ONNXPadOpAdaptor operandAdaptor);
  // Additional data for PadOp.
  llvm::SmallVector<IndexExpr, 4> pads;
};

// Shape for OneHotOp.
struct ONNXOneHotOpShapeHelper : public ONNXOpShapeHelper<mlir::ONNXOneHotOp> {
  ONNXOneHotOpShapeHelper(
      mlir::ONNXOneHotOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXOneHotOpShapeHelper(mlir::ONNXOneHotOp *newOp, mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);
  mlir::LogicalResult computeShape(mlir::ONNXOneHotOpAdaptor operandAdaptor);
  // Additional data for ExpandOp.
  int64_t axis = -1; // Default value.
  IndexExpr depth;   // Depth which may/maynot be known at compile time.
};

// Shape for ONNXRoiAlignOp
struct ONNXRoiAlignOpShapeHelper
    : public ONNXOpShapeHelper<mlir::ONNXRoiAlignOp> {
  ONNXRoiAlignOpShapeHelper(
      mlir::ONNXRoiAlignOp *newOp, IndexExprScope *inScope = nullptr);
  ONNXRoiAlignOpShapeHelper(mlir::ONNXRoiAlignOp *newOp,
      mlir::OpBuilder *rewriter,
      ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
      ArrayValueIndexCapture::LoadVal fLoadVal,
      IndexExprScope *inScope = nullptr);
  mlir::LogicalResult computeShape(mlir::ONNXRoiAlignOpAdaptor operandAdaptor);
  // Additional data for RoiAlignOp.
  llvm::SmallVector<IndexExpr, 4> xDims;            // Dim of X.
  llvm::SmallVector<IndexExpr, 1> batchIndicesDims; // Dim of batch_indices.
};

#define DECLARE_POOL_SHAPE_HELPER(OpName)                                      \
  class OpName##ShapeHelper : public ONNXGenericPoolShapeHelper<mlir::OpName,  \
                                  mlir::OpName##Adaptor> {                     \
  public:                                                                      \
    OpName##ShapeHelper(                                                       \
        mlir::OpName *newOp, IndexExprScope *inScope = nullptr);               \
    OpName##ShapeHelper(mlir::OpName *newOp, mlir::OpBuilder *rewriter,        \
        ArrayValueIndexCapture::GetDenseVal fGetDenseVal,                      \
        ArrayValueIndexCapture::LoadVal fLoadVal,                              \
        IndexExprScope *inScope = nullptr);                                    \
    mlir::LogicalResult computeShape(mlir::OpName##Adaptor operandAdaptor);    \
  };
DECLARE_POOL_SHAPE_HELPER(ONNXAveragePoolOp)
DECLARE_POOL_SHAPE_HELPER(ONNXConvOp)
DECLARE_POOL_SHAPE_HELPER(ONNXMaxPoolSingleOutOp)
#undef DECLARE_POOL_SHAPE_HELPER

#define DECLARE_BROADCASTED_SHAPE_HELPER(OpName)                               \
  class OpName##ShapeHelper                                                    \
      : public ONNXOpBroadcastedShapeHelper<mlir::OpName> {                    \
  public:                                                                      \
    OpName##ShapeHelper(                                                       \
        mlir::OpName *newOp, IndexExprScope *inScope = nullptr)                \
        : ONNXOpBroadcastedShapeHelper<mlir::OpName>(newOp, inScope) {}        \
    OpName##ShapeHelper(mlir::OpName *newOp, mlir::OpBuilder *rewriter,        \
        ArrayValueIndexCapture::GetDenseVal fGetDenseVal,                      \
        ArrayValueIndexCapture::LoadVal fLoadVal,                              \
        IndexExprScope *inScope = nullptr)                                     \
        : ONNXOpBroadcastedShapeHelper<mlir::OpName>(                          \
              newOp, rewriter, fGetDenseVal, fLoadVal, inScope) {}             \
    mlir::LogicalResult computeShape(mlir::OpName##Adaptor operandAdaptor);    \
  };
DECLARE_BROADCASTED_SHAPE_HELPER(ONNXExpandOp)
#undef DECLARE_BROADCASTED_SHAPE_HELPER

} // namespace onnx_mlir
