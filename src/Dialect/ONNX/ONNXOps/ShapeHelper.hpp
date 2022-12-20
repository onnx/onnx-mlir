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

#include <utility>

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

using DimsExpr = llvm::SmallVector<IndexExpr, 4>;

//===----------------------------------------------------------------------===//
// Top shape helper class
//===----------------------------------------------------------------------===//

struct ONNXOpShapeHelper {
  /* Constructor for shape inference.

   This class and its specialized subclasses are used in one of two situation:
   1) For shape analysis (where no code is generated) during shape inference.
   2) For shape generation (where runtime shapes are computed using generated
   code) during lowering.

   @param op Operation to be analyzed.

   @param operands Operands of the operation to be analyzed. When passing an
   empty list, the operands are taken from the operations. When passing an
   explicit, non-empty list, these operands are used instead of the ones from
   the operation.

   The former option (empty list) is used during shape inference.

   The later option (explicit list) is used during lowering, as typically there
   are two sets of operands, the ones already lowered (explicit list) and the
   original ones (contained within the op). Using shapes during lowering
   typically deals with already-lowered operands.

   However, during lowering, it may be sometime advantageous to perform the
   analysis of the index expressions in the "old" dialect, e.g. in ONNX instead
   of the destination dialect. To enable this, just pass `{}` as operands and
   the original operands associated with the unmodified operation will be used.

   @param ieBuilder Class that scans the operands to gather IndexExpr from them.
   Typically used to gather shape and constant values.

   During shape inference, we typically use IndexExprBuilderForAnalysis
   (src/Dialect/Mlir/DialectBuilder.hpp), which uses questionmark for values
   unkown at compile time. This builder is default when no ieBuilder is given.

   During lowering, we typically use and Index Expr Builder that generates code
   for values unknown at compile time. Example of such subclasses are
   IndexExprBuilderForKrnl (generates Krnl ops, in
   src/Dialect/Krnl/DialectBuilder.hpp, ) or IndexExprBuilderForMhlo (generates
   Shape/MHLO ops, in src/Conversion/ONNXToMhlo/DialectBuilder.hpp).

   @param scope Index expression scope to be used. If none is provided, a new
   scope is created and stored internally. This scope will then be destructed
   when the current object is destructed.

   Passing a scope is critically important when, to evaluate the shape of a
   given operation, we must also analysis the shape of an other operation. Both
   shape helper MUST share the same scope as otherwise there will be references
   to "deleted" index expressions (as all index expressions are deleted when its
   directly enclosing scope vanishes).
   */

  ONNXOpShapeHelper(mlir::Operation *op,    /* Op to be analyzed. */
      mlir::ArrayRef<mlir::Value> operands, /* If empty, use operands from op.*/
      IndexExprBuilder *ieBuilder, /* Use IndexExprBuilderForAnalysis if null.*/
      IndexExprScope *scope);      /* Install local scope if null. */
  virtual ~ONNXOpShapeHelper();

  // Every leaf class is expected to create a computeShape with the following
  // signature. This method is responsible to compute at a minimum the output
  // dims.
  virtual mlir::LogicalResult computeShape() = 0;
  // Helper function that set n'th output dims from the given value.
  mlir::LogicalResult computeShapeFromOperand(mlir::Value operand, int n = 0);

  // Compute shape and assert on failure.
  void computeShapeAndAssertOnFailure();

  // Invoke the virtual computeShape, and on success, update the types of the
  // original operation. First call is used for operations where all the results
  // share the same output type, second for operations where all results have
  // their own output types.
  mlir::LogicalResult computeShapeAndUpdateType(
      mlir::Type elementType, mlir::Attribute encoding = nullptr);
  // If encoding list can be empty or have one entry per type.
  mlir::LogicalResult computeShapeAndUpdateTypes(
      mlir::TypeRange elementTypeRange,
      mlir::ArrayRef<mlir::Attribute> encodingList = {});

  // Get/set output dims for the N-th output dimension as Index Expressions.
  // Scalar may have a DimsExpr that is empty.
  DimsExpr &getOutputDims(int n = 0) { return privateOutputsDims[n]; }
  void setOutputDims(const DimsExpr &inferredDims, int n = 0);

  // Obtain the n-th output result as value.
  mlir::Value getOutput(int n = 0) { return op->getResult(n); }

  // Get index expression scope and operation.
  IndexExprScope *getScope() { return scope; }
  mlir::Operation *getOp() { return op; }

protected:
  // Data that must be present for every ShapeHelper operation. Op and scope
  // are initialized in the constructor.
  mlir::Operation *op;
  mlir::ArrayRef<mlir::Value> operands;
  IndexExprBuilder *createIE;
  IndexExprScope *scope;

private:
  //  outputsDims is computed by the child's struct `computeShape` function. It
  //  can be set using setOutputDims and retrieved using getOutputDims.
  llvm::SmallVector<DimsExpr, 1> privateOutputsDims;
  // Used to cache the operation's operands (shape inference only).
  llvm::SmallVector<mlir::Value> privateOperandsCache;
  bool ownScope, ownBuilder;
};

// Update a tensor type by using the given shape, elementType and encoding.
// TODO: when all ops are migrated to the new scheme, make this function private
// to ONNXOpShapeHelper.
void updateType(mlir::Value val, llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType = nullptr, mlir::Attribute encoding = nullptr,
    bool refineShape = true);

//===----------------------------------------------------------------------===//
// Unary Ops
//===----------------------------------------------------------------------===//

/// Compute an output shape for a unary element-wise operation. The output and
/// input of an unary element-wise operation have the same shape.
struct ONNXUnaryOpShapeHelper : public ONNXOpShapeHelper {
  ONNXUnaryOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXUnaryOpShapeHelper() {}

  mlir::LogicalResult computeShape() final {
    return computeShapeFromOperand(operands[0]);
  }
};

// Handle shape inference for unary element-wise operators. Perform the entire
// operation: (1) create the shape helper, compute shape, apply shape to the
// output types.
mlir::LogicalResult inferShapeForUnaryOps(mlir::Operation *op);
// Same as above, but allow a type change.
mlir::LogicalResult inferShapeForUnaryOps(
    mlir::Operation *op, mlir::Type elementType);
// Same as above, but allow a type and encoding change.
mlir::LogicalResult inferShapeForUnaryOps(
    mlir::Operation *op, mlir::Type elementType, mlir::Attribute encoding);

//===----------------------------------------------------------------------===//
// Broadcast Ops
//===----------------------------------------------------------------------===//

// Compute a broadcasted shape from the shapes of given operands. Operands must
// be ranked in advance.
struct ONNXBroadcastOpShapeHelper : public ONNXOpShapeHelper {
  ONNXBroadcastOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr,
      bool hasUniBroadcasting = false, bool hasNoBroadcasting = false)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope), inputsDims(),
        outputRank(0), hasUniBroadcasting(hasUniBroadcasting),
        hasNoBroadcasting(hasNoBroadcasting) {}
  virtual ~ONNXBroadcastOpShapeHelper() {}

  // Custom shape compute which takes additional parameters.
  mlir::LogicalResult customComputeShape(
      mlir::ArrayRef<mlir::Value> initialOperands, DimsExpr *additionalOperand);

  // Default shape compute (every operands of the operation and no additional
  // parameters).
  mlir::LogicalResult computeShape() override {
    return customComputeShape(operands, nullptr);
  }

  // Compute access indices to load/store value from/to a given 'operand'.
  // Used in a loop to access the operand.
  // Parameters:
  //   - operand: operand to access.
  //   - operandIndex: index of the operand in 'this->inputsDims'.
  //   - loopAccessExprs: IndexExprs for the loop's IVs.
  //   - operandAccessExprs: access indices to access the operand.
  //     This is the output of this function. Use it in subsequent load/stores.
  mlir::LogicalResult getAccessExprs(mlir::Value operand, uint64_t i,
      const llvm::SmallVectorImpl<IndexExpr> &outputAccessExprs,
      llvm::SmallVectorImpl<IndexExpr> &operandAccessExprs);

  // A vector of input shapes where dimensions are padded with 1 if necessary,
  // so that all inputs have the same rank. Instantiated during ComputeShape.
  llvm::SmallVector<DimsExpr, 4> inputsDims;
  // A vector of IndexExprs representing the output shape.
  // in upper DimsExpr outputDims;
  uint64_t outputRank;

protected:
  // When unidirectional broadcasting is true, the other operands are always
  // unidirectional broadcastable to the first operand.
  bool hasUniBroadcasting;
  // When isNoBroadcasting is true, the shape of all input is assumed to be
  // same. This flag is used to test dynamic shape. There is no impact on static
  // shape.
  bool hasNoBroadcasting;
};

// Helper for ExpandOp
struct ONNXExpandOpShapeHelper : public ONNXBroadcastOpShapeHelper {
  ONNXExpandOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXBroadcastOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXExpandOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

//===----------------------------------------------------------------------===//
// Shape op
//===----------------------------------------------------------------------===//

struct ONNXShapeOpShapeHelper : public ONNXOpShapeHelper {
  ONNXShapeOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope), start(-1), end(-1) {}
  virtual ~ONNXShapeOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Compute the shape values of input data for dimensions between start and
  // end.
  void computeSelectedDataShape(DimsExpr &selectedDataShape);
  // Compute start & end value without calls to computeShape.
  static void getStartEndValues(
      mlir::ONNXShapeOp shapeOp, int64_t &startVal, int64_t &endVal);
  // Additional data for ShapeOp.
  int64_t start, end; // Start and end properly normalized (-1 is undef).
};

//===----------------------------------------------------------------------===//
// Pooling Ops (ONNXMaxPoolSingleOutOp, ONNXAveragePoolOp, ONNXConvOp)
//===----------------------------------------------------------------------===//

// Generic pool shape helper.
template <typename OP_TYPE>
struct ONNXGenericPoolOpShapeHelper : public ONNXOpShapeHelper {
  ONNXGenericPoolOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXGenericPoolOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Actual computation of the pool shape and parameters using every different
  // switches that differs between pooling and conv ops.
  mlir::LogicalResult customComputeShape(mlir::Value X /* image */,
      mlir::Value W /* filter */,
      mlir::Optional<mlir::ArrayAttr> kernelShapeOpt, llvm::StringRef autoPad,
      mlir::Optional<mlir::ArrayAttr> padOpt,
      mlir::Optional<mlir::ArrayAttr> strideOpt,
      mlir::Optional<mlir::ArrayAttr> dilationOpt,
      bool hasFilter, // If has filter, it also has CO and optional kernel.
      bool ceilMode); // Use ceil or floor for auto_pad=NOTSET policy.

  // Values set by customComputeShape.
  llvm::SmallVector<IndexExpr, 2> kernelShape;
  llvm::SmallVector<IndexExpr, 4> pads;
  llvm::SmallVector<int64_t, 2> strides;
  llvm::SmallVector<int64_t, 2> dilations;
};

using ONNXAveragePoolOpShapeHelper =
    ONNXGenericPoolOpShapeHelper<mlir::ONNXAveragePoolOp>;
using ONNXConvOpShapeHelper = ONNXGenericPoolOpShapeHelper<mlir::ONNXConvOp>;
using ONNXConvIntegerOpShapeHelper =
    ONNXGenericPoolOpShapeHelper<mlir::ONNXConvIntegerOp>;
using ONNXQLinearConvOpShapeHelper =
    ONNXGenericPoolOpShapeHelper<mlir::ONNXQLinearConvOp>;
using ONNXMaxPoolSingleOutOpShapeHelper =
    ONNXGenericPoolOpShapeHelper<mlir::ONNXMaxPoolSingleOutOp>;

//===----------------------------------------------------------------------===//
// Slice Op
//===----------------------------------------------------------------------===//

struct ONNXSliceOpShapeHelper : public ONNXOpShapeHelper {
  ONNXSliceOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope){};
  virtual ~ONNXSliceOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Additional data for SliceOp.
  llvm::SmallVector<IndexExpr, 4> starts, ends, steps;
};

//===----------------------------------------------------------------------===//
// Gemm Op
//===----------------------------------------------------------------------===//

struct ONNXGemmOpShapeHelper : public ONNXOpShapeHelper {
  ONNXGemmOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope), aDims(), bDims(),
        cDims(), hasBias(/*dummy value*/ false), cRank(-1) {}
  virtual ~ONNXGemmOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Additional data for GemmOp: output = a * b.
  llvm::SmallVector<IndexExpr, 4> aDims; // Dim after applying transpose.
  llvm::SmallVector<IndexExpr, 4> bDims; // Dim after applying transpose.
  llvm::SmallVector<IndexExpr, 4> cDims; // Dim after padding 1 when broadcast.
  bool hasBias; // Whether there is a bias (aka C exists).
  int cRank;    // Dim of the original C (not padding dims by 1).
};

//===----------------------------------------------------------------------===//
// Matmul Ops (ONNXMatMulOp, ONNXMatMulIntegerOp, ONNXQLinearMatMulOp)
//===----------------------------------------------------------------------===//

template <typename OP_TYPE>
struct ONNXGenericMatMulOpShapeHelper : public ONNXOpShapeHelper {
  ONNXGenericMatMulOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope), aDims(), bDims(),
        aPadDims(), bPadDims() {}
  virtual ~ONNXGenericMatMulOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Additional data for MatMulOp: output = a * b.
  llvm::SmallVector<IndexExpr, 4> aDims; // Dim after applying padding.
  llvm::SmallVector<IndexExpr, 4> bDims; // Dim after applying padding.
  llvm::BitVector aPadDims;              // When true, that dim was padded.
  llvm::BitVector bPadDims;              // When true, that dim was padded.
};

// Type definition for the ops that uses ONNXGenericMatMulOpShapeHelper.
using ONNXMatMulOpShapeHelper =
    ONNXGenericMatMulOpShapeHelper<mlir::ONNXMatMulOp>;
using ONNXMatMulIntegerOpShapeHelper =
    ONNXGenericMatMulOpShapeHelper<mlir::ONNXMatMulIntegerOp>;
using ONNXQLinearMatMulOpShapeHelper =
    ONNXGenericMatMulOpShapeHelper<mlir::ONNXQLinearMatMulOp>;

//===----------------------------------------------------------------------===//
// Pad Op
//===----------------------------------------------------------------------===//

struct ONNXPadOpShapeHelper : public ONNXOpShapeHelper {
  ONNXPadOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope), pads() {}
  virtual ~ONNXPadOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Additional data for PadOp.
  llvm::SmallVector<IndexExpr, 4> pads;
};

//===----------------------------------------------------------------------===//
// OneHot Op
//===----------------------------------------------------------------------===//

struct ONNXOneHotOpShapeHelper : public ONNXOpShapeHelper {
  ONNXOneHotOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope), axis(-1), depth() {}
  virtual ~ONNXOneHotOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Additional data for OneHotOp.
  int64_t axis;    // Default value.
  IndexExpr depth; // Depth which may/may not be known at compile time.
};

//===----------------------------------------------------------------------===//
// RoiAlign Op
//===----------------------------------------------------------------------===//

struct ONNXRoiAlignOpShapeHelper : public ONNXOpShapeHelper {
  ONNXRoiAlignOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope), xDims(),
        batchIndicesDims() {}
  virtual ~ONNXRoiAlignOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Additional data for RoiAlignOp.
  llvm::SmallVector<IndexExpr, 4> xDims;            // Dim of X.
  llvm::SmallVector<IndexExpr, 1> batchIndicesDims; // Dim of batch_indices.
};

//===----------------------------------------------------------------------===//
// Arg Min/Max Op
//===----------------------------------------------------------------------===//

// Arg Min and Max operations use the same code, we just have to use their
// respective identical operand adaptor, so specialize with templated code.
template <typename OP_TYPE>
struct ONNXArgMinMaxOpShapeHelper : public ONNXOpShapeHelper {
  ONNXArgMinMaxOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXArgMinMaxOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

using ONNXArgMaxOpShapeHelper = ONNXArgMinMaxOpShapeHelper<mlir::ONNXArgMaxOp>;
using ONNXArgMinOpShapeHelper = ONNXArgMinMaxOpShapeHelper<mlir::ONNXArgMinOp>;

//===----------------------------------------------------------------------===//
// Split ops
//===----------------------------------------------------------------------===//

// Different versions of split op use common code, so specialize with
// templated code.
template <typename OP_TYPE>
struct ONNXCommonSplitOpShapeHelper : public ONNXOpShapeHelper {
  ONNXCommonSplitOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXCommonSplitOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Common code for compute shape.
  mlir::LogicalResult customComputeShape(
      mlir::ArrayRef<IndexExpr> indexExprArray);
};

using ONNXSplitOpShapeHelper = ONNXCommonSplitOpShapeHelper<mlir::ONNXSplitOp>;
using ONNXSplitV11OpShapeHelper =
    ONNXCommonSplitOpShapeHelper<mlir::ONNXSplitV11Op>;

//===----------------------------------------------------------------------===//
// Squeeze ops
//===----------------------------------------------------------------------===//

// Different versions of split op use common code, so specialize with
// templated code.
template <typename OP_TYPE>
struct ONNXCommonSqueezeOpShapeHelper : public ONNXOpShapeHelper {
  ONNXCommonSqueezeOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXCommonSqueezeOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Custom method to save axes in the op/graph.
  void saveAxes();
  // Common code for compute shape.
  mlir::LogicalResult customComputeShape(
      DimsExpr &squeezedDims, bool squeezeFromShape);
  // Data: squeezedAxes contains all of the axles to squeeze, normalized (i.e.
  // between 0 and dataRank).
  llvm::SmallVector<int64_t, 4> squeezedAxes;
};

using ONNXSqueezeOpShapeHelper =
    ONNXCommonSqueezeOpShapeHelper<mlir::ONNXSqueezeOp>;
using ONNXSqueezeV11OpShapeHelper =
    ONNXCommonSqueezeOpShapeHelper<mlir::ONNXSqueezeV11Op>;

//===----------------------------------------------------------------------===//
// Unsqueeze ops
//===----------------------------------------------------------------------===//

// Different versions of split op use common code, so specialize with
// templated code.
template <typename OP_TYPE>
struct ONNXCommonUnsqueezeOpShapeHelper : public ONNXOpShapeHelper {
  ONNXCommonUnsqueezeOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXCommonUnsqueezeOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Custom method to save axes in the op/graph.
  void saveAxes();
  // Common code for compute shape.
  mlir::LogicalResult customComputeShape(DimsExpr &unsqueezedDims);
  // Data: unsqueezedAxes contains all of the axles to unsqueeze, normalized
  // (i.e. between 0 and dataRank).
  llvm::SmallVector<int64_t, 4> unsqueezedAxes;
};

using ONNXUnsqueezeOpShapeHelper =
    ONNXCommonUnsqueezeOpShapeHelper<mlir::ONNXUnsqueezeOp>;
using ONNXUnsqueezeV11OpShapeHelper =
    ONNXCommonUnsqueezeOpShapeHelper<mlir::ONNXUnsqueezeV11Op>;

//===----------------------------------------------------------------------===//
// Reduction Ops ()
//===----------------------------------------------------------------------===//

// Generic Reduction shape helper.
template <typename OP_TYPE>
struct ONNXGenericReductionOpShapeHelper : public ONNXOpShapeHelper {
  ONNXGenericReductionOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXGenericReductionOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Actual computation of the pool shape and parameters using every different
  // switches that differs between pooling and conv ops.
  mlir::LogicalResult customComputeShape(DimsExpr &axes, int noopWithEmptyAxes);
  // Values set by customComputeShape.
  llvm::SmallVector<bool, 4> isReductionAxis;
};

using ONNXReduceL1OpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceL1Op>;
using ONNXReduceL2OpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceL2Op>;
using ONNXReduceLogSumOpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceLogSumOp>;
using ONNXReduceLogSumExpOpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceLogSumExpOp>;
using ONNXReduceMaxOpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMaxOp>;
using ONNXReduceMeanOpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMeanOp>;
using ONNXReduceMinOpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMinOp>;
using ONNXReduceProdOpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceProdOp>;
using ONNXReduceSumOpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceSumOp>;
using ONNXReduceSumV11OpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceSumV11Op>;
using ONNXReduceSumSquareOpShapeHelper =
    ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceSumSquareOp>;

//===----------------------------------------------------------------------===//
// Non specific Ops, namely ops that
//   * need customization only for themselves (no sharing of code)
//   * have no specific parameters
//===----------------------------------------------------------------------===//

/*
  These ops require a template instantiation where computeShape is defined.
  For example like this for ONNXCategoryMapperOp:

  namespace onnx_mlir {
  template struct ONNXNonSpecificOpShapeHelper<ONNXCategoryMapperOp>;
  } // namespace onnx_mlir

*/

template <typename OP_TYPE>
struct ONNXNonSpecificOpShapeHelper : public ONNXOpShapeHelper {
  ONNXNonSpecificOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXNonSpecificOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

// Ops listed in alphabetical order. Disable formatting for easier sorting.
// clang-format off
using ONNXCategoryMapperOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXCategoryMapperOp>;
using ONNXClipOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXClipOp>;
using ONNXCompressOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXCompressOp>;
using ONNXConcatOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXConcatOp>;
using ONNXConcatShapeTransposeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXConcatShapeTransposeOp>;
using ONNXConstantOfShapeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXConstantOfShapeOp>;
using ONNXDFTOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXDFTOp>;
using ONNXDepthToSpaceOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXDepthToSpaceOp>;
using ONNXDropoutOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXDropoutOp>;
using ONNXEinsumOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXEinsumOp>;
using ONNXEyeLikeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXEyeLikeOp>;
using ONNXFlattenOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXFlattenOp>;
using ONNXGatherElementsOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXGatherElementsOp>;
using ONNXGatherNDOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXGatherNDOp>;
using ONNXGatherOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXGatherOp>;
using ONNXLRNOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXLRNOp>;
using ONNXOneHotEncoderOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXOneHotEncoderOp>;
using ONNXReshapeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXReshapeOp>;
using ONNXReverseSequenceOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXReverseSequenceOp>;
using ONNXSpaceToDepthOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXSpaceToDepthOp>;
using ONNXTileOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXTileOp>;
using ONNXTopKOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXTopKOp>;
using ONNXTransposeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXTransposeOp>;
using ONNXRangeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXRangeOp>;
using ONNXResizeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXResizeOp>;
using ONNXDequantizeLinearOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXDequantizeLinearOp>;
// clang-format on

// Pattern to use:
// using ShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::>;

//===----------------------------------------------------------------------===//
// Setting a new constant or attribute value.
//===----------------------------------------------------------------------===//

/*
   Save into op's mutable operand a newly formed ONNX Constant that holds the
   integer values provided in "vals".

   Example:
     ONNXUnsqueezeOp unsqueezeOp = llvm::cast<ONNXUnsqueezeOp>(op);
     SaveOnnxConstInOp(op, unsqueezeOp.axesMutable(), unsqueezedAxes);
*/
void SaveOnnxConstInOp(mlir::Operation *op, mlir::MutableOperandRange operand,
    const llvm::SmallVectorImpl<int64_t> &vals);

/*
   Save into op's attributes a newly formed attributes that holds the integer
   values provided in "vals". The actual setting is performed by a lambda
   function with parameter OP_TYPE op and new Attribute. The user can then set
   that value in the proper attribute.

   Example:
     SaveOnnxAttrInOp<ONNXUnsqueezeV11Op>(op, unsqueezedAxes,
       [](ONNXUnsqueezeV11Op op, ArrayAttr attr) { op.axesAttr(attr); });
*/

template <typename OP_TYPE>
void SaveOnnxAttrInOp(mlir::Operation *op,
    const llvm::SmallVectorImpl<int64_t> &vals,
    mlir::function_ref<void(OP_TYPE op, mlir::ArrayAttr &attr)> setAttr) {
  // Inlined so that we don't need template instantiation.
  mlir::OpBuilder builder(op->getContext());
  mlir::ArrayAttr newAttr = builder.getI64ArrayAttr(vals);
  OP_TYPE specificOp = llvm::cast<OP_TYPE>(op);
  setAttr(specificOp, newAttr);
}
} // namespace onnx_mlir
