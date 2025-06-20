/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ONNXShapeHelper.hpp - help for shapes ---------------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SHAPE_HELPER_H
#define ONNX_MLIR_SHAPE_HELPER_H

#include <utility>

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
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

#include "src/Dialect/ONNX/ONNXOps.hpp.inc"

// ONNXOpShapeHelper is defined in the interface file below.
#include "src/Interface/ShapeHelperOpInterface.hpp"

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Support functions.
//===----------------------------------------------------------------------===//

// Check if axis is in [-rank, rank), or [-rank, rank] when includeRank is
// true.  Assert when not in range. Return positive axis.
int64_t getAxisInRange(int64_t axis, int64_t rank, bool includeRank = false);
int64_t getAxisInRange(int64_t axis, mlir::Value val, bool includeRank = false);
// Check if axis is in [-rank, rank), or [-rank, rank] when includeRank is true.
// Return false when not in range; set axis to positive value when in range.
bool isAxisInRange(int64_t &axis, int64_t rank, bool includeRank = false);
bool isAxisInRange(int64_t &axis, mlir::Value val, bool includeRank = false);

//===----------------------------------------------------------------------===//
// Support functions.
//===----------------------------------------------------------------------===//

// Update a tensor type by using the given shape, elementType and encoding.
// TODO: when all ops are migrated to the new scheme, make this function private
// to ONNXOpShapeHelper.
// Parameters:
// Val: this function will update val's type.
// shape: shape of the ranked tensor type of val.
// elementType: When nullptr, pick the elementary type from val.
// encoding: When nullptr, pick the encoding from val if defined.
void updateType(mlir::Operation *op, mlir::Value val,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType = nullptr,
    mlir::Attribute encoding = nullptr, bool refineShape = true);

// When we perform shape inference, we always assume that the type's shape in
// onnx is correct. There are rare instance where we transform an existing op
// (see RNN's handling of layout in RNNOpRewriteLayoutPattern) and then seek to
// perform shape inference on it. As the operation has changed, then we must
// first "erase" its constant shape's for the output type as they are not
// correct anymore. It might be wiser to not reuse an existing op, but since we
// currently have this pattern, this function must be called prior to infer
// shapes of existing but modified operations.
void resetTypesShapeToQuestionmarks(mlir::Operation *op);

//===----------------------------------------------------------------------===//
// Unimplemented Ops (to be used sparingly, currently for Loop and Scan).
// Other uses should be converted to shape inferences.
//===----------------------------------------------------------------------===//

struct ONNXUnimplementedOpShapeHelper : public ONNXOpShapeHelper {
  ONNXUnimplementedOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXUnimplementedOpShapeHelper() {}

  bool isImplemented() override { return false; }
  mlir::LogicalResult computeShape() final { return mlir::success(); }
};

// Classes for unsupported ops, including shape inference and shape helpers.
#define UNSUPPORTED_OPS(OP_TYPE)                                               \
  using OP_TYPE##ShapeHelper = ONNXUnimplementedOpShapeHelper;
#include "src/Dialect/ONNX/ONNXUnsupportedOps.hpp"
#undef UNSUPPORTED_OPS

// Classes with implemented shape inference but not shape helper.

// clang-format off
using ONNXCallOpShapeHelper = ONNXUnimplementedOpShapeHelper;
using ONNXIfOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: recursive, Opt, Seq
using ONNXLoopOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: recursive, Opt, Seq
using ONNXOptionalGetElementOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: Opt, Seq
using ONNXOptionalHasElementOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: Opt, Seq
using ONNXOptionalOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: Opt, Seq
using ONNXScanOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: recursive
using ONNXSequenceAtOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: Seq
using ONNXSequenceConstructOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: Seq
using ONNXSequenceEmptyOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: Seq
using ONNXSequenceEraseOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: Seq
using ONNXSequenceInsertOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: Seq
using ONNXSequenceLengthOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: Seq
using ONNXSplitToSequenceOpShapeHelper = ONNXUnimplementedOpShapeHelper; // Reason: Seq
// clang-format on

//===----------------------------------------------------------------------===//
// Broadcast Ops
//===----------------------------------------------------------------------===//

// Compute a broadcasted shape from the shapes of given operands. Operands must
// be ranked in advance.
struct ONNXBroadcastOpShapeHelper : public ONNXOpShapeHelper {
  ONNXBroadcastOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr,
      bool hasUniBroadcasting = false)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope), inputsDims(),
        outputRank(0), hasUniBroadcasting(hasUniBroadcasting) {}
  virtual ~ONNXBroadcastOpShapeHelper() {}

  // Custom shape compute which takes additional parameters.
  mlir::LogicalResult customComputeShape(
      mlir::ValueRange initialOperands, DimsExpr *additionalOperand);

  // Default shape compute (every operands of the operation and no additional
  // parameters).
  mlir::LogicalResult computeShape() override {
    return customComputeShape(operands, nullptr);
  }

  // Compute access indices to load/store value from/to a given 'operand'.
  // Used in a loop to access the operand.
  // Parameters:
  //   - operand: operand to access.
  //   - i: index of the operand in Index Expr Dims 'this->inputsDims'.
  //   - loopAccessExprs: IndexExprs for the loop's IVs.
  //   - operandAccessExprs: access indices to access the operand.
  //     This is the output of this function. Use it in subsequent load/stores.
  //   - flattenedInnerDims: whether the innermost dimension corresponds to a
  //   collapsed/flattened loop index or not.
  //   - ruledOutBroadcast: determined using shape analysis that there is no
  //     broadcasting here.
  virtual mlir::LogicalResult getAccessExprs(mlir::Value operand, int64_t i,
      const llvm::SmallVectorImpl<IndexExpr> &loopAccessExprs,
      llvm::SmallVectorImpl<IndexExpr> &operandAccessExprs,
      bool flattenedInnerDims = false, bool ruledOutBroadcast = false);

  // Determine if broadcast can be ruled out at compile time. Use DimAnalysis
  // when available. Broadcasting is defined if one value of one input is used
  // two or more times with a value of another input (when only looking at the
  // tensors, not the actual algorithms).
  //
  // Examples with broadcasts:
  // * 2x5xf32 and 1x5xf32 has broadcast as the second input's value are used
  //   for each of the two instances of 5xf32 in the first input.
  // * Same holds for 2x5xf32 and 5xf32 as shorter ranked
  //   inputs are extended by prepending 1x.
  //
  // Example without broadcast:
  // * 3x5xf32 and 3x5xf32 have no broadcast.
  // * 1x5xf32 and 5xf32 have also no broadcast as prepending 1x results as
  //   comparing 1x5xf32 with 1x5xf32.
  virtual bool hasNoBroadcast(DimAnalysis *dimAnalysis = nullptr);

  // Determine if the broadcast operation has operands with different ranks.
  // Use DimAnalysis when available. The broadcasting will then add dimensions
  // with size 1.
  //
  // Examples with rank broadcast:
  // * 2x5xf32 and 5xf32 has rank broadcast for the second type. It will be
  // interpreted as 1x5xf32
  //
  // Examples without rank broadcast:
  // * 2x5xf32 and 1x5xf32 does not have rank broadcasting because the ranks are
  // already equal
  virtual bool hasRankBroadcast();

  // Determine of the broadcast operation has manageable broadcast (MB), and if
  // so, at which level/rank. We first attempt to see if the innermost dimension
  // has MB, and if it does, we then attempt to test at the next innermost
  // level... until we fail or we run out of dimensions.
  //
  // Below: ?1 and ?2 indicate 2 dynamic dimensions, which may/may not be
  // guaranteed to be equal depending on what dynamic analysis says.
  //
  // Manageable broadcast (MB) is either that:
  //   1) we have no broadcast up to that level, or
  //   2) we have scalars up to that level being broadcasted.
  //
  // The function return true if there is some MB, and then
  // * collapsedInnermostLoops: indicates how many inner loops are involved in
  //   the MB. They are named "collapsed" as in the SIMD code execution, we may
  //   collapse these dimensions in a single long iteration. For example,
  //   `0x?x4x5` and `0x?x4x5` have a collapsedInnermostLoops==2 (if the two `?`
  //   cannot be shown as equals). This means that we may implement operations
  //   on these inputs as `?x20` and `?x20` respectively.
  // * collapsedLiteralSize: cumulative static size of the collapsed inner
  //   loops.
  // * collapsedDynamicSize: cumulative dynamic size of the collapsed inner
  //   loops.
  //
  // Below are examples of Manageable Broadcast (MB) at a given
  // collapsedInnermostLoops (CIL) level.
  //
  // What is a scalar: `4x2x1x1 is scalar at CIL==1 and 2, but not 3 and 4.
  //
  //  - (?1, 1, 4, 1) and (10, 1, 4, 1) have MB at CIL 1, 2, 3, not 4,
  //    unless dynamic analysis can show ?1 to be equal to 10 (unlikely).
  //  - (?1, 1, ?2, 1) and (10, 1, ?3, 1) have MB at CIL 1 and at 2
  //    if dynamic analysis can show ?2 and ?3 to be the same.
  //  - (1, 4, 1) and (2, 4, 1) have MB at CIL 1 and 2, but not 3 as
  //    there is broadcasting (1 vs 2) at CIL 3... but the first
  //    operand is not a scalar at CIL 3.
  //  - (1, 1, 1) and (2, 4, 1) have MB at CIL 1, 2, and 3 as there is
  //    broadcast at inner dim 2 and 3, and the first operand is a scalar at
  //    CIL 1, 2, and 3.
  // - (1,3) and (1, 1) have MB at CIL 1; technically, CIL 2 is also a MB but
  //    there is nothing to be gained by collapsing dimensions where all
  //    inputs have dimensions of 1. We thus do not include them in our CILs.
  //    Revision: it is actually good to detects 1s everywhere as we can
  //    collapse the loop and have less overhead.

  virtual bool hasManageableBroadcastForInnerDims(
      int64_t &collapsedInnermostLoops, int64_t &collapsedLiteralSize,
      IndexExpr &collapsedDynamicSize, DimAnalysis *dimAnalysis);

  // A vector of input shapes where dimensions are padded with 1 if necessary,
  // so that all inputs have the same rank. Instantiated during ComputeShape.
  llvm::SmallVector<DimsExpr, 4> inputsDims;
  // Rank of the output shape.
  uint64_t outputRank;

protected:
  // When unidirectional broadcasting is true, the other operands are always
  // unidirectional broadcastable to the first operand.
  bool hasUniBroadcasting;
};

// clang-format off
using ONNXAddOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXAndOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXBitwiseAndOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXBitwiseOrOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXBitwiseXorOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXBitShiftOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXDivOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXEqualOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXGreaterOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXGreaterOrEqualOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXLessOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXLessOrEqualOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXMaxOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXMeanOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXMinOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXModOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXMulOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXOrOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXPowOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXSubOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXSumOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXWhereOpShapeHelper = ONNXBroadcastOpShapeHelper;
using ONNXXorOpShapeHelper = ONNXBroadcastOpShapeHelper;
// clang-format on

// Helper for ExpandOp
struct ONNXExpandOpShapeHelper : public ONNXBroadcastOpShapeHelper {
  ONNXExpandOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXBroadcastOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXExpandOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

// Helper for ONNXPReluOp
struct ONNXPReluOpShapeHelper : public ONNXBroadcastOpShapeHelper {
  ONNXPReluOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXBroadcastOpShapeHelper(op, operands, ieBuilder, scope,
            /*hasUniBroadcasting*/ true) {}
  virtual ~ONNXPReluOpShapeHelper() {}
  mlir::LogicalResult computeShape() final {
    return ONNXBroadcastOpShapeHelper::computeShape();
  }
};

// Template for Layer Normalization (LN) ops
template <typename OP_TYPE>
struct ONNXLNOpShapeHelper : public ONNXBroadcastOpShapeHelper {
  ONNXLNOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXBroadcastOpShapeHelper(op, operands, ieBuilder, scope,
            /*hasUniBroadcasting*/ true) {}
  virtual ~ONNXLNOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

// clang-format off
using ONNXLayerNormalizationOpShapeHelper = ONNXLNOpShapeHelper<mlir::ONNXLayerNormalizationOp>;
using ONNXRMSLayerNormalizationOpShapeHelper = ONNXLNOpShapeHelper<mlir::ONNXRMSLayerNormalizationOp>;
// clang-format on

//===----------------------------------------------------------------------===//
// Unary Ops
//===----------------------------------------------------------------------===//

/// Compute an output shape for a unary element-wise operation. The output and
/// input of an unary element-wise operation have the same shape.
struct ONNXUnaryOpShapeHelper : public ONNXBroadcastOpShapeHelper {
  ONNXUnaryOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXBroadcastOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXUnaryOpShapeHelper() {}

  mlir::LogicalResult computeShape() override;

  // Inherited methods that return trivial results
  mlir::LogicalResult getAccessExprs(mlir::Value operand, int64_t i,
      const llvm::SmallVectorImpl<IndexExpr> &loopAccessExprs,
      llvm::SmallVectorImpl<IndexExpr> &operandAccessExprs,
      bool flattenedInnerDims = false, bool ruledOutBroadcast = false) override;
  bool hasNoBroadcast(DimAnalysis *dimAnalysis = nullptr) override;
  bool hasManageableBroadcastForInnerDims(int64_t &collapsedInnermostLoops,
      int64_t &collapsedLiteralSize, IndexExpr &collapsedDynamicSize,
      DimAnalysis *dimAnalysis) override;
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

// clang-format off
using ONNXAbsOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXAcosOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXAcoshOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXAsinOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXAsinhOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXAtanOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXAtanhOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXBernoulliOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXBinarizerOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXBitwiseNotOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXCastOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXCeilOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXCeluOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXClipOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXCosOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXCoshOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXCumSumOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXEluOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXErfOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXExpOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXFloorOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXGeluOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXHardSigmoidOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXHardSwishOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXHardmaxOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXInstanceNormalizationOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXIsInfOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXIsNaNOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXLayoutTransformOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXLeakyReluOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXLogOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXLogSoftmaxOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXLpNormalizationOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXMeanVarianceNormalizationOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXMishOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXNegOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXNotOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXRandomNormalLikeOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXReciprocalOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXReluOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXRoundOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXScalerOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXScatterElementsOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXScatterNDOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXScatterOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXSeluOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXShrinkOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXSigmoidOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXSignOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXSinOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXSinhOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXSoftmaxOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXSoftmaxV11OpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXSoftplusOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXSoftsignOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXSqrtOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXTanOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXTanhOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXThresholdedReluOpShapeHelper = ONNXUnaryOpShapeHelper;
using ONNXTriluOpShapeHelper = ONNXUnaryOpShapeHelper;
// clang-format on

//===----------------------------------------------------------------------===//
// Shape op
//===----------------------------------------------------------------------===//

struct ONNXShapeOpShapeHelper : public ONNXOpShapeHelper {
  ONNXShapeOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
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
  ONNXGenericPoolOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXGenericPoolOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Actual computation of the pool shape and parameters using every different
  // switches that differs between pooling and conv ops.
  mlir::LogicalResult customComputeShape(mlir::Value X /* image */,
      mlir::Value W /* filter */, std::optional<mlir::ArrayAttr> kernelShapeOpt,
      llvm::StringRef autoPad, std::optional<mlir::ArrayAttr> padOpt,
      std::optional<mlir::ArrayAttr> strideOpt,
      std::optional<mlir::ArrayAttr> dilationOpt,
      bool hasFilter, // If has filter, it also has CO and optional kernel.
      bool ceilMode); // Use ceil or floor for auto_pad=NOTSET policy.

  // Values set by customComputeShape.
  llvm::SmallVector<IndexExpr, 2> kernelShape;
  llvm::SmallVector<IndexExpr, 4> pads;
  llvm::SmallVector<int64_t, 2> strides;
  llvm::SmallVector<int64_t, 2> dilations;
};

// clang-format off
using ONNXAveragePoolOpShapeHelper = ONNXGenericPoolOpShapeHelper<mlir::ONNXAveragePoolOp>;
using ONNXConvOpShapeHelper = ONNXGenericPoolOpShapeHelper<mlir::ONNXConvOp>;
using ONNXConvIntegerOpShapeHelper = ONNXGenericPoolOpShapeHelper<mlir::ONNXConvIntegerOp>;
using ONNXQLinearConvOpShapeHelper = ONNXGenericPoolOpShapeHelper<mlir::ONNXQLinearConvOp>;
using ONNXMaxPoolSingleOutOpShapeHelper = ONNXGenericPoolOpShapeHelper<mlir::ONNXMaxPoolSingleOutOp>;
// clang-format on

//===----------------------------------------------------------------------===//
// ConvTranspose Op
//===----------------------------------------------------------------------===//

struct ONNXConvTransposeOpShapeHelper : public ONNXOpShapeHelper {
  ONNXConvTransposeOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope), kernelShape(),
        pads(), strides(), dilations(), outputPadding() {}
  virtual ~ONNXConvTransposeOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Values set by computeShape.
  llvm::SmallVector<IndexExpr, 2> kernelShape;
  llvm::SmallVector<IndexExpr, 4> pads;
  llvm::SmallVector<int64_t, 2> strides;
  llvm::SmallVector<int64_t, 2> dilations;
  llvm::SmallVector<int64_t, 2> outputPadding;
};

//===----------------------------------------------------------------------===//
// Global pooling ops
//===----------------------------------------------------------------------===//

template <typename OP_TYPE>
struct ONNXGenericGlobalPoolOpShapeHelper : public ONNXOpShapeHelper {
  ONNXGenericGlobalPoolOpShapeHelper(mlir::Operation *op,
      mlir::ValueRange operands, IndexExprBuilder *ieBuilder = nullptr,
      IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXGenericGlobalPoolOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

// clang-format off
using ONNXGlobalAveragePoolOpShapeHelper = ONNXGenericGlobalPoolOpShapeHelper<mlir::ONNXGlobalAveragePoolOp>;
using ONNXGlobalLpPoolOpShapeHelper = ONNXGenericGlobalPoolOpShapeHelper<mlir::ONNXGlobalLpPoolOp>;
using ONNXGlobalMaxPoolOpShapeHelper = ONNXGenericGlobalPoolOpShapeHelper<mlir::ONNXGlobalMaxPoolOp>;
// clang-format on

//===----------------------------------------------------------------------===//
// Slice Op
//===----------------------------------------------------------------------===//

struct ONNXSliceOpShapeHelper : public ONNXOpShapeHelper {
  ONNXSliceOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
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
  ONNXGemmOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
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
  ONNXGenericMatMulOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
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

// clang-format off
using ONNXMatMulOpShapeHelper = ONNXGenericMatMulOpShapeHelper<mlir::ONNXMatMulOp>;
using ONNXMatMulIntegerOpShapeHelper = ONNXGenericMatMulOpShapeHelper<mlir::ONNXMatMulIntegerOp>;
using ONNXQLinearMatMulOpShapeHelper = ONNXGenericMatMulOpShapeHelper<mlir::ONNXQLinearMatMulOp>;
// clang-format on

//===----------------------------------------------------------------------===//
// Pad Op
//===----------------------------------------------------------------------===//

struct ONNXPadOpShapeHelper : public ONNXOpShapeHelper {
  ONNXPadOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
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
  ONNXOneHotOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
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
  ONNXRoiAlignOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
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
  ONNXArgMinMaxOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXArgMinMaxOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

// clang-format off
using ONNXArgMaxOpShapeHelper = ONNXArgMinMaxOpShapeHelper<mlir::ONNXArgMaxOp>;
using ONNXArgMinOpShapeHelper = ONNXArgMinMaxOpShapeHelper<mlir::ONNXArgMinOp>;
// clang-format on

//===----------------------------------------------------------------------===//
// Split ops
//===----------------------------------------------------------------------===//

// Different versions of split op use common code, so specialize with
// templated code.
template <typename OP_TYPE>
struct ONNXCommonSplitOpShapeHelper : public ONNXOpShapeHelper {
  ONNXCommonSplitOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXCommonSplitOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Common code for compute shape.
  mlir::LogicalResult customComputeShape(
      mlir::ArrayRef<IndexExpr> indexExprArray);
};

// clang-format off
using ONNXSplitOpShapeHelper = ONNXCommonSplitOpShapeHelper<mlir::ONNXSplitOp>;
using ONNXSplitV11OpShapeHelper = ONNXCommonSplitOpShapeHelper<mlir::ONNXSplitV11Op>;
using ONNXSplitV13OpShapeHelper = ONNXCommonSplitOpShapeHelper<mlir::ONNXSplitV13Op>;
// clang-format on

//===----------------------------------------------------------------------===//
// Squeeze ops
//===----------------------------------------------------------------------===//

// Different versions of split op use common code, so specialize with
// templated code.
template <typename OP_TYPE>
struct ONNXCommonSqueezeOpShapeHelper : public ONNXOpShapeHelper {
  ONNXCommonSqueezeOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
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

// clang-format off
using ONNXSqueezeOpShapeHelper = ONNXCommonSqueezeOpShapeHelper<mlir::ONNXSqueezeOp>;
using ONNXSqueezeV11OpShapeHelper = ONNXCommonSqueezeOpShapeHelper<mlir::ONNXSqueezeV11Op>;
// clang-format on

//===----------------------------------------------------------------------===//
// Unique ops
//===----------------------------------------------------------------------===//

// Different versions of split op use common code, so specialize with
// templated code.
struct ONNXUniqueOpShapeHelper : public ONNXOpShapeHelper {
  ONNXUniqueOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXUniqueOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Additional data for UniqueOp:
};

//===----------------------------------------------------------------------===//
// Unsqueeze ops
//===----------------------------------------------------------------------===//

// Different versions of split op use common code, so specialize with
// templated code.
template <typename OP_TYPE>
struct ONNXCommonUnsqueezeOpShapeHelper : public ONNXOpShapeHelper {
  ONNXCommonUnsqueezeOpShapeHelper(mlir::Operation *op,
      mlir::ValueRange operands, IndexExprBuilder *ieBuilder = nullptr,
      IndexExprScope *scope = nullptr)
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

// clang-format off
using ONNXUnsqueezeOpShapeHelper = ONNXCommonUnsqueezeOpShapeHelper<mlir::ONNXUnsqueezeOp>;
using ONNXUnsqueezeV11OpShapeHelper = ONNXCommonUnsqueezeOpShapeHelper<mlir::ONNXUnsqueezeV11Op>;
// clang-format on

//===----------------------------------------------------------------------===//
// DFT Ops
//===----------------------------------------------------------------------===//

// Generic DFT shape helper.
template <typename OP_TYPE>
struct ONNXGenericDFTOpShapeHelper : public ONNXOpShapeHelper {
  ONNXGenericDFTOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXGenericDFTOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  mlir::LogicalResult customComputeShape(IndexExpr &axis);
};

// clang-format off
using ONNXDFTOpShapeHelper = ONNXGenericDFTOpShapeHelper<mlir::ONNXDFTOp>;

//===----------------------------------------------------------------------===//
// Reduction Ops
//===----------------------------------------------------------------------===//

// Generic Reduction shape helper.
template <typename OP_TYPE>
struct ONNXGenericReductionOpShapeHelper : public ONNXOpShapeHelper {
  ONNXGenericReductionOpShapeHelper(mlir::Operation *op,
      mlir::ValueRange operands, IndexExprBuilder *ieBuilder = nullptr,
      IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXGenericReductionOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Actual computation of the pool shape and parameters using every different
  // switches that differs between pooling and conv ops.
  mlir::LogicalResult customComputeShape(DimsExpr &axes, int noopWithEmptyAxes);
  // Values set by customComputeShape.
  llvm::SmallVector<bool, 4> isReductionAxis;
};

// clang-format off
using ONNXReduceL1OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceL1Op>;
using ONNXReduceL1V13OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceL1V13Op>;
using ONNXReduceL2OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceL2Op>;
using ONNXReduceL2V13OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceL2V13Op>;
using ONNXReduceLogSumOpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceLogSumOp>;
using ONNXReduceLogSumV13OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceLogSumV13Op>;
using ONNXReduceLogSumExpOpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceLogSumExpOp>;
using ONNXReduceLogSumExpV13OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceLogSumExpV13Op>;
using ONNXReduceMaxOpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMaxOp>;
using ONNXReduceMaxV13OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMaxV13Op>;
using ONNXReduceMaxV18OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMaxV18Op>;
using ONNXReduceMeanOpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMeanOp>;
using ONNXReduceMeanV13OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMeanV13Op>;
using ONNXReduceMinOpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMinOp>;
using ONNXReduceMinV13OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMinV13Op>;
using ONNXReduceMinV18OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceMinV18Op>;
using ONNXReduceProdOpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceProdOp>;
using ONNXReduceProdV13OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceProdV13Op>;
using ONNXReduceSumOpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceSumOp>;
using ONNXReduceSumV11OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceSumV11Op>;
using ONNXReduceSumSquareOpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceSumSquareOp>;
using ONNXReduceSumSquareV13OpShapeHelper = ONNXGenericReductionOpShapeHelper<mlir::ONNXReduceSumSquareV13Op>;
// clang-format on

//===----------------------------------------------------------------------===//
// RNN Ops (ONNXRNNOp, ONNXLSTMOp, ONNXRNNOp)
//===----------------------------------------------------------------------===//

// Generic Reduction shape helper.
template <typename OP_TYPE>
struct ONNXGenericRNNShapeHelper : public ONNXOpShapeHelper {
  ONNXGenericRNNShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXGenericRNNShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Actual computation of the RNN shape and parameters using every different
  // switches that differs between pooling and conv ops.
  mlir::LogicalResult customComputeShape(int gates);
  // Values set by customComputeShape.
  llvm::SmallVector<bool, 4> isReductionAxis;
};

// clang-format off
using ONNXGRUOpShapeHelper = ONNXGenericRNNShapeHelper<mlir::ONNXGRUOp>;
using ONNXLSTMOpShapeHelper = ONNXGenericRNNShapeHelper<mlir::ONNXLSTMOp>;
using ONNXRNNOpShapeHelper = ONNXGenericRNNShapeHelper<mlir::ONNXRNNOp>;
// clang-format on

//===----------------------------------------------------------------------===//
// Resize Op
//===----------------------------------------------------------------------===//

struct ONNXResizeOpShapeHelper : public ONNXOpShapeHelper {
  ONNXResizeOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXResizeOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Values set by computeShape: scales is a float index expression. It is
  // directly the `scale` argument when scale is provided by the op. When
  // `size` is provided, then scale is float(`size`)/float(dim).
  llvm::SmallVector<IndexExpr, 4> scales;
};

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
  ONNXNonSpecificOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXNonSpecificOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

// Ops listed in alphabetical order. Disable formatting for easier sorting.
// clang-format off
using ONNXBatchNormalizationInferenceModeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXBatchNormalizationInferenceModeOp>;
using ONNXCategoryMapperOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXCategoryMapperOp>;
using ONNXCompressOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXCompressOp>;
using ONNXConcatOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXConcatOp>;
using ONNXConcatShapeTransposeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXConcatShapeTransposeOp>;
using ONNXConstantOfShapeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXConstantOfShapeOp>;
using ONNXConstantOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXConstantOp>;
using ONNXDepthToSpaceOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXDepthToSpaceOp>;
using ONNXDequantizeLinearOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXDequantizeLinearOp>;
using ONNXDimOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXDimOp>;
using ONNXDropoutOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXDropoutOp>;
using ONNXDynamicQuantizeLinearOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXDynamicQuantizeLinearOp>;
using ONNXEinsumOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXEinsumOp>;
using ONNXGridSampleOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXGridSampleOp>;
using ONNXEyeLikeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXEyeLikeOp>;
using ONNXFlattenOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXFlattenOp>;
using ONNXGatherElementsOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXGatherElementsOp>;
using ONNXGatherNDOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXGatherNDOp>;
using ONNXGatherOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXGatherOp>;
using ONNXIdentityOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXIdentityOp>;
using ONNXLRNOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXLRNOp>;
using ONNXMaxRoiPoolOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXMaxRoiPoolOp>;
using ONNXNonMaxSuppressionOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXNonMaxSuppressionOp>;
using ONNXNonZeroOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXNonZeroOp>;
using ONNXOneHotEncoderOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXOneHotEncoderOp>;
using ONNXQuantizeLinearOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXQuantizeLinearOp>;
using ONNXRandomNormalOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXRandomNormalOp>;
using ONNXRandomUniformOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXRandomUniformOp>;
using ONNXRangeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXRangeOp>;
using ONNXReshapeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXReshapeOp>;
using ONNXReverseSequenceOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXReverseSequenceOp>;
using ONNXShapeTransformOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXShapeTransformOp>;
using ONNXSizeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXSizeOp>;
using ONNXSpaceToDepthOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXSpaceToDepthOp>;
using ONNXTileOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXTileOp>;
using ONNXTopKOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXTopKOp>;
using ONNXTransposeOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXTransposeOp>;
using ONNXUpsampleOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXUpsampleOp>;
// clang-format on

//===----------------------------------------------------------------------===//
// WindowsOp Shape Helper.
//===----------------------------------------------------------------------===//

template <typename OP_TYPE>
struct ONNXWindowsOpShapeHelper : public ONNXOpShapeHelper {
  ONNXWindowsOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~ONNXWindowsOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

// clang-format off
using ONNXHammingWindowOpShapeHelper = ONNXWindowsOpShapeHelper<mlir::ONNXHammingWindowOp>;
using ONNXBlackmanWindowOpShapeHelper = ONNXWindowsOpShapeHelper<mlir::ONNXBlackmanWindowOp>;
// clang-format on
//===----------------------------------------------------------------------===//
// CustomOp Shape Helper.
//===----------------------------------------------------------------------===//

struct ONNXCustomOpShapeHelper : public ONNXUnaryOpShapeHelper {
  ONNXCustomOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr,
      bool hasUniBroadcasting = false);

  bool isImplemented() override;
  // Default shape compute (every operands of the operation and no additional
  // parameters).
  mlir::LogicalResult computeShape() override;

protected:
  // Shape inference pattern
  int pattern;
};

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
       [](ONNXUnsqueezeV11Op op, ArrayAttr attr) { op.setAxesAttr(attr); });
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
#endif
