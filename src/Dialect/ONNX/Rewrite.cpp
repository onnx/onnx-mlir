/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXRewrite.cpp - ONNX High Level Optimizer --------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters for operations in the ONNX dialect
// that can be rewritten by using other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// If 'A' is NoneType, return -B. Otherwise return A-B.
Value subtractOrNeg(PatternRewriter &rewriter, Location loc, Value A, Value B) {
  if (A.getType().isa<NoneType>())
    return rewriter.create<ONNXNegOp>(loc, B);
  return rewriter.create<ONNXSubOp>(loc, A, B);
}

// Create an ArrayAttr of IntergerAttr(s) of values in [1, N].
ArrayAttr createArrayAttrOfOneToN(PatternRewriter &rewriter, int N) {
  SmallVector<int64_t, 4> vals;
  for (int i = 1; i <= N; ++i)
    vals.emplace_back(i);
  return rewriter.getI64ArrayAttr(vals);
}

// Create an ArrayAttr of IntergerAttr(s) of values in [N, M].
ArrayAttr createArrayAttrOfNToM(PatternRewriter &rewriter, int N, int M) {
  SmallVector<int64_t, 4> vals;
  for (int i = N; i <= M; ++i)
    vals.emplace_back(i);
  return rewriter.getI64ArrayAttr(vals);
}

// Get return type for a MatMulOp whose A's rank is N (>2) and B's rank is 2.
Type getReturnTypeForMatMulOpND2D(Value A, Value B) {
  ArrayRef<int64_t> aShape = A.getType().cast<RankedTensorType>().getShape();
  ArrayRef<int64_t> bShape = B.getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t> resShape(aShape.begin(), aShape.end() - 1);
  resShape.emplace_back(bShape[bShape.size() - 1]);
  return RankedTensorType::get(
      resShape, A.getType().cast<ShapedType>().getElementType());
}

// Get the index of the axis value in the given permutation array.
IntegerAttr getIndexOfAxisInPerm(
    PatternRewriter &rewriter, ArrayAttr permAttr, IntegerAttr axis) {
  IntegerAttr result;
  for (uint64_t i = 0; i < permAttr.getValue().size(); ++i) {
    IntegerAttr attr = permAttr.getValue()[i].cast<IntegerAttr>();
    assert(attr && "Element in ArrayAttr is not IntegerAttr");
    if (attr.getValue().getSExtValue() == axis.getValue().getSExtValue())
      return rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), i);
  }
  return result;
}

// Transpose a variadic input using a permutation array.
SmallVector<Value, 4> transposeVariadicInput(PatternRewriter &rewriter,
    Location loc, ValueRange inputs, ArrayAttr permAttr) {
  SmallVector<Value, 4> transposedInputs;
  for (Value inp : inputs) {
    ShapedType inpType = inp.getType().cast<ShapedType>();
    assert(inpType && "Type is not ShapedType");
    ONNXTransposeOp transposeOp = rewriter.create<ONNXTransposeOp>(
        loc, UnrankedTensorType::get(inpType.getElementType()), inp, permAttr);
    (void)transposeOp.inferShapes([](Region &region) {});
    transposedInputs.emplace_back(transposeOp.getResult());
  }
  return transposedInputs;
}

// Check if all values are produced by ONNXTransposeOp.
bool areProducedByTransposeOp(ValueRange values) {
  return llvm::all_of(values, [](Value v) {
    if (v.isa<BlockArgument>())
      return false;
    return isa<ONNXTransposeOp>(v.getDefiningOp());
  });
}

} // namespace onnx_mlir

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Dialect/ONNX/ONNXRewrite.inc"

// In some ONNX models, the maximum trip count for LoopOp is set to a big value,
// e.g. LONG_MAX and termination depends on the break condition inside the loop.
// In the current lowering of LoopOp, the maximum trip count is used to allocate
// a buffer for all intermediate loop results. Since the actual number of loop
// iterations may be much smaller than the maximum trip count, it is redundant
// and error-prone to allocate a large buffer. For example, we may get segfault
// if the maximum trip count is out of range.
//
// This pattern tries to derive a new maximum trip count for LoopOp by analyzing
// the break condition. It only handles a special case where the loop is like a
// for-loop with step, e.g. `for (i = LB, i < UB, i = i + Step)`.
//
// For example, the following loop which mimics LoopOp:
// ```
// max_trip_count=9223372036854775807
// LB = -100
// UB = 100
// Step = 1
//
// i = 0
// k = LB
// keepgoing = true
// while (i < max_trip_count && keepgoing == true) {
//    k = k + STEP
//    keepgoing = (k < UB)
// }
// ```
//
// will be rewritten into:
//
// ```
// max_trip_count=200
// LB = -100
// UB = 100
//
// i = 0
// k = LB
// keepgoing = true
// while (i < max_trip_count && keepgoing == true) {
//    k = k + STEP
//    keepgoing = (k < UB)
// }
// ```
// where `max_trip_count` is replaced by an actual value derived from the loop.
//
class LoopOpRewriteMaxTripCountPattern : public OpRewritePattern<ONNXLoopOp> {
public:
  using OpRewritePattern<ONNXLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXLoopOp loopOp, PatternRewriter &rewriter) const override {
    Location loc = loopOp.getLoc();
    Operation *genericLoopOp = loopOp.getOperation();
    Value maxTripCountValue = genericLoopOp->getOperands()[0];

    // A helper function to get a constant from a value.
    auto getOneConstantValue = [&](Value v) -> int64_t {
      if (v.isa<BlockArgument>())
        return -1;
      Operation *definingOp = v.getDefiningOp();
      if (!isa<ONNXConstantOp>(definingOp))
        return -1;
      if (!cast<ONNXConstantOp>(definingOp)
               .valueAttr()
               .isa<DenseElementsAttr>())
        return -1;
      DenseElementsAttr valueAttr = cast<ONNXConstantOp>(definingOp)
                                        .valueAttr()
                                        .cast<DenseElementsAttr>();
      return (*valueAttr.getValues<APInt>().begin()).getSExtValue();
    };

    // A helper function to get a constant from a value. The value must be
    // defined by ConstantOp inside the loop or fed by ConstantOp outside the
    // loop.
    auto constantOrFedByConstant = [&](Value v, Operation *loopOp,
                                       Operation *returnOp) -> int64_t {
      if (v.isa<BlockArgument>()) {
        // Value is an block argument and its value is outside the loop.
        // 1. is unchanged by iterations. By the definition of LoopOp, block
        // arguments are shifted by 1 to the left in ReturnOp.
        if (v !=
            returnOp->getOperands()[v.cast<BlockArgument>().getArgNumber() - 1])
          return -1;
        // 2. is fed with a constant.
        Value vInput =
            loopOp->getOperands()[v.cast<BlockArgument>().getArgNumber()];
        return getOneConstantValue(vInput);
      }
      return getOneConstantValue(v);
    };

    // Only do this rewriting if the maximum trip count is a constant.
    int64_t maxTripCount = getOneConstantValue(maxTripCountValue);
    if (maxTripCount < 0)
      return failure();

    // Get the loop region.
    Region &loopBody = loopOp.body();
    // Make sure the region has only one block.
    if (!loopBody.hasOneBlock())
      return failure();

    // Get the body block.
    Block &bodyBlock = loopBody.front();

    // Make sure that the loop body has only one ReturnOp.
    ONNXReturnOp returnOp;
    if (bodyBlock
            .walk([&](ONNXReturnOp op) -> WalkResult {
              if (returnOp)
                return WalkResult::interrupt();
              returnOp = op;
              return WalkResult::advance();
            })
            .wasInterrupted() ||
        !returnOp)
      return failure();
    Operation *genericReturnOp = returnOp.getOperation();

    // Analyze the break condition of the loop body to see if we can derive a
    // new maximum trip count or not.

    // The break condition is the first argument of ReturnOp.
    Value breakCond = genericReturnOp->getOperands()[0];
    if (breakCond.isa<BlockArgument>())
      return failure();
    Operation *breakCondOp = breakCond.getDefiningOp();

    // Only support LessOp as the op that defines the break condition at this
    // moment.
    if (!isa<ONNXLessOp>(breakCondOp))
      return failure();

    // Compute a trip count from the break condition, given that the upper bound
    // is fixed and the lower bound is increased by a constant step at each
    // iteration. So, the trip count will be `(upper_bound - lower_bound)/step`.
    Value lbValue = breakCondOp->getOperands()[0];
    Value ubValue = breakCondOp->getOperands()[1];

    // Check the lower bound of the break condition.
    // The lowerbound is updated by ONNXAddOp, given that ONNXAddOp is
    // canonicalized to Add(lhs, rhs=constant).
    if (lbValue.isa<BlockArgument>() ||
        !isa<ONNXAddOp>(lbValue.getDefiningOp()))
      return failure();
    Value lhs = cast<ONNXAddOp>(lbValue.getDefiningOp())->getOperands()[0];
    Value rhs = cast<ONNXAddOp>(lbValue.getDefiningOp())->getOperands()[1];
    // 1. lhs is updated by AddOp at each iteration.
    if (!lhs.isa<BlockArgument>())
      return failure();
    if (lbValue !=
        genericReturnOp
            ->getOperands()[lhs.cast<BlockArgument>().getArgNumber() - 1])
      return failure();
    // 2. lhs is increased by a constant step.
    int64_t step = constantOrFedByConstant(rhs, genericLoopOp, genericReturnOp);
    if (step < 0)
      return failure();
    // 3. lhs is initialized by a constant.
    Value lhsInput =
        genericLoopOp->getOperands()[lhs.cast<BlockArgument>().getArgNumber()];
    int64_t lowerBound = getOneConstantValue(lhsInput);
    if (lowerBound < 0)
      return failure();

    // Check the upper bound of the break condition.
    int64_t upperBound =
        constantOrFedByConstant(ubValue, genericLoopOp, genericReturnOp);
    if (upperBound < 0)
      return failure();

    // Replace the max trip count by a new trip count if the new trip count is
    // smaller.
    int64_t derivedTripCount = (int64_t)((upperBound - lowerBound) / step);
    if (maxTripCount <= derivedTripCount)
      return failure();

    SmallVector<int64_t, 1> values(1, derivedTripCount);
    DenseElementsAttr valueAttr = DenseElementsAttr::get(
        RankedTensorType::get({},
            maxTripCountValue.getType().cast<ShapedType>().getElementType()),
        makeArrayRef(values));
    Value newMaxTripCountValue =
        onnx_mlir::createONNXConstantOpWithDenseAttr(rewriter, loc, valueAttr);

    maxTripCountValue.replaceAllUsesWith(newMaxTripCountValue);
    return success();
  }
};

/// Register optimization patterns as "canonicalization" patterns
/// on the ONNXAddOp.
void ONNXAddOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<NormalizeAddPattern>(context);
  results.insert<MulAddToGemmOptPattern>(context);
  results.insert<FuseGemmFollowedByAddition>(context);
  results.insert<FuseAddConvPattern>(context);
  results.insert<FuseAddConvNullBiasPattern>(context);
}

/// on the ONNXMulOp.
void ONNXMulOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<NormalizeMulPattern>(context);
  results.insert<FuseMulConvNullBiasPattern>(context);
}

/// on the ONNXIdentityOp.
void ONNXIdentityOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<IdentityEliminationPattern>(context);
}

/// on the ONNXCastOp.
void ONNXCastOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<CastEliminationPattern>(context);
}

/// on the ONNXTransposeOp.
void ONNXTransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<FuseTransposePattern>(context);
  result.insert<RemoveIdentityTransposePattern>(context);
  result.insert<SwapTransposeConcatPattern>(context);
}

/// on the ONNXReshapeOp.
void ONNXReshapeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<FuseReshapePattern>(context);
  result.insert<RemoveIdentityReshapePattern>(context);
  result.insert<SwapReshapeMatMulPattern>(context);
}

/// on the ONNXDropoutOp.
void ONNXDropoutOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<DropoutEliminationPattern>(context);
}

/// on the ONNXSqueezeOp.
void ONNXSqueezeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveSqueezeUnsqueezePattern>(context);
}

void ONNXSqueezeV11Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveSqueezeV11UnsqueezeV11Pattern>(context);
}

/// on the ONNXUnsqueezeOp.
void ONNXUnsqueezeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveUnsqueezeSqueezePattern>(context);
}

void ONNXUnsqueezeV11Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveUnsqueezeV11SqueezeV11Pattern>(context);
}

/// on the ONNXBatchNormalizationInferenceModeOp.
void ONNXBatchNormalizationInferenceModeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FuseBatchNormInferenceModeConvPattern>(context);
  results.insert<RewriteBatchNormInferenceModeConvPattern1>(context);
  results.insert<RewriteBatchNormInferenceModeConvPattern2>(context);
}

/// on the ONNXShapeOp.
void ONNXShapeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ShapeToConstantPattern>(context);
}

/// on the ONNXSizeOp.
void ONNXSizeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<SizeToConstantPattern>(context);
}

/// on the ONNXGlobalAveragePoolOp.
void ONNXGlobalAveragePoolOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<GlobalAveragePoolPattern>(context);
}

/// on the ONNXGlobalMaxPoolOp.
void ONNXGlobalMaxPoolOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<GlobalMaxPoolPattern>(context);
}

/// on the ONNXConstantOp.
void ONNXConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ConstantOpNormalizationPattern1>(context);
  results.insert<ConstantOpNormalizationPattern2>(context);
  results.insert<ConstantOpNormalizationPattern3>(context);
  results.insert<ConstantOpNormalizationPattern4>(context);
  results.insert<ConstantOpNormalizationPattern5>(context);
  results.insert<ConstantOpNormalizationPattern6>(context);
}

/// on the ONNXLessOp.
void ONNXLessOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<LessOpSameCastPattern>(context);
}

/// on the ONNXLoopOp.
void ONNXLoopOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<LoopOpRewriteMaxTripCountPattern>(context);
}
