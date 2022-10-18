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

#include <math.h>

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

// =============================================================================
// Helper functions for Rewrite.td and Rewrite.cpp files.
// =============================================================================

// If 'A' is NoneType, return -B. Otherwise return A-B.
Value subtractOrNeg(PatternRewriter &rewriter, Location loc, Value A, Value B) {
  if (A.getType().isa<NoneType>())
    return rewriter.create<ONNXNegOp>(loc, B);
  return rewriter.create<ONNXSubOp>(loc, A, B);
}

// Create an ArrayAttr of IntegerAttr(s) of values in [1, N].
ArrayAttr createArrayAttrOfOneToN(PatternRewriter &rewriter, int N) {
  SmallVector<int64_t, 4> vals;
  for (int i = 1; i <= N; ++i)
    vals.emplace_back(i);
  return rewriter.getI64ArrayAttr(vals);
}

// Create an ArrayAttr of IntegerAttr(s) of values in [N, M].
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

// Create a DenseElementsAttr based on the shape of type.
DenseElementsAttr createDenseElementsAttrFromShape(PatternRewriter &rewriter,
    Value value, int64_t start = 0, llvm::Optional<int64_t> end = llvm::None) {

  auto inType = value.getType().cast<ShapedType>();
  assert(inType.hasRank() && "inType must be ranked");
  auto shape = inType.getShape();
  int64_t rank = inType.getRank();

  int64_t endValue = end.has_value() ? end.value() : rank;

  SmallVector<int64_t, 1> dims = {endValue - start};
  SmallVector<int64_t, 4> values(
      shape.begin() + start, shape.begin() + endValue);
  auto tensorType = RankedTensorType::get(dims, rewriter.getIntegerType(64));
  return DenseElementsAttr::get(tensorType, makeArrayRef(values));
}

// Create a DenseElementsAttr from Shape Op
DenseElementsAttr createDenseElementsAttrFromShapeOp(
    PatternRewriter &rewriter, Operation *op) {
  ONNXShapeOp shapeOp = llvm::cast<ONNXShapeOp>(op);
  ONNXShapeOpAdaptor operandAdaptor(shapeOp);

  int64_t start, end;
  std::tie(start, end) = getDataShapeBounds(operandAdaptor);

  return createDenseElementsAttrFromShape(rewriter, shapeOp.data(), start, end);
}

} // namespace onnx_mlir

// =============================================================================
/// Include the patterns defined in the Declarative Rewrite framework.
// =============================================================================

#include "src/Dialect/ONNX/ONNXRewrite.inc"

// =============================================================================
// Rewrite pattern for loop (not handled in Rewrite.td).
// =============================================================================

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
// keepGoing = true
// while (i < max_trip_count && keepGoing == true) {
//    k = k + STEP
//    keepGoing = (k < UB)
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
// keepGoing = true
// while (i < max_trip_count && keepGoing == true) {
//    k = k + STEP
// }
// ```
// where `max_trip_count` is replaced by an actual value derived from the loop.
//
class LoopOpRewriteMaxTripCountPattern : public OpRewritePattern<ONNXLoopOp> {
public:
  using OpRewritePattern<ONNXLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXLoopOp onnxLoopOp, PatternRewriter &rewriter) const override {
    Location loc = onnxLoopOp.getLoc();
    Operation *loopOp = onnxLoopOp.getOperation();
    Value maxTripCountValue = loopOp->getOperands()[0];

    // Match the following pattern:
    // ```
    // ubValue = ONNXConstantOp() {value = ...}
    // startValue = ONNXConstantOp() {value = ...}
    // ONNXLoop(max_trip_count, true, ..., ubValue, ..., startValue, ...)
    //   ^bb(max_trip_count, cond, ..., ubValue, ..., counterValue, ...):
    //     stepValue = ONNXConstantOp() {value = ...}
    //     newCounterValue = ONNXAddOp(counterValue, stepValue).
    //     cond_new = cond
    //     ONNXReturnOp (cond_new, ..., ubValue, ..., newCounterValue, ...)
    // ```
    bool matched;
    Value newMaxTripCountValue;
    std::tie(matched, newMaxTripCountValue) =
        matchOp(rewriter, loc, onnxLoopOp);
    if (!matched)
      return failure();

    // Rewrite
    loopOp->replaceUsesOfWith(maxTripCountValue, newMaxTripCountValue);
    // Modify the condition return
    Region &loopBody = onnxLoopOp.body();
    Operation *loopBodyTerminator = loopBody.front().getTerminator();
    loopBodyTerminator->setOperand(0, loopBody.front().getArgument(1));
    return success();
  }

private:
  // A helper function to check whether a value is defined by ONNXConstantOp in
  // the same block or not.
  bool isDefinedByIntegerConstantOp(Value v) const {
    if (v.isa<BlockArgument>())
      return false;
    Operation *definingOp = v.getDefiningOp();
    if (v.getType().cast<ShapedType>().getElementType().isa<IntegerType>() &&
        isa<ONNXConstantOp>(definingOp) &&
        cast<ONNXConstantOp>(definingOp).valueAttr().isa<DenseElementsAttr>())
      return true;
    return false;
  }

  // A helper function to check whether an block argument is invariant to
  // iterations or not. By the definition of LoopOp, input block arguments are
  // shifted by 1 to the left in ReturnOp. If a block argument is unchanged when
  // being shifted in ReturnOp, then it is invariant to iterations.
  bool isInvariantBlockArg(Value v, Operation *returnOp) const {
    return v.isa<BlockArgument>() &&
           (v ==
               returnOp
                   ->getOperands()[v.cast<BlockArgument>().getArgNumber() - 1]);
  }

  // A helper function to check whether a value is defined by ONNXConstantOp in
  // the same block or an invariant block argument.
  bool isIntConstantOrInvariantBlockArg(Value v, Operation *returnOp) const {
    return ((v.isa<BlockArgument>() && isInvariantBlockArg(v, returnOp)) ||
            (!v.isa<BlockArgument>() && isDefinedByIntegerConstantOp(v)));
  }

  // A helper function to check whether an block argument is updated by a Value
  // inside the loop or not.
  bool isUpdatedArgByValue(Value v, Value newV, Operation *returnOp) const {
    return v.isa<BlockArgument>() &&
           (newV ==
               returnOp
                   ->getOperands()[v.cast<BlockArgument>().getArgNumber() - 1]);
  }

  // A helper function to get the value that is fed to an operation's argument.
  Value getFedValue(Value arg, Operation *op) const {
    return op->getOperands()[arg.cast<BlockArgument>().getArgNumber()];
  }

  // A helper function to get an integer constant from a value.
  int64_t getOneIntegerConstant(Value v) const {
    Operation *definingOp = v.getDefiningOp();
    DenseElementsAttr valueAttr =
        cast<ONNXConstantOp>(definingOp).valueAttr().cast<DenseElementsAttr>();
    return (*valueAttr.getValues<APInt>().begin()).getSExtValue();
  }

  // A helper function to match the pattern of the given operation. It also
  // returns a constant value for the max trip count during the matching, which
  // is to avoid recomputing values in the rewriting phase.
  //
  // Pattern:
  // ```
  // ubValue = ONNXConstantOp() {value = ...}
  // startValue = ONNXConstantOp() {value = ...}
  // ONNXLoop(max_trip_count, true, ..., ubValue, ..., startValue, ...)
  //   ^bb(max_trip_count, cond, ..., ubValue, ..., counterValue, ...):
  //     stepValue = ONNXConstantOp() {value = ...}
  //     newCounterValue = ONNXAddOp(counterValue, stepValue).
  //     cond = LessOp(newCounterValue, ubValue)
  //     ONNXReturnOp (cond, ..., ubValue, ..., newCounterValue, ...)
  // ```
  std::pair<bool, Value> matchOp(
      PatternRewriter &rewriter, Location loc, ONNXLoopOp onnxLoopOp) const {
    OnnxBuilder onnx(rewriter, loc);
    Operation *loopOp = onnxLoopOp.getOperation();
    Value maxTripCountValue = loopOp->getOperands()[0];

    // The maximum trip count is a constant.
    if (!isDefinedByIntegerConstantOp(maxTripCountValue))
      return std::make_pair(false, maxTripCountValue);

    // Get the loop region.
    Region &loopBody = onnxLoopOp.body();
    // Make sure the region has only one block.
    if (!loopBody.hasOneBlock())
      return std::make_pair(false, maxTripCountValue);

    // Get ReturnOp of the body block.
    Block &bodyBlock = loopBody.front();
    Operation *returnOp = bodyBlock.getTerminator();
    if (!isa<ONNXReturnOp>(returnOp))
      return std::make_pair(false, maxTripCountValue);

    // Analyze the break condition of the loop body to see if we can derive a
    // new maximum trip count or not.

    // The break condition is the first argument of ReturnOp.
    // `ONNXReturnOp (cond, ..., ubValue, ..., newCounterValue, ...)`
    Value breakCond = returnOp->getOperands()[0];
    if (breakCond.isa<BlockArgument>())
      return std::make_pair(false, maxTripCountValue);
    Operation *breakCondOp = breakCond.getDefiningOp();

    // Only support LessOp as the op that defines the break condition at this
    // moment.
    // `cond = LessOp(newCounterValue, ubValue)`
    if (!isa<ONNXLessOp>(breakCondOp))
      return std::make_pair(false, maxTripCountValue);
    Value newCounterValue = breakCondOp->getOperands()[0];
    Value ubValue = breakCondOp->getOperands()[1];
    // Input type of Less must be integer.
    if (!newCounterValue.getType()
             .cast<ShapedType>()
             .getElementType()
             .isa<IntegerType>())
      return std::make_pair(false, maxTripCountValue);

    // Compute a trip count from the break condition, given that the upper bound
    // is fixed and the lower bound is increased by a constant step at each
    // iteration. So, the trip count will be `(upper_bound - lower_bound)/step`.

    // Only support ONNXAddOp at this moment.
    if (newCounterValue.isa<BlockArgument>() ||
        !isa<ONNXAddOp>(newCounterValue.getDefiningOp()))
      return std::make_pair(false, maxTripCountValue);
    // ONNXLoop(max_trip_count, true, ..., ubValue, ..., startValue, ...)
    //   ^bb(max_trip_count, cond, ..., ubValue, ..., counterValue, ...):
    //     stepValue = ONNXConstantOp() {value = ...}
    //     newCounterValue = ONNXAddOp(counterValue, stepValue).
    //     cond = LessOp(newCounterValue, ubValue)
    //     ONNXReturnOp (cond, ..., ubValue, ..., newCounterValue, ...)
    Operation *addOp = cast<ONNXAddOp>(newCounterValue.getDefiningOp());
    Value counterValue = addOp->getOperands()[0];
    Value stepValue = addOp->getOperands()[1];
    // Counter is a block argument and updated at each iteration.
    if (!isUpdatedArgByValue(counterValue, newCounterValue, returnOp))
      return std::make_pair(false, maxTripCountValue);
    // Step must be a constant inside the loop or an invariant argument.
    if (!isIntConstantOrInvariantBlockArg(stepValue, returnOp))
      return std::make_pair(false, maxTripCountValue);

    // Check the lower bound of the break condition.
    // LowerBound is the initial value of the counter.
    Value lbValue = getFedValue(counterValue, loopOp);

    // Check the upper bound of the break condition.
    // UpperBound must be a constant inside the loop or an invariant argument.
    if (!isIntConstantOrInvariantBlockArg(ubValue, returnOp))
      return std::make_pair(false, maxTripCountValue);

    // Get values for upper bound and step if they are invariant arguments.
    // Otherwise, clone them to location outside the loop.
    if (isInvariantBlockArg(ubValue, returnOp))
      ubValue = getFedValue(ubValue, loopOp);
    else
      ubValue = cast<ONNXConstantOp>(rewriter.clone(*ubValue.getDefiningOp()))
                    .getResult();
    if (isInvariantBlockArg(stepValue, returnOp))
      stepValue = getFedValue(stepValue, loopOp);
    else
      stepValue =
          cast<ONNXConstantOp>(rewriter.clone(*stepValue.getDefiningOp()))
              .getResult();

    // Case 1: the upper bound, lower bound and step are constants.
    // - Compute the new max trip count at the compile time.
    if (isDefinedByIntegerConstantOp(lbValue) &&
        isDefinedByIntegerConstantOp(ubValue) &&
        isDefinedByIntegerConstantOp(stepValue)) {
      int64_t lowerBound = getOneIntegerConstant(lbValue);
      int64_t upperBound = getOneIntegerConstant(ubValue);
      int64_t step = getOneIntegerConstant(stepValue);
      if ((step <= 0) || (upperBound <= lowerBound))
        return std::make_pair(false, maxTripCountValue);
      int64_t derivedTripCount =
          ceil((1.0 * (upperBound - lowerBound)) / (1.0 * step));
      int64_t maxTripCount = getOneIntegerConstant(maxTripCountValue);

      // Check that the new trip count is smaller than the original trip count.
      if (maxTripCount <= derivedTripCount)
        return std::make_pair(false, maxTripCountValue);

      SmallVector<int64_t, 1> values(1, derivedTripCount);
      DenseElementsAttr valueAttr = DenseElementsAttr::get(
          RankedTensorType::get({},
              maxTripCountValue.getType().cast<ShapedType>().getElementType()),
          makeArrayRef(values));
      return std::make_pair(true, onnx.constant(valueAttr));
    }

    // Case 2: Not all of the lower bound, upper bound and step are constants,
    // emit code to compute the new max trip count.
    // - new_max_trip_count =
    //      min(old_max_trip_count, ceil(upper_bound - lower_bound)/step)
    TypeAttr tripCountType = TypeAttr::get(
        maxTripCountValue.getType().cast<ShapedType>().getElementType());

    // Cast the upper and lower bounds to the correct type.
    if (maxTripCountValue.getType().cast<ShapedType>().getElementType() !=
        ubValue.getType().cast<ShapedType>().getElementType())
      ubValue = onnx.cast(ubValue, tripCountType);
    if (maxTripCountValue.getType().cast<ShapedType>().getElementType() !=
        lbValue.getType().cast<ShapedType>().getElementType())
      lbValue = onnx.cast(lbValue, tripCountType);

    // Emit code to compute the max trip count.
    Value range = onnx.sub(ubValue, lbValue);
    Value rangeInFloat = onnx.cast(range, TypeAttr::get(rewriter.getF32Type()));
    Value stepInFloat =
        onnx.cast(stepValue, TypeAttr::get(rewriter.getF32Type()));
    Value tripCountInFloat = onnx.ceil(onnx.div(rangeInFloat, stepInFloat));
    Value newMaxTripCountValue = onnx.cast(tripCountInFloat, tripCountType);

    return std::make_pair(
        true, onnx.min(ValueRange({maxTripCountValue, newMaxTripCountValue})));
  }
};

namespace {
// RNNOpRewriteLayoutPattern helper functions and classes.

template <typename ONNXOp>
void inferShapes(ONNXOp op) {
  if (failed(op.inferShapes([](Region &region) {})))
    llvm_unreachable("unexpected inferShapes failure");
}

// To transpose between [batch_size, seq_length/num_directions, size]
//                  and [seq_length/num_directions, batch_size, size].
ArrayAttr perm3RNN(Builder &b) { return b.getI64ArrayAttr({1, 0, 2}); }

// To transpose from [seq_length, num_directions, batch_size, hidden_size]
//                to [batch_size, seq_length, num_directions, hidden_size].
ArrayAttr perm4RNN(Builder &b) { return b.getI64ArrayAttr({2, 0, 1, 3}); }

class InputOutputTransposer {
public:
  InputOutputTransposer(mlir::OpBuilder &b, mlir::Location loc)
      : create(b, loc) {}

  void transposeInput(MutableOperandRange operand, ArrayAttr perm) {
    assert(operand.size() == 1 && "should be called with singleton range");
    Value input = operand[0];
    if (!input.getType().isa<NoneType>()) {
      Value transposed = transpose(input, perm);
      operand.assign(transposed);
    }
  }

  void transposeOutput(Value output, ArrayAttr perm) {
    if (!output.getType().isa<NoneType>()) {
      Value transposed = transpose(output, perm);
      output.replaceAllUsesExcept(transposed, transposed.getDefiningOp());
    }
  }

private:
  // Helper to create an ONNX transposition, using
  // ONNXTransposeOp::inferShapes() to infer the output shape.
  Value transpose(Value input, ArrayAttr perm) {
    Type elType = onnx_mlir::getElementType(input.getType());
    Type unrankedType = UnrankedTensorType::get({elType}); // placeholder
    Value transposed = create.transpose(unrankedType, input, perm);
    auto transposeOp = llvm::cast<ONNXTransposeOp>(transposed.getDefiningOp());
    inferShapes(transposeOp); // sets transposed's shape
    return transposed;
  }

  onnx_mlir::OnnxBuilder create;
};
} // namespace

// Rewrites layout=1 to layout=0 by transposing inputs and outputs.
template <typename ONNXOp>
class RNNOpRewriteLayoutPattern : public OpRewritePattern<ONNXOp> {
public:
  using OpRewritePattern<ONNXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXOp onnxOp, PatternRewriter &rewriter) const override {
    if (onnxOp.layout() == 0) {
      return success();
    }

    InputOutputTransposer transposer(rewriter, onnxOp.getLoc());
    ArrayAttr perm3 = perm3RNN(rewriter);

    // LSTM requires extra work for initial_c input and Y_c output.
    auto onnxLSTMOp = llvm::dyn_cast<ONNXLSTMOp>(*onnxOp);

    // Rewrite in-place because there are so many attributes, inputs, outputs.
    // Constructing a new op would be lengthy and hard to maintain.
    rewriter.updateRootInPlace(onnxOp, [&]() {
      // Transpose the X and initial_h inputs by inserting an ONNXTransposeOp
      // before each and replacing the each input with the transpose output.
      rewriter.setInsertionPoint(onnxOp); // insert before (redundant)
      transposer.transposeInput(onnxOp.XMutable(), perm3);
      transposer.transposeInput(onnxOp.initial_hMutable(), perm3);
      if (onnxLSTMOp)
        transposer.transposeInput(onnxLSTMOp.initial_cMutable(), perm3);
      // Set layout to zero.
      onnxOp->setAttr(onnxOp.layoutAttrName(),
          rewriter.getIntegerAttr(
              rewriter.getIntegerType(64, /*isSigned=*/true), 0));
      // Update the output shape.
      inferShapes(onnxOp);
    });
    // Transpose the Y and Y_h outputs by inserting an ONNXTransposeOp
    // after each and replace all uses of each with the transpose output.
    ValueRange results = onnxOp.getResults();
    if (results.size() > 0) {
      rewriter.setInsertionPointAfter(onnxOp);
      transposer.transposeOutput(onnxOp.Y(), perm4RNN(rewriter));
      transposer.transposeOutput(onnxOp.Y_h(), perm3);
      if (onnxLSTMOp)
        transposer.transposeOutput(onnxLSTMOp.Y_c(), perm3);
    }

    return success();
  }
};

// =============================================================================
/// Register optimization patterns as "canonicalization" patterns.
/// Add op to OpsWithCanonicalizer in gen_onnx_mlir.py to activate.
// =============================================================================

/// on the ONNXAddOp.
void ONNXAddOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<NormalizeAddPattern>(context);
  results.insert<MulAddToGemmOptPattern>(context);
  results.insert<FuseGemmFollowedByAddition>(context);
  results.insert<FuseAddConvPattern>(context);
  results.insert<FuseAddConvNullBiasPattern>(context);
}

/// on the ONNXDimOp.
void ONNXDimOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<DimOpToConstantPattern>(context);
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
  result.insert<FuseCastCastPattern>(context);
}

/// on the ONNXTransposeOp.
void ONNXTransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<FuseTransposePattern>(context);
  result.insert<FuseTransposeAndAtanPattern>(context);
  result.insert<FuseTransposeAndCastPattern>(context);
  result.insert<FuseTransposeAndCeilPattern>(context);
  result.insert<FuseTransposeAndCosPattern>(context);
  result.insert<FuseTransposeAndCoshPattern>(context);
  result.insert<FuseTransposeAndEluPattern>(context);
  result.insert<FuseTransposeAndErfPattern>(context);
  result.insert<FuseTransposeAndAcosPattern>(context);
  result.insert<FuseTransposeAndAcoshPattern>(context);
  result.insert<FuseTransposeAndAsinPattern>(context);
  result.insert<FuseTransposeAndAsinhPattern>(context);
  result.insert<FuseTransposeAndAtanhPattern>(context);
  result.insert<FuseTransposeAndExpPattern>(context);
  result.insert<FuseTransposeAndFloorPattern>(context);
  result.insert<FuseTransposeAndHardSigmoidPattern>(context);
  result.insert<FuseTransposeAndIsNaNPattern>(context);
  result.insert<FuseTransposeAndLeakyReluPattern>(context);
  result.insert<FuseTransposeAndLogPattern>(context);
  result.insert<FuseTransposeAndNegPattern>(context);
  result.insert<FuseTransposeAndNotPattern>(context);
  result.insert<FuseTransposeAndReciprocalPattern>(context);
  result.insert<FuseTransposeAndReluPattern>(context);
  result.insert<FuseTransposeAndRoundPattern>(context);
  result.insert<FuseTransposeAndSeluPattern>(context);
  result.insert<FuseTransposeAndSigmoidPattern>(context);
  result.insert<FuseTransposeAndSignPattern>(context);
  result.insert<FuseTransposeAndSinPattern>(context);
  result.insert<FuseTransposeAndSinhPattern>(context);
  result.insert<FuseTransposeAndSoftplusPattern>(context);
  result.insert<FuseTransposeAndSoftsignPattern>(context);
  result.insert<FuseTransposeAndSqrtPattern>(context);
  result.insert<FuseTransposeAndTanPattern>(context);
  result.insert<FuseTransposeAndTanhPattern>(context);
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
  result.insert<RemoveSqueezeCastUnsqueezePattern>(context);
}

void ONNXSqueezeV11Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveSqueezeV11UnsqueezeV11Pattern>(context);
  result.insert<RemoveSqueezeV11CastUnsqueezeV11Pattern>(context);
}

/// on the ONNXUnsqueezeOp.
void ONNXUnsqueezeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveUnsqueezeSqueezePattern>(context);
  result.insert<RemoveUnsqueezeCastSqueezePattern>(context);
}

void ONNXUnsqueezeV11Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveUnsqueezeV11SqueezeV11Pattern>(context);
  result.insert<RemoveUnsqueezeV11CastSqueezeV11Pattern>(context);
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

/// on the ONNXSpaceToDepthOp.
void ONNXSpaceToDepthOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveSpaceToDepthDepthToSpacePattern>(context);
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

/// on the ONNXDepthToSpaceOp.
void ONNXDepthToSpaceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveDepthToSpaceSpaceToDepthPattern>(context);
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

/// on the ONNXGRUOp.
void ONNXGRUOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RNNOpRewriteLayoutPattern<ONNXGRUOp>>(context);
}

/// on the ONNXLSTMOp.
void ONNXLSTMOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RNNOpRewriteLayoutPattern<ONNXLSTMOp>>(context);
}

/// on the ONNXRNNOp.
void ONNXRNNOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RNNOpRewriteLayoutPattern<ONNXRNNOp>>(context);
}
