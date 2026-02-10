// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/// Create a dense constant tensor<i64> (1D) suitable for ONNX Slice parameters.
static Value createI64ConstTensor(PatternRewriter &rewriter, Location loc,
    const llvm::ArrayRef<int64_t> &values) {
  auto tensorType = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, rewriter.getI64Type());
  auto denseAttr = DenseElementsAttr::get(tensorType, values);
  return rewriter.create<ONNXConstantOp>(loc, tensorType,
      /*sparse_value=*/Attribute(),
      /*value=*/denseAttr, /*value_float=*/FloatAttr(),
      /*value_floats=*/ArrayAttr(), /*value_int=*/IntegerAttr(),
      /*value_ints=*/ArrayAttr(), /*value_string=*/StringAttr(),
      /*value_strings=*/ArrayAttr());
}

/// Concat -> (single user) InstanceNormalization -> (single user) Conv.
static bool checkInstanceNormConv2d(ONNXConcatOp concatOp) {
  if (!concatOp->hasOneUse())
    return false;
  Operation *u0 = *concatOp.getResult().getUsers().begin();
  auto instOp = dyn_cast<ONNXInstanceNormalizationOp>(u0);
  if (!instOp || !instOp->hasOneUse())
    return false;
  Operation *u1 = *instOp.getResult().getUsers().begin();
  return isa<ONNXConvOp>(u1);
}

static bool checkEliminationOpportunity(ONNXConcatOp concatOp) {
  return checkInstanceNormConv2d(concatOp);
}

/// Rewrite one of two sibling concats that share exactly one input by swapping
/// its inputs and preserving semantics via Slice+Concat.
struct OptimizeSiblingConcatPattern : public OpRewritePattern<ONNXConcatOp> {
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const override {
    Location loc = concatOp.getLoc();

    auto inputs = concatOp.getInputs();
    if (inputs.size() != 2)
      return failure();

    auto axisAttrLiteral = concatOp.getAxisAttr();
    if (!axisAttrLiteral)
      return failure();
    int64_t axis = axisAttrLiteral.getSInt();

    auto outTy = dyn_cast<RankedTensorType>(concatOp.getType());
    if (!outTy || outTy.getRank() != 4 || axis != 1)
      return failure();
    if (!outTy.hasStaticShape())
      return failure();

    // Find the unique sibling concat (other concat that uses any input of
    // concatOp).
    llvm::SmallPtrSet<Operation *, 4> siblingSet;
    for (Value in : inputs) {
      for (Operation *user : in.getUsers()) {
        auto otherConcat = dyn_cast<ONNXConcatOp>(user);
        if (!otherConcat || otherConcat == concatOp)
          continue;
        siblingSet.insert(otherConcat.getOperation());
      }
    }
    if (siblingSet.size() != 1)
      return failure();

    auto siblingOp = dyn_cast<ONNXConcatOp>(*siblingSet.begin());
    if (!siblingOp)
      return failure();

    if (siblingOp.getInputs().size() != 2)
      return failure();
    auto siblingAxisAttrLiteral = siblingOp.getAxisAttr();
    if (!siblingAxisAttrLiteral || siblingAxisAttrLiteral.getSInt() != axis)
      return failure();
    auto siblingOutTy = dyn_cast<RankedTensorType>(siblingOp.getType());
    if (!siblingOutTy || siblingOutTy.getRank() != 4 ||
        !siblingOutTy.hasStaticShape())
      return failure();

    // Identify the shared input value.
    Value shareInput;
    if (inputs[0] == siblingOp.getInputs()[0] ||
        inputs[0] == siblingOp.getInputs()[1]) {
      shareInput = inputs[0];
    } else if (inputs[1] == siblingOp.getInputs()[0] ||
               inputs[1] == siblingOp.getInputs()[1]) {
      shareInput = inputs[1];
    } else {
      return failure();
    }

    // Enforce "no other sibling concats": for each input of both concats, the
    // concat users must be a subset of {concatOp, siblingOp}.
    auto isOnlyUsedByTheseConcats = [&](Value v) -> bool {
      for (Operation *user : v.getUsers()) {
        auto c = dyn_cast<ONNXConcatOp>(user);
        if (!c)
          continue;
        if (c != concatOp && c != siblingOp)
          return false;
      }
      return true;
    };
    if (!isOnlyUsedByTheseConcats(concatOp.getInputs()[0]) ||
        !isOnlyUsedByTheseConcats(concatOp.getInputs()[1]) ||
        !isOnlyUsedByTheseConcats(siblingOp.getInputs()[0]) ||
        !isOnlyUsedByTheseConcats(siblingOp.getInputs()[1]) ||
        !isOnlyUsedByTheseConcats(shareInput))
      return failure();

    int concatShareIdx = (concatOp.getInputs()[0] == shareInput) ? 0 : 1;
    int siblingShareIdx = (siblingOp.getInputs()[0] == shareInput) ? 0 : 1;
    if (concatShareIdx != siblingShareIdx)
      return failure();

    // Choose which concat to rewrite.
    ONNXConcatOp targetConcat;
    if (checkEliminationOpportunity(concatOp))
      targetConcat = concatOp;
    else if (checkEliminationOpportunity(siblingOp))
      targetConcat = siblingOp;
    else
      return failure();

    // Only rewrite when invoked on the chosen target concat.
    if (targetConcat != concatOp)
      return failure();

    auto targetInputs = targetConcat.getInputs();
    if (targetInputs.size() != 2)
      return failure();

    auto targetOutTy = dyn_cast<RankedTensorType>(targetConcat.getType());
    if (!targetOutTy || !targetOutTy.hasStaticShape())
      return failure();

    // Types for slices correspond to original inputs.
    auto in0Ty = dyn_cast<RankedTensorType>(targetInputs[0].getType());
    auto in1Ty = dyn_cast<RankedTensorType>(targetInputs[1].getType());
    if (!in0Ty || !in1Ty || !in0Ty.hasStaticShape() || !in1Ty.hasStaticShape())
      return failure();
    if (in0Ty.getRank() != 4 || in1Ty.getRank() != 4)
      return failure();

    int64_t input1AxisSize = in1Ty.getShape()[axis];
    if (ShapedType::isDynamic(input1AxisSize))
      return failure();

    rewriter.setInsertionPoint(targetConcat);

    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);
    auto axisAttr = IntegerAttr::get(si64Type, axis);

    auto swappedConcat = rewriter.create<ONNXConcatOp>(loc, targetOutTy,
        ValueRange{targetInputs[1], targetInputs[0]}, axisAttr);

    SmallVector<int64_t> axesVec{0, 1, 2, 3};
    SmallVector<int64_t> stepsVec{1, 1, 1, 1};

    SmallVector<int64_t> begin1{0, 0, 0, 0};
    begin1[axis] = input1AxisSize;
    SmallVector<int64_t> end1(
        targetOutTy.getShape().begin(), targetOutTy.getShape().end());

    SmallVector<int64_t> begin2{0, 0, 0, 0};
    SmallVector<int64_t> end2(
        targetOutTy.getShape().begin(), targetOutTy.getShape().end());
    end2[axis] = input1AxisSize;

    Value starts1 = createI64ConstTensor(rewriter, loc, begin1);
    Value ends1 = createI64ConstTensor(rewriter, loc, end1);
    Value axes = createI64ConstTensor(rewriter, loc, axesVec);
    Value steps = createI64ConstTensor(rewriter, loc, stepsVec);
    Value starts2 = createI64ConstTensor(rewriter, loc, begin2);
    Value ends2 = createI64ConstTensor(rewriter, loc, end2);

    auto slice1 = rewriter.create<ONNXSliceOp>(
        loc, in0Ty, swappedConcat.getResult(), starts1, ends1, axes, steps);
    auto slice2 = rewriter.create<ONNXSliceOp>(
        loc, in1Ty, swappedConcat.getResult(), starts2, ends2, axes, steps);

    auto reconcat = rewriter.create<ONNXConcatOp>(loc, targetOutTy,
        ValueRange{slice1.getOutput(), slice2.getOutput()}, axisAttr);

    rewriter.replaceOp(targetConcat, reconcat.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct OptimizeSiblingConcatPass : public PassWrapper<OptimizeSiblingConcatPass,
                                       OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "optimize-sibling-concat"; }
  StringRef getDescription() const override {
    return "Optimize sibling 2-input concat ops that share one input by "
           "swapping one concat input order and preserving semantics with "
           "Slice+Concat";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<OptimizeSiblingConcatPattern>(context);

    GreedyRewriteConfig config;
    config.maxIterations = 10;
    config.useTopDownTraversal = false;

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createOptimizeSiblingConcatPass() {
  return std::make_unique<OptimizeSiblingConcatPass>();
}

} // namespace onnx_mlir
