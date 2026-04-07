// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass mirrors xcompiler dev OptimizeSliceReshapeTransposeBlockPass
// (get_template_block / get_template_block_mha + optimize_block / optimize_block_mha):
//
// Matmul (get_template_block filter): rank-3 reshape inputs; transpose orders are
//   exactly {0,2,1,3}, {0,2,3,1}, {0,2,1,3} on branches 0,1,2 (middle branch is
//   {0,2,3,1}). optimize_block fuses reshape+transpose then slices; only the middle
//   branch gets extra transpose {0,1,3,2} after its strided_slice.
//
// MHA (get_template_block_mha): same ranks; all three transposes {0,2,1,3}.
//   optimize_block_mha does not add that extra transpose.
//
// Transform (common):
//   input -> [slice_0, slice_1, slice_2] -> [reshape_0, reshape_1, reshape_2]
//          -> [transpose_0, transpose_1, transpose_2] -> consumer
// Into:
//   input -> reshape -> transpose {0,2,1,3} -> [slice_x, slice_y, (+ maybe Transpose)]
//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static size_t getRank(Type type) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  return ranked ? ranked.getRank() : 0;
}

/// Helper to get element type from a tensor type
Type getElementType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.getElementType();
  }
  return nullptr;
}

/// Create a constant op with the given int64 values
Value createConstantI64Array(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> values) {
  MLIRContext *ctx = rewriter.getContext();
  auto tensorType = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, IntegerType::get(ctx, 64));
  auto denseAttr = DenseIntElementsAttr::get(tensorType, values);
  return rewriter.create<ONNXConstantOp>(loc, Attribute(), denseAttr);
}

/// Reshape output layout: axis 0=N, 1=S, 2=H, 3=D. ONNX Transpose: output dim i maps
/// from input dim perm[i]. Valid "head-split" patterns put N,H first and S,D last:
/// perm[0]==0, perm[1]==2, and {perm[2],perm[3]} == {1,3}.
/// Sets \p needsSwapSdOnOutput when the branch output had (N,H,D,S) vs fused (N,H,S,D).
static bool parseHeadSplitTranspose(
    ArrayAttr permAttr, bool &needsSwapSdOnOutput) {
  if (!permAttr || permAttr.size() != 4)
    return false;
  int64_t p0 = cast<IntegerAttr>(permAttr[0]).getInt();
  int64_t p1 = cast<IntegerAttr>(permAttr[1]).getInt();
  int64_t p2 = cast<IntegerAttr>(permAttr[2]).getInt();
  int64_t p3 = cast<IntegerAttr>(permAttr[3]).getInt();
  if (p0 != 0 || p1 != 2)
    return false;
  if (p2 == 1 && p3 == 3) {
    needsSwapSdOnOutput = false;
    return true;
  }
  if (p2 == 3 && p3 == 1) {
    needsSwapSdOnOutput = true;
    return true;
  }
  return false;
}

/// Read one element from a 1-D onnx.Constant tensor (typical slice starts/steps).
static bool tryGet1DConstI64Element(Value v, size_t index, int64_t &out) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  auto cst = dyn_cast<ONNXConstantOp>(def);
  if (!cst || !cst.getValueAttr())
    return false;
  auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValueAttr());
  if (!dense || dense.getType().getRank() != 1)
    return false;
  int64_t dim = dense.getType().getDimSize(0);
  if (index >= static_cast<size_t>(dim))
    return false;
  size_t i = 0;
  for (int64_t el : dense.getValues<int64_t>()) {
    if (i == index) {
      out = el;
      return true;
    }
    ++i;
  }
  return false;
}

/// Channel-axis start for a typical rank-3 QKV slice (`starts[2]`).
static FailureOr<int64_t> getChannelSliceStartKey(ONNXSliceOp slice) {
  int64_t key = 0;
  if (!tryGet1DConstI64Element(slice.getStarts(), 2, key))
    return failure();
  return key;
}

struct ChainSlot {
  ONNXSliceOp slice;
  ONNXReshapeOp reshape;
  ONNXTransposeOp transpose;
  bool needsSwapSdOnOutput = false;
};

struct MatchedPattern {
  ONNXSliceOp slices[3];
  ONNXReshapeOp reshapes[3];
  ONNXTransposeOp transposes[3];
  Value commonInput;
  SmallVector<int64_t> reshapeShape;
  SmallVector<int64_t> inputShape;
  bool needsSwapSdOnOutput[3] = {false, false, false};
};

/// Map xcompiler branch order: matmul {0213,0231,0213}; MHA three {0213}; sort by
/// channel begin (`starts[2]`).
static LogicalResult canonializeThreeChains(SmallVector<ChainSlot, 3> &chains,
    MatchedPattern &pattern, Value sliceInput) {
  if (chains.size() != 3)
    return failure();
  int swapCount = 0;
  for (const auto &c : chains)
    swapCount += c.needsSwapSdOnOutput ? 1 : 0;
  if (swapCount > 1)
    return failure();

  int64_t keys[3];
  for (int i = 0; i < 3; ++i) {
    FailureOr<int64_t> k = getChannelSliceStartKey(chains[static_cast<unsigned>(i)].slice);
    if (failed(k))
      return failure();
    keys[i] = *k;
  }

  unsigned order[3];
  if (swapCount == 0) {
    // Sort chain indices 0..2 by keys (fixed 3-element ordering network).
    unsigned a = 0, b = 1, c = 2;
    if (keys[a] > keys[b])
      std::swap(a, b);
    if (keys[b] > keys[c])
      std::swap(b, c);
    if (keys[a] > keys[b])
      std::swap(a, b);
    order[0] = a;
    order[1] = b;
    order[2] = c;
  } else {
    unsigned mid = 0, side0 = 0, side1 = 0;
    bool seenSide = false;
    for (unsigned i = 0; i < 3; ++i) {
      if (chains[i].needsSwapSdOnOutput)
        mid = i;
      else if (!seenSide) {
        side0 = i;
        seenSide = true;
      } else {
        side1 = i;
      }
    }
    if (keys[side0] <= keys[side1])
      order[0] = side0, order[1] = mid, order[2] = side1;
    else
      order[0] = side1, order[1] = mid, order[2] = side0;
  }

  pattern.commonInput = sliceInput;
  for (int i = 0; i < 3; ++i) {
    const ChainSlot &c = chains[order[i]];
    pattern.slices[i] = c.slice;
    pattern.reshapes[i] = c.reshape;
    pattern.transposes[i] = c.transpose;
    pattern.needsSwapSdOnOutput[i] = c.needsSwapSdOnOutput;
  }

  // Matmul: canonical slot 1 is {0,2,3,1}; MHA: no swap flags.
  return swapCount == static_cast<int>(pattern.needsSwapSdOnOutput[1])
             ? success()
             : failure();
}

/// Try to match the MHA Slice-Reshape-Transpose pattern starting from a
/// transpose
LogicalResult matchPattern(
    ONNXTransposeOp transposeOp, MatchedPattern &pattern) {
  [[maybe_unused]] bool anchorSwapIgnored = false;
  if (!parseHeadSplitTranspose(transposeOp.getPermAttr(), anchorSwapIgnored))
    return failure();

  // Get the reshape op feeding this transpose
  auto reshapeOp = transposeOp.getData().getDefiningOp<ONNXReshapeOp>();
  if (!reshapeOp || !reshapeOp.getResult().hasOneUse()) {
    return failure();
  }

  // Get the slice op feeding the reshape
  auto sliceOp = reshapeOp.getData().getDefiningOp<ONNXSliceOp>();
  if (!sliceOp || !sliceOp.getResult().hasOneUse()) {
    return failure();
  }

  // Get the common input to the slice
  Value sliceInput = sliceOp.getData();

  // Now find all sibling slices that share the same input
  SmallVector<ONNXSliceOp> siblingSlices;
  for (Operation *user : sliceInput.getUsers()) {
    if (auto sibSlice = dyn_cast<ONNXSliceOp>(user)) {
      siblingSlices.push_back(sibSlice);
    }
  }

  // We need exactly 3 slices for the MHA pattern
  if (siblingSlices.size() != 3) {
    return failure();
  }

  SmallVector<ChainSlot, 3> chains;
  for (auto sibSlice : siblingSlices) {
    if (!sibSlice.getResult().hasOneUse())
      continue;

    auto sibReshape =
        dyn_cast<ONNXReshapeOp>(*sibSlice.getResult().getUsers().begin());
    if (!sibReshape || !sibReshape.getResult().hasOneUse())
      continue;

    if (getRank(sibReshape.getData().getType()) != 3)
      continue;

    auto sibTranspose =
        dyn_cast<ONNXTransposeOp>(*sibReshape.getResult().getUsers().begin());
    if (!sibTranspose)
      continue;

    bool branchSwap = false;
    if (!parseHeadSplitTranspose(sibTranspose.getPermAttr(), branchSwap))
      continue;

    ChainSlot slot;
    slot.slice = sibSlice;
    slot.reshape = sibReshape;
    slot.transpose = sibTranspose;
    slot.needsSwapSdOnOutput = branchSwap;
    chains.push_back(slot);
  }

  if (chains.size() != 3)
    return failure();
  if (failed(canonializeThreeChains(chains, pattern, sliceInput)))
    return failure();

  auto inTy = dyn_cast<RankedTensorType>(sliceInput.getType());
  auto out0Ty =
      dyn_cast<RankedTensorType>(pattern.reshapes[0].getResult().getType());
  if (!inTy || !out0Ty || inTy.getRank() != 3 || out0Ty.getRank() != 4)
    return failure();

  pattern.inputShape.assign(inTy.getShape().begin(), inTy.getShape().end());
  pattern.reshapeShape.assign(
      out0Ty.getShape().begin(), out0Ty.getShape().end());

  int64_t headDim = pattern.reshapeShape[3];
  int64_t c = pattern.inputShape[2];
  if (headDim == 0 || c % headDim != 0)
    return failure();
  if ((c / headDim) % 3 != 0)
    return failure();

  ArrayRef<int64_t> rshape = out0Ty.getShape();
  for (int i = 1; i < 3; ++i) {
    auto ri =
        dyn_cast<RankedTensorType>(pattern.reshapes[i].getResult().getType());
    if (!ri || ri.getShape() != rshape)
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pattern: Optimize MHA Slice-Reshape-Transpose Block
//===----------------------------------------------------------------------===//

class OptimizeSliceReshapeTransposeMHAPattern
    : public OpRewritePattern<ONNXTransposeOp> {
public:
  using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTransposeOp transposeOp, PatternRewriter &rewriter) const override {
    // Try to match the pattern
    MatchedPattern pattern;
    if (failed(matchPattern(transposeOp, pattern))) {
      return failure();
    }

    Location loc = transposeOp.getLoc();

    // Get element type from input - preserve it throughout
    Type inputType = pattern.commonInput.getType();
    Type elementType = getElementType(inputType);
    if (!elementType) {
      return failure();
    }

    // Calculate new reshape shape: {N, S, num_heads * 3, head_dim}
    // Original: input {N, S, C} -> slice -> reshape {N, S, num_heads, head_dim}
    // New: input {N, S, C} -> reshape {N, S, C/head_dim, head_dim} -> transpose
    auto &inputShape = pattern.inputShape;
    auto &reshapeShape = pattern.reshapeShape;

    int64_t headDim = reshapeShape[3];
    int64_t fusedHeads = inputShape[2] / headDim;
    SmallVector<int64_t> newReshapeShape = {
        inputShape[0], inputShape[1], fusedHeads, headDim};

    // Set insertion point before the first slice
    rewriter.setInsertionPoint(pattern.slices[0]);

    // Create the new reshape output type preserving element type
    auto newReshapeType = RankedTensorType::get(newReshapeShape, elementType);

    // Create reshape shape constant
    Value newReshapeShapeConst =
        createConstantI64Array(rewriter, loc, newReshapeShape);

    // Create the new reshape op
    auto newReshapeOp = rewriter.create<ONNXReshapeOp>(
        loc, newReshapeType, pattern.commonInput, newReshapeShapeConst);

    // Create new transpose with order {0, 2, 1, 3}
    SmallVector<int64_t> newTransposeShape = {newReshapeShape[0],
        newReshapeShape[2], newReshapeShape[1], newReshapeShape[3]};
    auto newTransposeType =
        RankedTensorType::get(newTransposeShape, elementType);

    auto newTransposeOp =
        rewriter.create<ONNXTransposeOp>(loc, newTransposeType,
            newReshapeOp.getResult(), rewriter.getI64ArrayAttr({0, 2, 1, 3}));

    // Calculate slice parameters for the 3 slices
    // Assuming slices divide the second dimension (num_heads dimension after
    // transpose)
    int64_t numHeadsPerSlice = newTransposeShape[1] / 3;

    static const int64_t kSliceSteps[] = {1, 1, 1, 1};
    static const int64_t kSliceAxes[] = {0, 1, 2, 3};
    Value stepsConst = createConstantI64Array(rewriter, loc, kSliceSteps);
    Value axesConst = createConstantI64Array(rewriter, loc, kSliceAxes);

    const int64_t nDim = newTransposeShape[0];
    const int64_t sDim = newTransposeShape[2];
    const int64_t dDim = newTransposeShape[3];

    SmallVector<int64_t, 4> sliceVals;
    SmallVector<Value> newSliceResults;
    for (int i = 0; i < 3; i++) {
      const int64_t h0 = i * numHeadsPerSlice;
      const int64_t h1 = (i + 1) * numHeadsPerSlice;
      sliceVals.assign({0, h0, 0, 0});
      Value startsConst = createConstantI64Array(rewriter, loc, sliceVals);
      sliceVals.assign({nDim, h1, sDim, dDim});
      Value endsConst = createConstantI64Array(rewriter, loc, sliceVals);

      sliceVals.assign({nDim, numHeadsPerSlice, sDim, dDim});
      auto newSliceType = RankedTensorType::get(sliceVals, elementType);

      // Create new slice op
      auto newSliceOp = rewriter.create<ONNXSliceOp>(loc, newSliceType,
          newTransposeOp.getResult(), startsConst, endsConst, axesConst,
          stepsConst);

      Value repl = newSliceOp.getResult();
      // Branch that used {0,2,3,1} had layout (N,H,D,S); slice above is (N,H,S,D).
      // Swap last two dims to match original consumers.
      if (pattern.needsSwapSdOnOutput[i]) {
        sliceVals.assign({nDim, numHeadsPerSlice, dDim, sDim});
        auto swappedType = RankedTensorType::get(sliceVals, elementType);
        repl = rewriter.create<ONNXTransposeOp>(
            loc, swappedType, repl, rewriter.getI64ArrayAttr({0, 1, 3, 2}));
      }

      newSliceResults.push_back(repl);
    }

    // Replace uses and erase old ops
    for (int i = 0; i < 3; i++) {
      rewriter.replaceOp(pattern.transposes[i], newSliceResults[i]);
    }

    // Erase intermediate ops (reshape and slice) - they should now be unused
    for (int i = 0; i < 3; i++) {
      if (pattern.reshapes[i]->use_empty())
        rewriter.eraseOp(pattern.reshapes[i]);
      if (pattern.slices[i]->use_empty())
        rewriter.eraseOp(pattern.slices[i]);
    }

    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to optimize MHA Slice-Reshape-Transpose blocks
struct OptimizeSliceReshapeTransposeBlockPass
    : public PassWrapper<OptimizeSliceReshapeTransposeBlockPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "optimize-slice-reshape-transpose-block";
  }
  StringRef getDescription() const override {
    return "Optimize MHA Slice-Reshape-Transpose blocks by moving reshape and "
           "transpose before slices";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<OptimizeSliceReshapeTransposeMHAPattern>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createOptimizeSliceReshapeTransposeBlockPass() {
  return std::make_unique<OptimizeSliceReshapeTransposeBlockPass>();
}

} // namespace onnx_mlir
