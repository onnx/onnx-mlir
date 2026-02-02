// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass eliminates reshape operations around slice operations when the
// reshapes only shuffle singular dimensions.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <vector>

#define DEBUG_TYPE "eliminate-reshape-around-slice"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static std::vector<int64_t> getShape(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return std::vector<int64_t>(
        tensorType.getShape().begin(), tensorType.getShape().end());
  return {};
}

static std::optional<size_t> findNonOnePosition(ArrayRef<int64_t> shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != 1)
      return i;
  }
  return std::nullopt;
}

static std::vector<int64_t> shuffleVec(
    const std::vector<int64_t> &in, size_t inPos, size_t outPos) {
  auto ret = in;
  std::swap(ret[inPos], ret[outPos]);
  return ret;
}

static std::vector<int64_t> extractI64Values(Value val) {
  std::vector<int64_t> result;
  if (auto constOp = val.getDefiningOp<ONNXConstantOp>()) {
    if (auto attr = constOp.getValueAttr()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
        for (auto v : denseAttr.getValues<APInt>()) {
          result.push_back(v.getSExtValue());
        }
      }
    }
  }
  return result;
}

static DenseElementsAttr createDenseI64Attr(
    MLIRContext *ctx, const std::vector<int64_t> &values) {
  auto tensorType = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, IntegerType::get(ctx, 64));
  return DenseElementsAttr::get(tensorType, llvm::ArrayRef(values));
}

static Value createI64ConstantOp(PatternRewriter &rewriter, Location loc,
    const std::vector<int64_t> &values) {
  MLIRContext *ctx = rewriter.getContext();
  auto attr = createDenseI64Attr(ctx, values);
  return rewriter
      .create<ONNXConstantOp>(loc, attr.getType(), Attribute(), attr,
          FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(),
          ArrayAttr())
      .getResult();
}

//===----------------------------------------------------------------------===//
// Branch Abstraction
//===----------------------------------------------------------------------===//

/// Represents a branch: topReshape -> slice -> [chain of ops] -> bottomReshape
struct Branch {
  ONNXReshapeOp topReshape;
  ONNXSliceOp slice;
  ONNXReshapeOp bottomReshape;
  SmallVector<Operation *> chainOps; // Ops between slice and reshape (eltwise)
};

/// Check if an operation is a supported eltwise op
static bool isSupportedEltwiseOp(Operation *op) {
  return isa<ONNXAddOp, ONNXMulOp, ONNXSubOp>(op);
}

/// Trace back from a bottom reshape to find the complete branch
/// Returns std::nullopt if pattern doesn't match
static std::optional<Branch> traceBackToTopReshape(
    ONNXReshapeOp bottomReshape) {
  Branch branch;
  branch.bottomReshape = bottomReshape;

  Value current = bottomReshape.getData();

  // Walk backward through the chain
  while (true) {
    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      return std::nullopt;

    // Check if we've reached a slice
    if (auto sliceOp = dyn_cast<ONNXSliceOp>(defOp)) {
      branch.slice = sliceOp;

      // Slice's input should be a reshape (the top reshape)
      auto topReshape = sliceOp.getData().getDefiningOp<ONNXReshapeOp>();
      if (!topReshape)
        return std::nullopt;

      branch.topReshape = topReshape;
      return branch;
    }

    // Check if it's a supported eltwise op
    if (isSupportedEltwiseOp(defOp)) {
      if (!defOp->hasOneUse())
        return std::nullopt;

      branch.chainOps.push_back(defOp);

      // Find the non-constant input to continue tracing
      Value nextInput = nullptr;
      for (auto operand : defOp->getOperands()) {
        if (!operand.getDefiningOp<ONNXConstantOp>()) {
          if (nextInput && nextInput != operand) {
            // Multiple non-constant inputs - take single-use one
            if (operand.hasOneUse()) {
              nextInput = operand;
            }
          } else {
            nextInput = operand;
          }
        }
      }

      if (!nextInput)
        return std::nullopt;

      current = nextInput;
    } else {
      // Unsupported op in chain
      return std::nullopt;
    }
  }
}

/// Recreate an eltwise op with new operands and output type
static Value recreateEltwiseOp(Operation *origOp, ValueRange newOperands,
    Type newOutputType, PatternRewriter &rewriter) {
  Location loc = origOp->getLoc();

  if (isa<ONNXAddOp>(origOp)) {
    return rewriter
        .create<ONNXAddOp>(loc, newOutputType, newOperands[0], newOperands[1])
        .getResult();
  }
  if (isa<ONNXMulOp>(origOp)) {
    return rewriter
        .create<ONNXMulOp>(loc, newOutputType, newOperands[0], newOperands[1])
        .getResult();
  }
  if (isa<ONNXSubOp>(origOp)) {
    return rewriter
        .create<ONNXSubOp>(loc, newOutputType, newOperands[0], newOperands[1])
        .getResult();
  }

  llvm_unreachable("Unsupported eltwise op");
}

/// Reshape a constant's data to a new shape, preserving element type
static Value createReshapedConstant(ONNXConstantOp origConst, size_t inPos,
    size_t outPos, PatternRewriter &rewriter) {
  auto constAttr = origConst.getValueAttr();
  auto constDenseAttr = dyn_cast<DenseElementsAttr>(constAttr);
  if (!constDenseAttr)
    return nullptr;

  auto origShape = getShape(origConst.getResult().getType());
  auto shuffledShape = shuffleVec(origShape, inPos, outPos);

  // Use the DenseElementsAttr's element type for reshaping (storage type)
  auto denseAttrType =
      RankedTensorType::get(shuffledShape, constDenseAttr.getElementType());
  auto newConstAttr = constDenseAttr.reshape(denseAttrType);

  // Use the result type's element type for the op (may be quantized)
  auto newResultType = RankedTensorType::get(shuffledShape,
      cast<RankedTensorType>(origConst.getResult().getType()).getElementType());

  return rewriter
      .create<ONNXConstantOp>(origConst.getLoc(), newResultType, Attribute(),
          newConstAttr, FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(),
          StringAttr(), ArrayAttr())
      .getResult();
}

/// Transform a branch: create new slice and recreate chain ops
/// Preserves element types through the transformation
static Value transformBranch(Branch &branch, Value originalInput,
    const std::vector<int64_t> &reshapeInShape, size_t inPos, size_t outPos,
    PatternRewriter &rewriter) {
  Location loc = branch.slice.getLoc();

  // Extract and shuffle slice parameters
  auto starts = extractI64Values(branch.slice.getStarts());
  auto ends = extractI64Values(branch.slice.getEnds());
  auto newStarts = shuffleVec(starts, inPos, outPos);
  auto newEnds = shuffleVec(ends, inPos, outPos);

  // Compute new slice output shape
  std::vector<int64_t> newSliceOutputShape = reshapeInShape;
  for (size_t i = 0; i < newStarts.size() && i < reshapeInShape.size(); ++i) {
    newSliceOutputShape[i] = newEnds[i] - newStarts[i];
  }

  // Create axes and steps
  std::vector<int64_t> axes(reshapeInShape.size());
  std::iota(axes.begin(), axes.end(), 0);
  std::vector<int64_t> steps(reshapeInShape.size(), 1);

  // Create new slice - preserve element type from original input
  auto newSliceType = RankedTensorType::get(newSliceOutputShape,
      cast<RankedTensorType>(originalInput.getType()).getElementType());
  auto newSlice = rewriter.create<ONNXSliceOp>(loc, newSliceType, originalInput,
      createI64ConstantOp(rewriter, loc, newStarts),
      createI64ConstantOp(rewriter, loc, newEnds),
      createI64ConstantOp(rewriter, loc, axes),
      createI64ConstantOp(rewriter, loc, steps));

  Value currentValue = newSlice.getResult();

  // Recreate chain ops (in reverse order since we collected them bottom-to-top)
  for (Operation *origOp : llvm::reverse(branch.chainOps)) {
    // Build new operands
    SmallVector<Value> newOperands;
    for (auto operand : origOp->getOperands()) {
      if (auto constOp = operand.getDefiningOp<ONNXConstantOp>()) {
        // Reshape the constant, preserving its element type
        Value newConst =
            createReshapedConstant(constOp, inPos, outPos, rewriter);
        if (!newConst)
          return nullptr;
        newOperands.push_back(newConst);
      } else {
        // Use the current chain value
        newOperands.push_back(currentValue);
      }
    }

    // Use the original op's output type with shuffled shape (preserves element
    // type)
    auto origOutputType =
        cast<RankedTensorType>(origOp->getResult(0).getType());
    auto origOutputShape = std::vector<int64_t>(
        origOutputType.getShape().begin(), origOutputType.getShape().end());
    auto shuffledOutputShape = shuffleVec(origOutputShape, inPos, outPos);
    auto newOutputType = RankedTensorType::get(
        shuffledOutputShape, origOutputType.getElementType());

    currentValue =
        recreateEltwiseOp(origOp, newOperands, newOutputType, rewriter);
  }

  return currentValue;
}

//===----------------------------------------------------------------------===//
// Pattern: Eliminate Reshape Around Slice
//===----------------------------------------------------------------------===//

class EliminateReshapeAroundSlicePattern
    : public OpRewritePattern<ONNXReshapeOp> {
public:
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReshapeOp bottomReshape, PatternRewriter &rewriter) const override {
    // Step 1: Trace back from this reshape to find the complete branch
    auto maybeBranch = traceBackToTopReshape(bottomReshape);
    if (!maybeBranch)
      return failure();

    Branch branch = *maybeBranch;

    // Step 2: Validate shape constraints on the top reshape
    auto reshapeInShape = getShape(branch.topReshape.getData().getType());
    auto reshapeOutShape = getShape(branch.topReshape.getResult().getType());

    if (reshapeInShape.size() != 4 || reshapeOutShape.size() != 4)
      return failure();

    // Should have exactly 3 ones (only one non-one dimension)
    auto countOnes = [](const std::vector<int64_t> &shape) {
      return std::count(shape.begin(), shape.end(), 1);
    };
    if (countOnes(reshapeInShape) != 3 || countOnes(reshapeOutShape) != 3)
      return failure();

    auto maybeInPos = findNonOnePosition(reshapeInShape);
    auto maybeOutPos = findNonOnePosition(reshapeOutShape);
    if (!maybeInPos || !maybeOutPos)
      return failure();

    size_t inPos = *maybeInPos;
    size_t outPos = *maybeOutPos;

    // Step 3: Transform this branch
    Value originalInput = branch.topReshape.getData();
    Value newOutput = transformBranch(
        branch, originalInput, reshapeInShape, inPos, outPos, rewriter);

    if (!newOutput)
      return failure();

    // Step 4: Replace bottom reshape with new output
    rewriter.replaceOp(bottomReshape, newOutput);

    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to eliminate reshape operations around slice operations
struct EliminateReshapeAroundSlicePass
    : public PassWrapper<EliminateReshapeAroundSlicePass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "eliminate-reshape-around-slice";
  }
  StringRef getDescription() const override {
    return "Eliminate reshape operations around slice operations when the "
           "reshapes only shuffle singular dimensions";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<EliminateReshapeAroundSlicePattern>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createEliminateReshapeAroundSlicePass() {
  return std::make_unique<EliminateReshapeAroundSlicePass>();
}

} // namespace onnx_mlir
