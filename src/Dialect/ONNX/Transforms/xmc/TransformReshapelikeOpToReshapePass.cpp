// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass converts reshape-like operations (Flatten, Squeeze, Unsqueeze,
// trivial Transpose) to Reshape operations for canonicalization.

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
#include "llvm/Support/raw_ostream.h"

#include <queue>
#include <type_traits>

#define DEBUG_TYPE "transform-reshapelike-to-reshape"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Extract shape dimensions from a ranked tensor type
SmallVector<int64_t> getShapeFromTensorType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return {};

  auto shape = tensorType.getShape();
  return SmallVector<int64_t>(shape.begin(), shape.end());
}

/// Creates a DenseElementsAttr constant from a vector of int64 values
DenseElementsAttr createDenseElementsAttr(MLIRContext *context,
    ArrayRef<int64_t> values) {
  auto tensorType = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, IntegerType::get(context, 64));
  return DenseElementsAttr::get(tensorType, values);
}

/// Check if the operation is a constant operation
bool isConstantOp(Operation *op) {
  if (!op)
    return false;
  return isa<ONNXConstantOp>(op);
}

/// Trace back through single-fanout operations to find if chain leads to
/// constant
bool hasConstantInputChain(Value input) {
  Operation *defOp = input.getDefiningOp();
  while (defOp) {
    if (isConstantOp(defOp)) {
      return true;
    }

    // Check if single input and single fanout
    if (defOp->getNumOperands() != 1) {
      break;
    }

    auto inputValue = defOp->getOperand(0);
    if (!inputValue.hasOneUse()) {
      break;
    }

    defOp = inputValue.getDefiningOp();
  }

  return false;
}

/// Check if transpose is trivial (only reshapes, doesn't reorder non-singular
/// dimensions)
bool isTrivialTranspose(ArrayRef<int64_t> inputShape,
    ArrayRef<int64_t> outputShape, ArrayAttr permAttr) {
  if (!permAttr)
    return false;

  SmallVector<int64_t> order;
  for (auto attr : permAttr) {
    order.push_back(cast<IntegerAttr>(attr).getInt());
  }

  // Build queue of non-singular dimension indices from input
  std::queue<int64_t> q;
  for (size_t i = 0; i < inputShape.size(); ++i) {
    if (inputShape[i] > 1) {
      q.push(i);
    }
  }

  // Check if transpose preserves relative order of non-singular dimensions
  for (size_t i = 0; i < outputShape.size(); ++i) {
    if (outputShape[i] > 1) {
      if (q.empty() || order[i] != q.front()) {
        return false;
      }
      q.pop();
    }
  }

  return true;
}

/// Helper to get input value - Flatten uses getInput(), others use getData()
template <typename OpTy>
static Value getOpInputValue(OpTy op) {
  if constexpr (std::is_same_v<OpTy, ONNXFlattenOp>) {
    return op.getInput();
  } else {
    return op.getData();
  }
}

/// Helper to replace an operation with a reshape
static void replaceWithReshape(Operation *op, Value input, Type outputType,
    ArrayRef<int64_t> outputShape, PatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op->getLoc();

  auto shapeAttr = createDenseElementsAttr(ctx, outputShape);
  auto shapeType = RankedTensorType::get(
      {static_cast<int64_t>(outputShape.size())}, rewriter.getI64Type());

  auto shapeConstantOp = rewriter.create<ONNXConstantOp>(loc, shapeType,
      Attribute(), shapeAttr, FloatAttr(), ArrayAttr(), IntegerAttr(),
      ArrayAttr(), StringAttr(), ArrayAttr());

  auto reshapeOp = rewriter.create<ONNXReshapeOp>(
      loc, outputType, input, shapeConstantOp.getResult());

  rewriter.replaceOp(op, reshapeOp.getResult());
}

//===----------------------------------------------------------------------===//
// Pattern: Trivial Transpose → Reshape
//===----------------------------------------------------------------------===//

/// Transform trivial Transpose operations (that don't reorder data) to Reshape
struct TransformTrivialTransposeToReshape
    : public OpRewritePattern<ONNXTransposeOp> {
  using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXTransposeOp transposeOp,
      PatternRewriter &rewriter) const override {
    auto inputType = transposeOp.getData().getType();
    auto outputType = transposeOp.getResult().getType();

    auto inputTensorType = dyn_cast<RankedTensorType>(inputType);
    auto outputTensorType = dyn_cast<RankedTensorType>(outputType);

    if (!inputTensorType || !outputTensorType) {
      return failure();
    }

    // Get shapes
    auto inputShape = getShapeFromTensorType(inputType);
    auto outputShape = getShapeFromTensorType(outputType);
    auto permAttr = transposeOp.getPermAttr();

    // Check if this is a trivial transpose (doesn't actually reorder data)
    if (!isTrivialTranspose(inputShape, outputShape, permAttr)) {
      return failure();
    }

    // Check if input comes from constant chain - skip transformation for
    // constant folding
    if (hasConstantInputChain(transposeOp.getData())) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Converting trivial Transpose to Reshape\n");

    replaceWithReshape(
        transposeOp, transposeOp.getData(), outputType, outputShape, rewriter);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Reshape-like ops (Flatten, Squeeze, Unsqueeze) → Reshape
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct TransformReshapelikeToReshape : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  static constexpr const char *getOpName() {
    if constexpr (std::is_same_v<OpTy, ONNXFlattenOp>)
      return "Flatten";
    else if constexpr (std::is_same_v<OpTy, ONNXSqueezeOp>)
      return "Squeeze";
    else
      return "Unsqueeze";
  }

  LogicalResult matchAndRewrite(OpTy op,
      PatternRewriter &rewriter) const override {
    Value inputValue = getOpInputValue(op);
    auto inputType = inputValue.getType();
    auto outputType = op.getResult().getType();

    auto inputTensorType = dyn_cast<RankedTensorType>(inputType);
    auto outputTensorType = dyn_cast<RankedTensorType>(outputType);

    if (!inputTensorType || !outputTensorType) {
      return failure();
    }

    auto outputShape = getShapeFromTensorType(outputType);

    LLVM_DEBUG(llvm::dbgs() << "Converting " << getOpName() << " to Reshape\n");

    replaceWithReshape(op, inputValue, outputType, outputShape, rewriter);
    return success();
  }
};

// Type aliases for clarity
using TransformFlattenToReshape = TransformReshapelikeToReshape<ONNXFlattenOp>;
using TransformSqueezeToReshape = TransformReshapelikeToReshape<ONNXSqueezeOp>;
using TransformUnsqueezeToReshape =
    TransformReshapelikeToReshape<ONNXUnsqueezeOp>;

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to transform reshape-like operations to Reshape.
/// This converts Flatten, Squeeze, Unsqueeze, and trivial Transpose operations
/// to canonical Reshape operations.
struct TransformReshapelikeOpToReshapePass
    : public PassWrapper<TransformReshapelikeOpToReshapePass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transform-reshapelike-op-to-reshape";
  }
  StringRef getDescription() const override {
    return "Convert reshape-like operations (Flatten, Squeeze, Unsqueeze, "
           "trivial Transpose) to Reshape";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // Add all patterns
    patterns.add<TransformTrivialTransposeToReshape>(ctx);
    patterns.add<TransformFlattenToReshape>(ctx);
    patterns.add<TransformSqueezeToReshape>(ctx);
    patterns.add<TransformUnsqueezeToReshape>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createTransformReshapelikeOpToReshapePass() {
  return std::make_unique<TransformReshapelikeOpToReshapePass>();
}

} // namespace onnx_mlir

