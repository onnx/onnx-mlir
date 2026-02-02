// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass transfers element-wise operations with non-4D shapes to 4D.
// Handles broadcasting by computing per-input 4D shapes based on which
// dimensions differ between inputs and output.

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
#include <cmath>
#include <set>

#define DEBUG_TYPE "transfer-op-shape-to-4d"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Helper to get shape from a ranked tensor type
SmallVector<int64_t> getShapeFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    auto shape = tensorType.getShape();
    return SmallVector<int64_t>(shape.begin(), shape.end());
  }
  return {};
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

/// Multiply all dimensions except those specified in excludeDims
int64_t mulWithoutDims(
    ArrayRef<int64_t> shape, const std::set<int64_t> &excludeDims) {
  int64_t result = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (excludeDims.find(i) == excludeDims.end()) {
      result *= shape[i];
    }
  }
  return result;
}

/// Simple integer decomposition into prime factors
SmallVector<int64_t> integerDecomposition(int64_t n) {
  SmallVector<int64_t> factors;
  for (int64_t d = 2; d * d <= n; ++d) {
    while (n % d == 0) {
      factors.push_back(d);
      n /= d;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

/// Check if all inputs are already 4D with batch=1
bool allInputsAlready4D(ArrayRef<Value> inputs) {
  return std::all_of(inputs.begin(), inputs.end(), [](Value input) {
    auto shape = getShapeFromType(input.getType());
    return shape.size() == 4 && shape.front() == 1;
  });
}

/// Detect which dimensions differ between inputs and output (broadcast dims)
/// Matches original: assumes same rank, uses input's shape.size() for loop
std::set<int64_t> detectBroadcastDims(
    ArrayRef<Value> inputs, ArrayRef<int64_t> outputShape) {
  std::set<int64_t> broadcastDims;

  for (Value input : inputs) {
    auto inputShape = getShapeFromType(input.getType());
    if (inputShape.empty())
      continue;

    // Original assumes same size - use input's size for loop bound
    for (size_t idx = 0; idx < inputShape.size(); ++idx) {
      if (inputShape[idx] != outputShape[idx]) {
        broadcastDims.insert(idx);
      }
    }
  }

  return broadcastDims;
}

/// Compute broadcast-aware 4D shape for an input tensor
/// Takes into account which dimensions are broadcast dimensions
/// Returns empty vector if input should be kept unchanged
SmallVector<int64_t> computeBroadcastAware4DShape(
    ArrayRef<int64_t> inputShape, const std::set<int64_t> &broadcastDims) {
  SmallVector<int64_t> shape(inputShape.begin(), inputShape.end());

  // For < 4D: pad with 1s at the front
  if (shape.size() < 4) {
    shape.insert(shape.begin(), 4 - shape.size(), 1);
    return shape;
  }

  // For >= 4D: compute N, H, W, C based on broadcast dimensions
  int64_t N = 1, H = 1, W = 1, C = 1;

  int64_t shapeSize = static_cast<int64_t>(shape.size());

  if (broadcastDims.empty() ||
      (broadcastDims.size() == 1 && *broadcastDims.begin() == shapeSize - 1)) {
    // No broadcast or broadcast only at last dim
    // Use sqrt-based H/W decomposition
    C = shape.back();
    int64_t remainder =
        mulWithoutDims(shape, {static_cast<int64_t>(shapeSize - 1)});
    double hCandidate = std::ceil(std::sqrt(static_cast<double>(remainder)));
    auto factors = integerDecomposition(remainder);
    std::sort(factors.begin(), factors.end());

    for (int64_t item : factors) {
      if (H < hCandidate) {
        H *= item;
      } else {
        W *= item;
      }
    }
  } else {
    // Broadcast at a specific dimension
    int64_t curDim = *broadcastDims.begin();

    if (curDim == 0) {
      // Broadcast at dim 0
      H = shape[0];
      W = mulWithoutDims(shape, {0, shapeSize - 1});
      C = shape.back();
    } else {
      // Broadcast at middle dimension
      for (int64_t idx = 0; idx < shapeSize; ++idx) {
        if (idx < curDim) {
          H *= shape[idx];
        } else if (idx == curDim) {
          W = shape[idx];
        } else {
          C *= shape[idx];
        }
      }
    }
  }

  return {N, H, W, C};
}

//===----------------------------------------------------------------------===//
// Pattern: Transfer element-wise ops with non-4D shapes to 4D
//===----------------------------------------------------------------------===//

template <typename OpTy>
class TransferEltwiseTo4DPattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OpTy op, PatternRewriter &rewriter) const override {
    auto outputShape = getShapeFromType(op.getResult().getType());
    Type elementType = getElementType(op.getResult().getType());

    if (outputShape.empty() || !elementType) {
      return failure();
    }

    // Gather all inputs
    SmallVector<Value> inputs(
        op->getOperands().begin(), op->getOperands().end());

    // Skip if all inputs are already 4D with batch=1 (matches original)
    if (allInputsAlready4D(inputs)) {
      return failure();
    }

    // Detect broadcast dimensions
    std::set<int64_t> broadcastDims = detectBroadcastDims(inputs, outputShape);

    // Guard: current logic only supports at most one broadcast dimension.
    if (broadcastDims.size() > 1) {
      return failure();
    }

    Location loc = op.getLoc();

    // Compute output 4D shape using broadcast-aware logic
    SmallVector<int64_t> outputShape4D =
        computeBroadcastAware4DShape(outputShape, broadcastDims);
    if (outputShape4D.empty()) {
      // Cannot handle this broadcast pattern (>1 broadcast dims)
      return failure();
    }

    // If output shape is unchanged, skip
    if (outputShape4D.size() == outputShape.size() &&
        std::equal(
            outputShape4D.begin(), outputShape4D.end(), outputShape.begin())) {
      return failure();
    }

    // Reshape all inputs to their respective 4D shapes
    SmallVector<Value> reshapedInputs;
    for (Value input : inputs) {
      auto inputShape = getShapeFromType(input.getType());
      Type inputElementType = getElementType(input.getType());

      if (inputShape.empty() || !inputElementType) {
        return failure();
      }

      SmallVector<int64_t> inputShape4D =
          computeBroadcastAware4DShape(inputShape, broadcastDims);

      if (inputShape4D.empty()) {
        // Cannot reshape this input (>1 broadcast dims) - keep unchanged
        reshapedInputs.push_back(input);
        continue;
      }

      // Check if reshape is actually needed
      if (inputShape4D.size() == inputShape.size() &&
          std::equal(
              inputShape4D.begin(), inputShape4D.end(), inputShape.begin())) {
        reshapedInputs.push_back(input);
        continue;
      }

      auto inputReshapeType =
          RankedTensorType::get(inputShape4D, inputElementType);
      Value shapeConst = createConstantI64Array(rewriter, loc, inputShape4D);
      auto reshapeOp = rewriter.create<ONNXReshapeOp>(
          loc, inputReshapeType, input, shapeConst);
      reshapedInputs.push_back(reshapeOp.getResult());
    }

    // Create new element-wise op with 4D inputs
    auto newOutputType = RankedTensorType::get(outputShape4D, elementType);
    Operation *newOp =
        rewriter.create<OpTy>(loc, newOutputType, reshapedInputs);

    // Reshape output back to original shape
    auto outputReshapeType = RankedTensorType::get(outputShape, elementType);
    Value outputShapeConst = createConstantI64Array(rewriter, loc, outputShape);
    auto outputReshapeOp = rewriter.create<ONNXReshapeOp>(
        loc, outputReshapeType, newOp->getResult(0), outputShapeConst);

    rewriter.replaceOp(op, outputReshapeOp.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to transfer element-wise ops with non-4D shapes to 4D
struct TransferOpShapeTo4dPass
    : public PassWrapper<TransferOpShapeTo4dPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "transfer-op-shape-to-4d"; }
  StringRef getDescription() const override {
    return "Transfer element-wise operations with non-4D shapes to 4D";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // Add element-wise patterns for various ops
    patterns.add<TransferEltwiseTo4DPattern<ONNXAddOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXMulOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXSubOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXDivOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXMaxOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXMinOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXAbsOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXExpOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXSqrtOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXReluOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXTanhOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXLeakyReluOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXEluOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXNegOp>>(ctx);
    patterns.add<TransferEltwiseTo4DPattern<ONNXClipOp>>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createTransferOpShapeTo4dPass() {
  return std::make_unique<TransferOpShapeTo4dPass>();
}

} // namespace onnx_mlir
