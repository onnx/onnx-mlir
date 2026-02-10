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
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
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
/// Returns -1 if any dimension is dynamic (< 0)
int64_t mulWithoutDims(
    ArrayRef<int64_t> shape, const std::set<int64_t> &excludeDims) {
  int64_t result = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (excludeDims.find(i) == excludeDims.end()) {
      if (shape[i] < 0) {
        LLVM_DEBUG(llvm::dbgs() << "Dynamic dimension at index " << i
                                << ", cannot compute static product\n");
        return -1; // Signal error - dynamic dimension encountered
      }
      result *= shape[i];
    }
  }
  return result;
}

/// Simple integer decomposition into prime factors
SmallVector<int64_t> integerDecomposition(int64_t n) {
  if (n <= 0) {
    LLVM_DEBUG(llvm::dbgs() << "ERROR: Invalid input to integerDecomposition: "
                            << n << " (must be > 0)\n");
    return {1};
  }

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

/// Check if all tensor inputs are already 4D with batch=1 (optional/none inputs
/// ignored)
bool allInputsAlready4D(ArrayRef<Value> inputs) {
  return std::all_of(inputs.begin(), inputs.end(), [](Value input) {
    if (isa<NoneType>(input.getType()))
      return true;
    auto shape = getShapeFromType(input.getType());
    return shape.size() == 4 && shape.front() == 1;
  });
}

/// Detect which dimensions differ between inputs and output (broadcast dims)
/// Returns empty set and sets hasError=true if rank mismatch detected
std::set<int64_t> detectBroadcastDims(
    ArrayRef<Value> inputs, ArrayRef<int64_t> outputShape, bool &hasError) {
  std::set<int64_t> broadcastDims;
  hasError = false;

  for (size_t inputIdx = 0; inputIdx < inputs.size(); ++inputIdx) {
    Value input = inputs[inputIdx];
    auto inputShape = getShapeFromType(input.getType());

    // Skip scalars (rank-0 tensors) or unknown shapes - they don't contribute
    // to broadcast detection
    if (inputShape.empty())
      continue;

    // Skip scalar-like tensors (e.g., [1]) - treat them as broadcastable
    // scalars
    if (inputShape.size() == 1 && inputShape[0] == 1)
      continue;

    // Rank mismatch is an error - cannot reliably detect broadcast dims
    if (inputShape.size() != outputShape.size()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Rank mismatch: Input " << inputIdx << " rank ("
                 << inputShape.size() << ") != output rank ("
                 << outputShape.size() << "), cannot handle\n");
      hasError = true;
      return {};
    }

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
/// Returns empty vector on error (dynamic dimensions or size mismatch)
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
    if (remainder < 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot compute 4D shape: dynamic dimensions present\n");
      return {}; // Return empty to signal error
    }
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
      if (W < 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Cannot compute 4D shape: dynamic dimensions present\n");
        return {}; // Return empty to signal error
      }
      C = shape.back();
    } else {
      // Broadcast at middle dimension
      for (int64_t idx = 0; idx < shapeSize; ++idx) {
        if (shape[idx] < 0) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Cannot compute 4D shape: dynamic dimension at index "
                     << idx << "\n");
          return {}; // Return empty to signal error
        }
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

  // Verify product matches original
  int64_t original_size = 1;
  for (auto dim : shape) {
    original_size *= dim;
  }
  int64_t new_size = N * H * W * C;
  if (original_size != new_size) {
    LLVM_DEBUG(llvm::dbgs() << "Size mismatch! Original=" << original_size
                            << ", New=" << new_size
                            << " - transformation would be incorrect\n");
    return {}; // Signal error - don't create invalid IR
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
    LLVM_DEBUG(llvm::dbgs() << "Trying to match " << op->getName() << "\n");

    auto outputShape = getShapeFromType(op.getResult().getType());
    Type elementType = getElementType(op.getResult().getType());

    if (outputShape.empty() || !elementType) {
      return failure();
    }

    // Skip if output is already 4D with batch=1
    if (outputShape.size() == 4 && outputShape.front() == 1) {
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
    bool hasError = false;
    std::set<int64_t> broadcastDims =
        detectBroadcastDims(inputs, outputShape, hasError);
    if (hasError) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "SKIP: Cannot detect broadcast dims due to rank mismatch\n");
      return failure();
    }

    // Guard: current logic only supports at most one broadcast dimension.
    if (broadcastDims.size() > 1) {
      LLVM_DEBUG(llvm::dbgs() << "SKIP: More than 1 broadcast dimension ("
                              << broadcastDims.size() << ")\n");
      return failure();
    }

    Location loc = op.getLoc();

    // Compute output 4D shape using broadcast-aware logic
    SmallVector<int64_t> outputShape4D =
        computeBroadcastAware4DShape(outputShape, broadcastDims);
    if (outputShape4D.empty()) {
      return failure();
    }

    // Reshape all inputs to their respective 4D shapes
    SmallVector<Value> reshapedInputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
      Value input = inputs[i];
      // Optional inputs (e.g. type none) pass through as-is.
      if (isa<NoneType>(input.getType())) {
        reshapedInputs.push_back(input);
        continue;
      }
      auto inputShape = getShapeFromType(input.getType());
      Type inputElementType = getElementType(input.getType());

      // If input shape differs from output shape (e.g. scalar for broadcast),
      // pass through as-is.
      if (inputShape.size() != outputShape.size() ||
          !std::equal(
              inputShape.begin(), inputShape.end(), outputShape.begin())) {
        reshapedInputs.push_back(input);
        continue;
      }

      if (!inputElementType) {
        return failure();
      }

      SmallVector<int64_t> inputShape4D =
          computeBroadcastAware4DShape(inputShape, broadcastDims);

      if (inputShape4D.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "SKIP: Cannot compute 4D shape for input " << i << "\n");
        return failure();
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
    LLVM_DEBUG(
        llvm::dbgs() << "SUCCESS: Transformed " << op->getName() << " to 4D\n");
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

std::unique_ptr<mlir::Pass> createTransferOpShapeTo4dPass() {
  return std::make_unique<TransferOpShapeTo4dPass>();
}

} // namespace onnx_mlir
