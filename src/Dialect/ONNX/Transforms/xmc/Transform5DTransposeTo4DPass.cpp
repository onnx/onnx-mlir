// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass transforms 5D Transpose operations to Reshape + 4D Transpose + Reshape
// when the perm contains a consecutive pair, enabling more efficient execution.

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

#include <optional>

#define DEBUG_TYPE "transform-5d-transpose-to-4d"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Find first consecutive pair in perm where perm[i] + 1 == perm[i+1]
/// Returns the position i, or std::nullopt if not found
std::optional<size_t> findConsecutivePair(ArrayRef<int64_t> perm) {
  for (size_t i = 0; i + 1 < perm.size(); ++i) {
    if (perm[i] + 1 == perm[i + 1]) {
      return i;
    }
  }
  return std::nullopt;
}

/// Compute the 4D perm from 5D perm by merging dimensions at merge_pos
/// merge_dim is the value perm[merge_pos] (the first of the consecutive pair)
SmallVector<int64_t, 4> compute4DPerm(ArrayRef<int64_t> perm5D, size_t mergePos,
    int64_t mergeDim) {
  SmallVector<int64_t, 4> perm4D;
  for (size_t i = 0; i < perm5D.size(); ++i) {
    // Skip the second position of the consecutive pair
    if (i == mergePos + 1)
      continue;

    int64_t value = perm5D[i];
    // Adjust values: dimensions > mergeDim+1 get decremented
    if (value > mergeDim + 1) {
      perm4D.push_back(value - 1);
    } else if (value == mergeDim + 1) {
      // This shouldn't happen at positions other than mergePos+1 (which we
      // skip) But if it does, treat it as mergeDim
      perm4D.push_back(mergeDim);
    } else {
      perm4D.push_back(value);
    }
  }
  return perm4D;
}

/// Compute 4D shape by merging dimensions mergeDim and mergeDim+1
SmallVector<int64_t, 4> compute4DShape(ArrayRef<int64_t> shape5D,
    int64_t mergeDim) {
  SmallVector<int64_t, 4> shape4D;
  for (size_t i = 0; i < shape5D.size(); ++i) {
    if (static_cast<int64_t>(i) == mergeDim) {
      // Check for dynamic dimensions
      if (ShapedType::isDynamic(shape5D[i]) ||
          ShapedType::isDynamic(shape5D[i + 1])) {
        return {}; // Cannot merge dynamic dimensions
      }
      shape4D.push_back(shape5D[i] * shape5D[i + 1]);
    } else if (static_cast<int64_t>(i) == mergeDim + 1) {
      // Skip - already merged with previous
      continue;
    } else {
      shape4D.push_back(shape5D[i]);
    }
  }
  return shape4D;
}

/// Apply permutation to shape
SmallVector<int64_t, 4> applyPerm(ArrayRef<int64_t> shape,
    ArrayRef<int64_t> perm) {
  SmallVector<int64_t, 4> result;
  for (auto p : perm) {
    result.push_back(shape[p]);
  }
  return result;
}

/// Creates a DenseElementsAttr constant from shape values
DenseElementsAttr getShapeAttr(MLIRContext *ctx, ArrayRef<int64_t> shape) {
  auto tensorType = RankedTensorType::get({static_cast<int64_t>(shape.size())},
      IntegerType::get(ctx, 64));
  return DenseElementsAttr::get(tensorType, shape);
}

/// Converts a vector of integers into an MLIR ArrayAttr
ArrayAttr getI64ArrayAttr(MLIRContext *ctx, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  for (auto val : values)
    attrs.push_back(IntegerAttr::get(IntegerType::get(ctx, 64), val));
  return ArrayAttr::get(ctx, attrs);
}

//===----------------------------------------------------------------------===//
// Pattern: Transform 5D Transpose to Reshape + 4D Transpose + Reshape
//===----------------------------------------------------------------------===//

class Transform5DTransposeTo4DPattern
    : public OpRewritePattern<ONNXTransposeOp> {
public:
  using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXTransposeOp transposeOp,
      PatternRewriter &rewriter) const override {
    // Check input is 5D
    auto inputType =
        dyn_cast<RankedTensorType>(transposeOp.getData().getType());
    if (!inputType || inputType.getRank() != 5)
      return rewriter.notifyMatchFailure(transposeOp, "input is not 5D");

    auto outputType =
        dyn_cast<RankedTensorType>(transposeOp.getResult().getType());

    // Get perm attribute
    auto permAttr = transposeOp.getPermAttr();

    // Extract perm values
    SmallVector<int64_t, 5> perm5D;
    for (auto attr : permAttr)
      perm5D.push_back(cast<IntegerAttr>(attr).getInt());

    // Find first consecutive pair
    auto maybeMergePos = findConsecutivePair(perm5D);
    if (!maybeMergePos)
      return rewriter.notifyMatchFailure(
          transposeOp, "no consecutive pair in perm to merge dimensions");

    size_t mergePos = *maybeMergePos;
    int64_t mergeDim = perm5D[mergePos];

    // Compute 4D perm
    auto perm4D = compute4DPerm(perm5D, mergePos, mergeDim);

    // Get shapes
    auto inputShape5D = inputType.getShape();
    auto outputShape5D = outputType.getShape();
    auto elemTy = inputType.getElementType();

    // Compute 4D input shape (merge mergeDim and mergeDim+1)
    auto inputShape4D = compute4DShape(inputShape5D, mergeDim);
    if (inputShape4D.empty())
      return rewriter.notifyMatchFailure(transposeOp,
          "cannot merge dynamic dimensions");

    // Compute 4D output shape by applying 4D perm to 4D input shape
    auto outputShape4D = applyPerm(inputShape4D, perm4D);

    MLIRContext *ctx = rewriter.getContext();
    Location loc = transposeOp.getLoc();

    // Create first reshape: 5D -> 4D (merge dimensions)
    auto reshape0ShapeAttr = getShapeAttr(ctx, inputShape4D);
    auto reshape0ShapeOp = rewriter.create<ONNXConstantOp>(loc,
        reshape0ShapeAttr.getType(), Attribute(), reshape0ShapeAttr,
        FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(),
        ArrayAttr());

    auto reshape0Type = RankedTensorType::get(inputShape4D, elemTy);
    auto reshape0 = rewriter.create<ONNXReshapeOp>(loc, reshape0Type,
        transposeOp.getData(), reshape0ShapeOp.getResult());

    // Create 4D transpose
    auto transpose4DType = RankedTensorType::get(outputShape4D, elemTy);
    auto transpose4D = rewriter.create<ONNXTransposeOp>(loc, transpose4DType,
        reshape0.getResult(), getI64ArrayAttr(ctx, perm4D));

    // Create second reshape: 4D -> 5D (split back)
    auto reshape1ShapeAttr = getShapeAttr(ctx, outputShape5D);
    auto reshape1ShapeOp = rewriter.create<ONNXConstantOp>(loc,
        reshape1ShapeAttr.getType(), Attribute(), reshape1ShapeAttr,
        FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(),
        ArrayAttr());

    auto reshape1 = rewriter.create<ONNXReshapeOp>(loc, outputType,
        transpose4D.getResult(), reshape1ShapeOp.getResult());

    rewriter.replaceOp(transposeOp, reshape1.getResult());

    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass for transforming 5D Transpose to Reshape + 4D Transpose + Reshape
struct Transform5DTransposeTo4DPass
    : public PassWrapper<Transform5DTransposeTo4DPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transform-5d-transpose-to-4d";
  }
  StringRef getDescription() const override {
    return "Transform 5D Transpose to Reshape + 4D Transpose + Reshape";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<Transform5DTransposeTo4DPattern>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createTransform5DTransposeTo4DPass() {
  return std::make_unique<Transform5DTransposeTo4DPass>();
}

} // namespace onnx_mlir

