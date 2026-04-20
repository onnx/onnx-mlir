// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Replaces onnx.Tile with onnx.Add: A=input, B=splat(identity additive) where
// B has the tiled output shape.
// - Uniformly quantized tensors: B is splat(zero_point); input/output must
//   share the same scale and zero_point.
// - Non-quantized tensors: B is splat(0) in the element type.
// The fused add is identity in the real domain when NumPy-style broadcast
// matches tile expansion (e.g. repeats on size-1 axes or identity repeats).
//
// The splat second operand uses the onnx.Tile result shape from the model (IR),
// not a recomputed input×repeats product.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cmath>

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

bool hasMatchingUniformQuant(
    quant::UniformQuantizedType typeA, quant::UniformQuantizedType typeB) {
  return typeA.getStorageType() == typeB.getStorageType() &&
         std::fabs(typeA.getScale() - typeB.getScale()) <= 1e-6 &&
         typeA.getZeroPoint() == typeB.getZeroPoint();
}

/// Returns true if `lhs` and `rhs` are NumPy-style broadcast-compatible and the
/// broadcast result equals `expected` (same rules as onnx.Add broadcast).
bool broadcastMatchesExpectedShape(
    ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs, ArrayRef<int64_t> expected) {
  SmallVector<int64_t> bcastShape;
  if (!OpTrait::util::getBroadcastedShape(lhs, rhs, bcastShape))
    return false;
  return ArrayRef<int64_t>(bcastShape) == expected;
}

struct ReplaceQuantizedTileToAddPattern : public OpRewritePattern<ONNXTileOp> {
  using OpRewritePattern<ONNXTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTileOp tileOp, PatternRewriter &rewriter) const override {
    Value input = tileOp.getInput();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(tileOp.getType());
    if (!inType || !outType)
      return rewriter.notifyMatchFailure(
          tileOp, "input and result must be ranked tensor types");

    Type inElem = inType.getElementType();
    Type outElem = outType.getElementType();
    auto inQuant = dyn_cast<quant::UniformQuantizedType>(inElem);
    auto outQuant = dyn_cast<quant::UniformQuantizedType>(outElem);

    if (isa<IntegerType>(inElem) || isa<IntegerType>(outElem)) {
      return rewriter.notifyMatchFailure(
          tileOp, "Only float or quantized type supported");
    } else if (static_cast<bool>(inQuant) != static_cast<bool>(outQuant)) {
      return rewriter.notifyMatchFailure(
          tileOp, "mixed quant and non-quant not supported");
    } else if (inQuant) {
      if (!hasMatchingUniformQuant(inQuant, outQuant))
        return rewriter.notifyMatchFailure(tileOp,
            "quant scale and zero_point must match on input and output");
    } else if (inElem != outElem) {
      return rewriter.notifyMatchFailure(
          tileOp, "non-quantized input and output element types must be same");
    }

    int64_t rank = inType.getRank();
    if (rank <= 0)
      return rewriter.notifyMatchFailure(
          tileOp, "expected positive tensor rank");

    auto inShape = inType.getShape();
    for (int64_t i = 0; i < rank; ++i) {
      if (inShape[i] == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(tileOp,
            "input dimensions must be static (dynamic dims not supported)");
    }

    ArrayRef<int64_t> outShape = outType.getShape();
    if (static_cast<int64_t>(outShape.size()) != rank)
      return rewriter.notifyMatchFailure(
          tileOp, "output rank must match input rank");

    // onnx.Add requires NumPy-style broadcast; reject when tile is not
    // expressible as broadcast(input, splat_const) with output = tile shape.
    if (!broadcastMatchesExpectedShape(inShape, outShape, outShape))
      return rewriter.notifyMatchFailure(tileOp,
          "tile expansion is not expressible as NumPy broadcast from input to "
          "a splat constant with the tiled output shape");

    Location loc = tileOp.getLoc();

    ONNXConstantOp zpTensorConst;
    if (inQuant) {
      int64_t zeroPoint = inQuant.getZeroPoint();
      Type storageTy = inQuant.getStorageType();

      auto quantResultType = RankedTensorType::get(outShape, inQuant);
      // Match ReplaceExpandWithEltwise: value dense uses storage element type;
      // result tensor type remains uniformly quantized.
      auto zpStorageType = RankedTensorType::get(outShape, storageTy);
      auto zpSplatAttr = DenseElementsAttr::get(
          zpStorageType, rewriter.getIntegerAttr(storageTy, zeroPoint));
      auto valueNamedAttr = rewriter.getNamedAttr("value", zpSplatAttr);
      zpTensorConst = rewriter.create<ONNXConstantOp>(loc, quantResultType,
          ValueRange{}, ArrayRef<NamedAttribute>{valueNamedAttr});
    } else {
      // Non-quantized: identity add uses splat(0), same as
      // ReplaceExpandWithEltwise.
      auto zeroAttr =
          DenseElementsAttr::get(RankedTensorType::get(outShape, inElem),
              rewriter.getZeroAttr(inElem));
      auto valueNamedAttr = rewriter.getNamedAttr("value", zeroAttr);
      zpTensorConst = rewriter.create<ONNXConstantOp>(loc, tileOp.getType(),
          ValueRange{}, ArrayRef<NamedAttribute>{valueNamedAttr});
    }
    auto addOp = rewriter.create<ONNXAddOp>(
        loc, tileOp.getType(), input, zpTensorConst.getResult());
    rewriter.replaceOp(tileOp, addOp.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ReplaceQuantizedTileToAddPass
    : public PassWrapper<ReplaceQuantizedTileToAddPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "replace-quantized-tile-to-add";
  }
  StringRef getDescription() const override {
    return "Lower onnx.Tile to onnx.Add: quantized tile uses "
           "splat(zero_point); non-quantized tile uses splat(0)";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReplaceQuantizedTileToAddPattern>(context);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config))) {
      getOperation().emitError(
          "replace-quantized-tile-to-add: greedy pattern rewrite did not "
          "converge");
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createReplaceQuantizedTileToAddPass() {
  return std::make_unique<ReplaceQuantizedTileToAddPass>();
}

} // namespace onnx_mlir
