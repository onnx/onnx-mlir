// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Replaces onnx.Tile on uniformly quantized tensors with
// onnx.XCOMPILERFusedEltwiseOp(type=ADD, nonlinear=NONE): A=input,
// B=splat(zero_point) where B has the tiled output shape.
// Same scale/zero_point as the tile input/output so the fused add is identity
// in the real domain when NumPy-style broadcast matches tile expansion (e.g.
// repeats on size-1 axes or identity repeats).

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

static bool hasMatchingUniformQuant(Type tensorTypeA, Type tensorTypeB) {
  auto ra = dyn_cast<RankedTensorType>(tensorTypeA);
  auto rb = dyn_cast<RankedTensorType>(tensorTypeB);
  if (!ra || !rb)
    return false;
  auto qa = dyn_cast<quant::UniformQuantizedType>(ra.getElementType());
  auto qb = dyn_cast<quant::UniformQuantizedType>(rb.getElementType());
  if (!qa || !qb)
    return false;
  return std::fabs(qa.getScale() - qb.getScale()) <= 1e-6 &&
         qa.getZeroPoint() == qb.getZeroPoint();
}

/// Fills `out` with repeat counts; requires length == inputRank and all > 0.
static LogicalResult getStaticRepeatsFromConstant(
    Value repeatsVal, int64_t inputRank, SmallVectorImpl<int64_t> &out) {
  auto cst = repeatsVal.getDefiningOp<ONNXConstantOp>();
  if (!cst)
    return failure();

  if (auto valAttr = cst.getValueAttr()) {
    auto dense = dyn_cast<DenseIntElementsAttr>(valAttr);
    if (!dense || dense.getType().getRank() != 1)
      return failure();
    if (static_cast<int64_t>(dense.getNumElements()) != inputRank)
      return failure();
    for (int64_t r : dense.getValues<int64_t>()) {
      if (r <= 0)
        return failure();
      out.push_back(r);
    }
    return success();
  }

  if (auto intsAttr = cst.getValueIntsAttr()) {
    if (static_cast<int64_t>(intsAttr.size()) != inputRank)
      return failure();
    for (Attribute a : intsAttr) {
      int64_t r = cast<IntegerAttr>(a).getInt();
      if (r <= 0)
        return failure();
      out.push_back(r);
    }
    return success();
  }

  return failure();
}

/// Returns true if `lhs` and `rhs` are NumPy-style broadcast-compatible and the
/// broadcast result equals `expected` (same rules as XCOMPILERFusedEltwise /
/// ONNX broadcast eltwise verification).
static bool broadcastMatchesExpectedShape(
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
      return failure();
    auto inQuant =
        dyn_cast<quant::UniformQuantizedType>(inType.getElementType());
    if (!inQuant || !isa<quant::UniformQuantizedType>(outType.getElementType()))
      return failure();

    if (!hasMatchingUniformQuant(input.getType(), tileOp.getType()))
      return failure();
    int64_t rank = inType.getRank();
    if (rank <= 0)
      return failure();

    SmallVector<int64_t, 8> repeats;
    if (failed(
            getStaticRepeatsFromConstant(tileOp.getRepeats(), rank, repeats)))
      return failure();
    auto inShape = inType.getShape();
    SmallVector<int64_t, 8> computedOutShape;
    computedOutShape.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      if (inShape[i] == ShapedType::kDynamic)
        return failure();
      computedOutShape.push_back(inShape[i] * repeats[static_cast<size_t>(i)]);
    }

    if (!outType.hasStaticShape())
      return failure();
    if (outType.getShape() != ArrayRef<int64_t>(computedOutShape))
      return failure();

    // Fused eltwise ADD requires NumPy-style broadcast; reject when tile is not
    // expressible as broadcast(input, splat_const) with output = tile shape.
    if (!broadcastMatchesExpectedShape(
            inShape, ArrayRef<int64_t>(computedOutShape), computedOutShape))
      return failure();

    Location loc = tileOp.getLoc();
    int64_t zeroPoint = inQuant.getZeroPoint();
    Type storageTy = inQuant.getStorageType();

    auto quantResultType = RankedTensorType::get(computedOutShape, inQuant);
    // Match ReplaceExpandWithEltwise: value dense uses storage element type;
    // result tensor type remains uniformly quantized.
    auto zpStorageType = RankedTensorType::get(computedOutShape, storageTy);
    auto zpSplatAttr = DenseElementsAttr::get(
        zpStorageType, rewriter.getIntegerAttr(storageTy, zeroPoint));
    auto valueNamedAttr = rewriter.getNamedAttr("value", zpSplatAttr);
    auto zpTensorConst = rewriter.create<ONNXConstantOp>(loc, quantResultType,
        ValueRange{}, ArrayRef<NamedAttribute>{valueNamedAttr});
    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(loc,
        tileOp.getType(), input, zpTensorConst.getResult(),
        /*clip_max=*/IntegerAttr(),
        /*clip_min=*/IntegerAttr(),
        /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
        /*leakyrelu_alpha=*/FloatAttr(),
        /*mul_y=*/FloatAttr(),
        /*nonlinear=*/rewriter.getStringAttr("NONE"),
        /*nonlinear_in_scales=*/FloatAttr(),
        /*nonlinear_in_zeropoints=*/IntegerAttr(),
        /*prelu_in=*/IntegerAttr(),
        /*prelu_shift=*/IntegerAttr(),
        /*type=*/rewriter.getStringAttr("ADD"));
    rewriter.replaceOp(tileOp, fusedOp.getResult());
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
    return "Lower quantized onnx.Tile to XCOMPILERFusedEltwise ADD with "
           "splat(zero_point) second operand";
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
