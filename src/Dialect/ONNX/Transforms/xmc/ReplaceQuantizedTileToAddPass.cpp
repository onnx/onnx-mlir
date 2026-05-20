// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// MoveBroadcastTileForward: PSA2.2-style reshape+tile on TopK indices is moved
// onto the TopK data path (tile before TopK); gather takes cast(TopK indices)
// directly. Tile/indices may be integer (e.g. i32); TopK data stays quantized.
// No Q/DQ nodes are inserted.
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
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
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

static int64_t getTopKStaticK(ONNXTopKOp topk) {
  auto kValue = topk.getK();
  if (!kValue)
    return ShapedType::kDynamic;
  auto kConst = kValue.getDefiningOp<ONNXConstantOp>();
  if (!kConst)
    return ShapedType::kDynamic;
  auto valueAttr = kConst.getValueAttr();
  if (!valueAttr)
    return ShapedType::kDynamic;
  if (auto dense = dyn_cast<DenseElementsAttr>(valueAttr)) {
    if (dense.isSplat())
      return dense.getSplatValue<APInt>().getSExtValue();
    if (dense.getNumElements() == 1)
      return dense.getValues<APInt>()[0].getSExtValue();
  }
  return ShapedType::kDynamic;
}

/// ONNX attrs are often si64; IntegerAttr::getInt() requires signless integer.
static int64_t getIntegerAttrSExt(IntegerAttr attr) {
  if (attr.getType().isSignlessInteger() || attr.getType().isIndex())
    return attr.getInt();
  if (attr.getType().isSignedInteger())
    return attr.getSInt();
  return static_cast<int64_t>(attr.getUInt());
}

/// Mirrors xcompiler TransferBroadcastTileToAddPass::
/// transfer_move_broadcast_tile_forward (no DQ/Q on the indices path).
struct MoveBroadcastTileForwardPattern : public OpRewritePattern<ONNXTileOp> {
  using OpRewritePattern<ONNXTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTileOp tileOp, PatternRewriter &rewriter) const override {
    auto tileOutTy = dyn_cast<RankedTensorType>(tileOp.getType());
    if (!tileOutTy || !tileOutTy.hasStaticShape())
      return rewriter.notifyMatchFailure(tileOp, "tile result must be static");

    if (!tileOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          tileOp, "tile must have a single user");

    auto gatherOp = dyn_cast<ONNXGatherElementsOp>(*tileOp->user_begin());
    if (!gatherOp)
      return rewriter.notifyMatchFailure(
          tileOp, "tile user must be onnx.GatherElements");
    if (!gatherOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          gatherOp, "gather_elements must have a single user");
    auto gatherIndicesTy =
        dyn_cast<RankedTensorType>(gatherOp.getIndices().getType());
    if (!gatherIndicesTy ||
        gatherIndicesTy.getElementType() != tileOutTy.getElementType())
      return rewriter.notifyMatchFailure(
          tileOp, "tile output type must match gather indices type");

    auto reshapeOp = tileOp.getInput().getDefiningOp<ONNXReshapeOp>();
    if (!reshapeOp || !reshapeOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          tileOp, "tile input must be a single-use onnx.Reshape");
    ONNXCastOp castOp;
    ONNXTopKOp topkOp;
    if (failed(matchTopKIndicesChain(reshapeOp.getData(), castOp, topkOp)))
      return rewriter.notifyMatchFailure(
          tileOp, "reshape input must reach TopK indices via Cast");
    Value topkDataIn = topkOp.getOperand(0);
    auto topkDataTy = dyn_cast<RankedTensorType>(topkDataIn.getType());
    if (!topkDataTy || !topkDataTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          topkOp, "TopK data input must be a static ranked tensor");

    if (!topkOp.getAxisAttr())
      return rewriter.notifyMatchFailure(topkOp, "TopK requires static axis");

    int64_t k = getTopKStaticK(topkOp);
    if (k == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(topkOp, "TopK K must be a constant");

    Location loc = tileOp.getLoc();

    llvm::SmallVector<int64_t, 4> newReshapeShape(topkDataTy.getShape());
    newReshapeShape.push_back(1);

    auto shapeConst = rewriter.create<ONNXConstantOp>(loc,
        RankedTensorType::get({static_cast<int64_t>(newReshapeShape.size())},
            rewriter.getI64Type()),
        nullptr, rewriter.getI64TensorAttr(newReshapeShape), nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    auto newReshapeTy =
        RankedTensorType::get(newReshapeShape, topkDataTy.getElementType());
    auto newReshape = rewriter.create<ONNXReshapeOp>(
        loc, newReshapeTy, topkDataIn, shapeConst.getResult());

    llvm::SmallVector<int64_t, 4> newTileOutShape(topkDataTy.getShape());
    newTileOutShape.push_back(tileOutTy.getShape().back());
    // Tile on the TopK *data* path uses the data tensor element type (quant).
    auto newTileOutTy =
        RankedTensorType::get(newTileOutShape, topkDataTy.getElementType());
    auto newTile = rewriter.create<ONNXTileOp>(
        loc, newTileOutTy, newReshape.getResult(), tileOp.getRepeats());

    int64_t axis = getIntegerAttrSExt(cast<IntegerAttr>(topkOp.getAxisAttr()));
    if (axis < 0)
      axis += static_cast<int64_t>(newTileOutShape.size());
    llvm::SmallVector<int64_t, 4> topkOutShape(newTileOutShape);
    topkOutShape[axis] = k;

    auto oldValuesTy = dyn_cast<RankedTensorType>(topkOp.getValues().getType());
    auto oldIndicesTy =
        dyn_cast<RankedTensorType>(topkOp.getIndices().getType());
    if (!oldValuesTy || !oldIndicesTy)
      return rewriter.notifyMatchFailure(topkOp, "TopK results must be ranked");

    auto newValuesTy =
        RankedTensorType::get(topkOutShape, oldValuesTy.getElementType());
    auto newIndicesTy =
        RankedTensorType::get(topkOutShape, oldIndicesTy.getElementType());
    auto newTopk = rewriter.create<ONNXTopKOp>(topkOp.getLoc(), newValuesTy,
        newIndicesTy, newTile.getResult(), topkOp.getK(), topkOp.getAxisAttr(),
        topkOp.getLargestAttr(), topkOp.getSortedAttr());

    auto castOutTy = dyn_cast<RankedTensorType>(castOp.getType());
    if (!castOutTy)
      return rewriter.notifyMatchFailure(castOp, "Cast must be ranked tensor");
    // onnx.Cast has SameOperandsAndResultShape: output shape must match
    // indices.
    auto newCastTy =
        RankedTensorType::get(topkOutShape, castOutTy.getElementType());
    int64_t saturate = getIntegerAttrSExt(castOp.getSaturateAttr());
    Value newCast = rewriter.create<ONNXCastOp>(castOp.getLoc(), newCastTy,
        newTopk.getIndices(), saturate, castOutTy.getElementType());
    rewriter.modifyOpInPlace(
        gatherOp, [&]() { gatherOp.getOperation()->setOperand(1, newCast); });
    rewriter.replaceOp(topkOp, newTopk->getResults());
    rewriter.eraseOp(tileOp);
    rewriter.eraseOp(reshapeOp);
    rewriter.eraseOp(castOp);
    return success();
  }

private:
  static LogicalResult matchTopKIndicesChain(
      Value reshapeInput, ONNXCastOp &castOp, ONNXTopKOp &topkOp) {
    Value v = reshapeInput;
    ONNXCastOp lastCast;
    for (int i = 0; i < 8; ++i) {
      if (auto cast = v.getDefiningOp<ONNXCastOp>()) {
        lastCast = cast;
        v = cast.getInput();
        continue;
      }
      if (auto topk = v.getDefiningOp<ONNXTopKOp>()) {
        if (!lastCast)
          return failure();
        if (v != topk.getIndices())
          return failure();
        castOp = lastCast;
        topkOp = topk;
        return success();
      }
      Operation *def = v.getDefiningOp();
      if (!def || def->getNumOperands() != 1)
        return failure();
      v = def->getOperand(0);
    }
    return failure();
  }
};

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

class ReplaceIntegerTileToQuantizedTile : public OpRewritePattern<ONNXTileOp> {
public:
  using OpRewritePattern<ONNXTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTileOp tileOp, PatternRewriter &rewriter) const override {
    auto inputType =
        dyn_cast<RankedTensorType>(tileOp->getOperand(0).getType());
    auto outputType =
        dyn_cast<RankedTensorType>(tileOp->getResult(0).getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(
          tileOp, "Not RankedTensorType input/output");

    auto inElemType = inputType.getElementType();
    if (!isa<IntegerType>(inElemType))
      return rewriter.notifyMatchFailure(tileOp, "Not an integer input");

    Value defRes = tileOp->getOperand(0);
    auto *defOp = defRes.getDefiningOp();
    while (isa_and_present<ONNXReshapeOp, ONNXCastOp>(defOp)) {
      defRes = defOp->getOperand(0);
      defOp = defRes.getDefiningOp();
    }

    auto topK = dyn_cast_if_present<ONNXTopKOp>(defOp);
    if (!topK)
      return rewriter.notifyMatchFailure(tileOp, "Unable to find TopK");

    // ONNXTopK has two results: 0=values, 1=indices. We only want the
    // indices path (TopK->Cast->[Reshape]->Tile). Anything else (values
    // path, block arg, ...) is rejected.
    if (auto topKResult = dyn_cast<OpResult>(defRes);
        topKResult && topKResult.getResultNumber() != 1)
      return rewriter.notifyMatchFailure(
          topK, "Not the indices result of TopK");

    auto topKQType = dyn_cast<quant::UniformQuantizedType>(
        getElementTypeOrSelf(topK->getOperand(0).getType()));
    if (!topKQType)
      return rewriter.notifyMatchFailure(topK, "TopK input not Quantized type");
    auto storageType = topKQType.getStorageType();

    // == Rewrite section == //
    // -> TopK -> Cast -> [Reshape] -dq-> Tile -q->

    auto castOutputType = topK.getIndices().getType().clone(storageType);
    Value value = rewriter.create<ONNXCastOp>(
        tileOp.getLoc(), castOutputType, topK.getIndices(), 1, storageType);
    if (inputType.getShape() != castOutputType.getShape()) {
      auto shape = rewriter.create<ONNXConstantOp>(tileOp.getLoc(),
          RankedTensorType::get(inputType.getRank(), rewriter.getI64Type()),
          nullptr, rewriter.getI64TensorAttr(inputType.getShape()), nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr);
      value = rewriter.create<ONNXReshapeOp>(
          tileOp.getLoc(), inputType.clone(storageType), value, shape);
    }

    bool isSigned =
        storageType.isSignedInteger() || storageType.isSignlessInteger();
    auto qType = quant::UniformQuantizedType::get(isSigned, storageType,
        topKQType.getExpressedType(), 1.0, 0,
        quant::UniformQuantizedType::getDefaultMinimumForInteger(
            isSigned, storageType.getIntOrFloatBitWidth()),
        quant::UniformQuantizedType::getDefaultMaximumForInteger(
            isSigned, storageType.getIntOrFloatBitWidth()));
    value = rewriter.create<quant::StorageCastOp>(
        tileOp.getLoc(), inputType.clone(qType), value);

    value = rewriter.create<ONNXTileOp>(
        tileOp.getLoc(), outputType.clone(qType), value, tileOp.getRepeats());
    value = rewriter.create<quant::StorageCastOp>(
        tileOp.getLoc(), outputType.clone(storageType), value);
    rewriter.replaceOp(tileOp, value);

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
    return "Move broadcast tile before TopK (quant or integer indices), then "
           "lower onnx.Tile to onnx.Add (splat zero_point or splat 0)";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<MoveBroadcastTileForwardPattern>(context);
    patterns.add<ReplaceQuantizedTileToAddPattern>(context);
    patterns.add<ReplaceIntegerTileToQuantizedTile>(context);

    GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
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
