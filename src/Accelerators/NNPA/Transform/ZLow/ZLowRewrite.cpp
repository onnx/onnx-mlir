/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZLowRewrite.cpp - ZLow Rewrite Patterns ---------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This pass implements optimizations for ZLow operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

#include <map>

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

/// This pattern rewrites
/// ```mlir
///   zlow.unstick(%input, %output)
///   %view = viewOp(%output)
///   zlow.stick(%view, %res)
/// ```
/// by removing `zlow.stick` and replacing `%res` by `%input`, which is
/// constrained by that `%input` and `%res` have the same static shape.
/// This pattern potentially removes `zlow.unstick` and `viewOp` if they are
/// dangling.
///
/// `viewOp` can be any op that inherits ViewLikeOpInterface, e.g.
/// memref.reinterpret_cast, memref.collapse_shape, memref.expand_shape.
//
class StickViewUnstickRemovalPattern : public OpRewritePattern<ZLowStickOp> {
public:
  using OpRewritePattern<ZLowStickOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowStickOp stickOp, PatternRewriter &rewriter) const override {
    Value stickInput = stickOp.getX();

    // Input is a block argument, ignore it.
    if (stickInput.dyn_cast<BlockArgument>())
      return failure();

    // Input must have no affine layout. In other words, it has been normalized.
    if (hasNonIdentityLayout(stickInput.getType()))
      return failure();

    // Input is a view.
    ViewLikeOpInterface viewOp =
        llvm::dyn_cast<ViewLikeOpInterface>(stickInput.getDefiningOp());
    if (!viewOp)
      return failure();
    // Get the source of the view.
    Value viewSource = viewOp.getViewSource();

    // Get UnstickOp that unstickifies the view source.
    // There is only one UnstickOp per buffer, so stop searching when we get
    // one.
    ZLowUnstickOp unstickOp;
    for (Operation *user : viewSource.getUsers()) {
      ZLowUnstickOp userOp = llvm::dyn_cast<ZLowUnstickOp>(user);
      if (!userOp)
        continue;
      // UnstickOp must be before the view operation.
      if (userOp.getOut() == viewSource &&
          user->isBeforeInBlock(viewOp.getOperation())) {
        unstickOp = userOp;
        break;
      }
    }
    if (!unstickOp)
      return failure();

    // Match shapes.
    Value stickRes = stickOp.getOut();
    Value unstickInput = unstickOp.getX();
    MemRefType stickResType = stickRes.getType().dyn_cast<MemRefType>();
    MemRefType unstickInputType = unstickInput.getType().dyn_cast<MemRefType>();
    if (!stickResType.hasStaticShape() ||
        (stickResType.getShape() != unstickInputType.getShape()))
      return failure();

    // Rewrite
    rewriter.eraseOp(stickOp);
    stickRes.replaceAllUsesWith(unstickInput);
    // Remove the view op if there is no use.
    if (viewOp.getOperation()->getResults()[0].use_empty())
      rewriter.eraseOp(viewOp);
    // Remove unstick if there is no use of its second operand except itself.
    if (unstickOp.getOut().hasOneUse())
      rewriter.eraseOp(unstickOp);

    return success();
  }
};

// clang-format off
///
/// * Pattern to rewrite
/// ```
/// zlow.unstick -> affine.for (affine.load -> affine.store) -> zlow.stick
///    |                            |
///    |                            '--------> affine.store) -> zlow.stick
///    |
///    '----------> affine.for (affine.load -> affine.store) -> zlow.stick
///                                                              ^
/// zlow.unstick -> affine.for (affine.load -> affine.store) ----'
/// ```
///
/// * Example:
///
/// Consider the following code: 
/// ```mlir
/// zlow.unstick(%stick, %A) {layout = "2D"}: memref<2x3xf16, #map2D>, memref<2x3xf32>
/// affine.for
///   %a = affine.load(%A, %load_indices) : memref<2x3xf32>
///   affine.store(%a, %B, %store_indices) : memref<4x5x6xf32>
/// %res = memref.alloc() : memref<4x5x6xf16, #map3D>
/// zlow.stick(%B, %res) {layout = "3D"}: memref<4x5x6xf32>, memref<4x5x6xf16,
/// #map3D>
/// ```
/// `%stick` memref is unstickified and shuffled by the pair of (affine.load,affine.store),
/// then stickified again. It said data are transfered from a stickified memref
/// into another stickified memref via a chain of affine transformation.
///
/// The above code can be rewritten into the following code:
/// ```mlir
/// %res = memref.alloc() : memref<4x5x6xf16, #map3D>
/// affine.for
///   %a = affine.load(%stick, %load_indices) : memref<2x3xf16, #map2D>
///   affine.store(%a, %res, %store_indices) : memref<4x5x6xf16, #map3D>
/// ```
/// where data will be directly loaded from / stored to stickified memref.
///
/// This pattern is often found in code generated for data transformation such
/// as Transpose, Concat, and Split.
///
/// * Why does this rewriting work?
/// 
/// - This rewriting depends on the fact that `zlow.stick` and `zlow.unstick`
/// maintain an affine map that maps one element in a memref to an element in
/// another memref. Those maps are `#map2D` and `#map3D` in the above example.
/// Combined with affine.load and affine.store, one element in a stickified
/// memref can be forwarded directly into an element in another stickifired
/// memref without `zlow.stick` and `zlow.unstick`.
///
/// - The shape of the input and output memrefs of `zlow.stick`/`zlow.unstick`
/// are the same except the case of layout NCHW. In case of NCHW, dimensions are
/// permuted, so we handle NCHW as a special case in this rewriting.
/// ```mlir
///  zlow.stick(%X, %res) {layout = "NCHW"}: memref<1x3x5x7xf32>, memref<1x5x7x3xf16, #mapNHWC>
///  ```
///  Shape of `%X` is in NCHW while shape of `%res` is in NHWC.
//
/// ```mlir
/// zlow.unstick(%X, %res) {layout = "NCHW"}: memref<1x5x7x3xf16, #mapNHWC>, memref<1x3x5x7xf32>
/// ```
///  Shape of `%X` is in NHWC while shape of `%res` is in NCHW.
///
/// * Limitations
///
/// - Unstickified memrefs (`%A` and `%B`) must have no affine map.
/// Theoretically, we could support affine map on unstickified memrefs by
/// composing affine-map.

// clang-format on

class UnstickLoadStoreStickRemovalPattern
    : public OpRewritePattern<ZLowUnstickOp> {
public:
  using OpRewritePattern<ZLowUnstickOp>::OpRewritePattern;

  UnstickLoadStoreStickRemovalPattern(MLIRContext *context,
      llvm::SmallDenseSet<ZLowStickOp, 4> &removableStickOps_)
      : OpRewritePattern(context, /*benefit=*/1),
        removableStickOps(removableStickOps_) {}

  LogicalResult matchAndRewrite(
      ZLowUnstickOp unstickOp, PatternRewriter &rewriter) const override {
    Location loc = unstickOp.getLoc();
    Operation *op = unstickOp.getOperation();
    // stickifiedMemRef has affine layout, e.g. MemRef<1x3x5xf32, #map>
    Value stickifiedMemRef = unstickOp.getX();
    // cpuMemRef has no affine layout, e.g. MemRef<1x3x5xf32>
    Value cpuMemRef = unstickOp.getOut();
    std::string unstickLayout = unstickOp.getLayout().value().str();
    bool unstickNCHWLayout = (unstickLayout == LAYOUT_NCHW);

    // Common types.
    Type stickifiedElementType =
        stickifiedMemRef.getType().cast<MemRefType>().getElementType();
    Type cpuElementType =
        cpuMemRef.getType().cast<MemRefType>().getElementType();

    // Stickified Memref must have affine layout to access elements.
    if (!hasNonIdentityLayout(stickifiedMemRef.getType()))
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "Stickified Memref has no affine layout";
      });

    // Do not support affine layout in the CPU Memref at this moment.
    if (hasNonIdentityLayout(cpuMemRef.getType()))
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "Unstickified Memref has affine layout";
      });

    // Do not support layout 1D and 2DS since their access index functions are
    // incorrect: https://github.com/onnx/onnx-mlir/issues/1940
    if ((unstickLayout == LAYOUT_1D) || (unstickLayout == LAYOUT_2DS))
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "Unsupport layout 1D and 2DS";
      });

    // 1. Match pattern: data flows from zlow.unstick to zlow.stick via
    // affine.load and affine.store.
    // - Support sharing load-from/store-to zlow.unstick/zlow.stick.
    //
    //  zlow.unstick -> affine.for (affine.load -> affine.store) -> zlow.stick
    //     |                            |
    //     |                            '--------> affine.store) -> zlow.stick
    //     |
    //     '----------> affine.for (affine.load -> affine.store) -> zlow.stick
    //                                                               ^
    //  zlow.unstick -> affine.for (affine.load -> affine.store) ----'
    //

    // All consumers of zlow.unstick must be affine.load.
    SmallVector<AffineLoadOp, 4> loadOps;
    if (!matchAndCollectAffineLoad(unstickOp, cpuMemRef, loadOps))
      rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "Failed to match AffineLoadOp";
      });

    // All consumers of affine.load must be affine.store.
    // affine.store must store to a Memref allocated by memref.alloc.
    SmallVector<AffineStoreOp, 4> storeOps;
    if (!matchAndCollectAffineStore(loadOps, storeOps))
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "Failed to match AffineStoreOp";
      });

    // Each affine.store is connected to one zlow.stick.
    std::map<AffineStoreOp, ZLowStickOp> StoreOpStickOpMap;
    SmallVector<ZLowStickOp, 4> stickOps;
    if (!matchAndCollectStickOp(storeOps, stickOps, StoreOpStickOpMap))
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "Two ZLowStickOp linked to an AffineStoreOp";
      });

    // 2. Rewrite
    // - Rewrite AffineLoadOp to use stickified Memref directly.
    MultiDialectBuilder<AffineBuilder> create(rewriter, loc);
    for (AffineLoadOp loadOp : loadOps) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(loadOp);
      // Clone loadOp with new Memref, indices and return type.
      IRMapping operandMap;
      operandMap.map(loadOp.getMemref(), stickifiedMemRef);
      if (unstickNCHWLayout) {
        // Permute indices in case of NCHW layout.
        // for zlow.unstick: input is NHWC, output is NCHW.
        ValueRange NCHWIndices = loadOp.getIndices();
        SmallVector<Value, 4> NHWCIndices;
        NHWCIndices.emplace_back(NCHWIndices[0]); // N
        NHWCIndices.emplace_back(NCHWIndices[2]); // H
        NHWCIndices.emplace_back(NCHWIndices[3]); // W
        NHWCIndices.emplace_back(NCHWIndices[1]); // C
        operandMap.map(NCHWIndices, NHWCIndices);
      }
      Operation *clonedOp = rewriter.clone(*loadOp.getOperation(), operandMap);
      clonedOp->getResult(0).setType(stickifiedElementType);
      // This DummyOp is used to make the intermediate generated code valid. It
      // wil be removed automatically via canonicalization.
      Value dummyConverter = rewriter.create<ZLowDummyOp>(
          loc, cpuElementType, clonedOp->getResult(0));
      rewriter.replaceOp(loadOp, {dummyConverter});
    }

    // - Rewrite AffineStoreOp to use stickified Memref directly.
    for (AffineStoreOp storeOp : storeOps) {
      Value storeMemref = storeOp.getMemref();
      Value storeValue = storeOp.getValue();
      ZLowStickOp myStickOp = StoreOpStickOpMap[storeOp];
      Value stickMemref = myStickOp.getOut();
      std::string stickLayout = myStickOp.getLayout().value().str();
      bool stickNCHWLayout = (stickLayout == LAYOUT_NCHW);

      // Move stickMemref's AllocOp up before affine.for so that it
      // dominates its uses. A good place is just after storeMemref's AllocOp.
      //
      // Get AllocOps that allocated storeMemref and stickMemref.
      Operation *storeAllocOp = storeMemref.getDefiningOp();
      Operation *stickAllocOp = stickMemref.getDefiningOp();
      // stickAllocOp should be after storeAllocOp, since dimensions come from
      // storeAllocOp according to the definition of zlow.stick.
      Operation *justMovedOp = nullptr;
      // Move AllocOp's operands first.
      for (unsigned i = 0; i < stickAllocOp->getNumOperands(); ++i) {
        Value oprd = stickAllocOp->getOperand(i);
        if (isa<BlockArgument>(oprd))
          continue;
        Operation *opToMove = oprd.getDefiningOp();
        // Do not move, it is potentially used by storeAllocOp and it is a good
        // place already.
        if (opToMove->isBeforeInBlock(storeAllocOp))
          continue;
        if (justMovedOp)
          opToMove->moveAfter(justMovedOp);
        else
          opToMove->moveAfter(storeAllocOp);
        justMovedOp = opToMove;
      }
      // Move AllocOp.
      if (justMovedOp)
        stickAllocOp->moveAfter(justMovedOp);
      else
        stickAllocOp->moveAfter(storeAllocOp);

      // Replace storeOp.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(storeOp);
      // This DummyOp is used to make the intermediate generated code valid. It
      // will be removed automatically via canonicalization.
      Value dummyConverter =
          rewriter.create<ZLowDummyOp>(loc, stickifiedElementType, storeValue);
      // Clone storeOp with new Memref, Value, and Indices.
      IRMapping operandMap;
      operandMap.map(storeOp.getMemref(), stickMemref);
      operandMap.map(storeOp.getValue(), dummyConverter);
      // Permute indices in case of NCHW layout.
      if (stickNCHWLayout) {
        // for zlow.stick: input is NCHW, output is NHWC.
        ValueRange NCHWIndices = storeOp.getIndices();
        SmallVector<Value, 4> NHWCIndices;
        NHWCIndices.emplace_back(NCHWIndices[0]); // N
        NHWCIndices.emplace_back(NCHWIndices[2]); // H
        NHWCIndices.emplace_back(NCHWIndices[3]); // W
        NHWCIndices.emplace_back(NCHWIndices[1]); // C
        operandMap.map(NCHWIndices, NHWCIndices);
      }
      rewriter.clone(*storeOp.getOperation(), operandMap);
      rewriter.eraseOp(storeOp);
    }

    // Remove ZLowUnstickOp.
    rewriter.eraseOp(unstickOp);
    // Copy ZLowStickOp to the removableStickOps. We cannot remove it now
    // because there are potentially other AffineStoreOps using to it.
    for (ZLowStickOp stickOp : stickOps)
      removableStickOps.insert(stickOp);
    return success();
  }

private:
  llvm::SmallDenseSet<ZLowStickOp, 4> &removableStickOps;

  // Collect affine.load operations that connect to zlow.unstick.
  bool matchAndCollectAffineLoad(ZLowUnstickOp unstickOp, Value loadMemref,
      SmallVectorImpl<AffineLoadOp> &loadOps) const {
    for (Operation *user : loadMemref.getUsers()) {
      if (user == unstickOp.getOperation())
        continue;
      if (auto loadOp = llvm::dyn_cast<AffineLoadOp>(user))
        loadOps.emplace_back(loadOp);
      else
        return false;
    }

    return (loadOps.size() != 0);
  }

  // Collect affine.store operations that connect to affine.load.
  bool matchAndCollectAffineStore(const SmallVectorImpl<AffineLoadOp> &loadOps,
      SmallVectorImpl<AffineStoreOp> &storeOps) const {
    for (AffineLoadOp loadOp : loadOps) {
      Value loadValue = loadOp.getValue();
      for (Operation *user : loadValue.getUsers()) {
        if (user == loadOp.getOperation())
          continue;
        if (auto storeOp = llvm::dyn_cast<AffineStoreOp>(user)) {
          // Store's input must be defined by a memref.alloc.
          Value storeMemref = storeOp.getMemref();
          if (storeMemref.isa<BlockArgument>())
            return false;
          Operation *allocOp = storeMemref.getDefiningOp();
          if (!isa<memref::AllocOp>(allocOp))
            return false;
          storeOps.emplace_back(storeOp);
        } else
          return false;
      }
    }
    return (storeOps.size() != 0);
  }

  // Collect zlow.stick operations that connect to affine.store.
  bool matchAndCollectStickOp(const SmallVectorImpl<AffineStoreOp> &storeOps,
      SmallVectorImpl<ZLowStickOp> &stickOps,
      std::map<AffineStoreOp, ZLowStickOp> &StoreOpStickOpMap) const {
    for (AffineStoreOp storeOp : storeOps) {
      ZLowStickOp myStickOp;
      Value destMemref = storeOp.getMemref();
      for (Operation *user : destMemref.getUsers()) {
        if (user == storeOp.getOperation())
          continue;
        if (auto storeOp = llvm::dyn_cast<AffineStoreOp>(user))
          continue;
        if (auto stick = llvm::dyn_cast<ZLowStickOp>(user)) {
          // Do not support layout 1D and 2DS since their access index
          // functions are incorrect:
          // https://github.com/onnx/onnx-mlir/issues/1940
          std::string stickLayout = stick.getLayout().value().str();
          if ((stickLayout == LAYOUT_1D) || (stickLayout == LAYOUT_2DS))
            return false;

          if (myStickOp)
            return false;
          else
            myStickOp = stick;
        } else
          return false;
      }
      stickOps.emplace_back(myStickOp);
      StoreOpStickOpMap[storeOp] = myStickOp;
    }

    return (stickOps.size() != 0);
  }
};

/*!
 *  Function pass that optimizes ZLowIR.
 */
class ZLowRewritePass
    : public PassWrapper<ZLowRewritePass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "zlow-rewrite"; }

  StringRef getDescription() const override { return "Rewrite ZLow Ops."; }

  void runOnOperation() override {
    Operation *function = getOperation();

    llvm::SmallDenseSet<ZLowStickOp, 4> removableStickOps;
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<StickViewUnstickRemovalPattern>(&getContext());
    patterns.insert<UnstickLoadStoreStickRemovalPattern>(
        &getContext(), removableStickOps);

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      return signalPassFailure();

    // Remove ZLowStickOp that were marked "removable".
    for (ZLowStickOp stickOp : removableStickOps) {
      if (!stickOp) // removed, continue.
        continue;
      stickOp.getOperation()->erase();
    }
  }
};

std::unique_ptr<Pass> createZLowRewritePass() {
  return std::make_unique<ZLowRewritePass>();
}

} // namespace zlow
} // namespace onnx_mlir
