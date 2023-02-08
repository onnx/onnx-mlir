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

    // Input must has no affine layout. In other words, it has been normalized.
    if (auto type = dyn_cast<MemRefType>(stickInput.getType())) {
      AffineMap m = type.getLayout().getAffineMap();
      if (m.getNumResults() != 1 && !m.isIdentity())
        return failure();
    }

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

/// This pattern rewrites
/// ```mlir
///   zlow.unstick(%stick, %A)
///   affine.for
///       %a = affine.load(%A, %load_indices)
///       affine.store(%a, %B, %store_indices)
///   %res = memref.alloc()
///   zlow.stick(%B, %res)
/// ```
/// by
/// ```mlir
/// %res = memref.alloc()
/// affine.for
///     %a = affine.load(%stick, %load_indices)
///     affine.store(%a, %res, %store_indices)
/// ```
/// where data will be directly loaded from / stored to stickified tensor.
//
/// This pattern potentially removes `zlow.unstick` and `zlow.stick` if they are
/// dangling.
///
/// This pattern is often found in code generated for data transformation
/// operations such as Transpose, Concat.
///

class UnstickLoadStoreStickRemovalPattern
    : public OpRewritePattern<ZLowUnstickOp> {
public:
  using OpRewritePattern<ZLowUnstickOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowUnstickOp unstickOp, PatternRewriter &rewriter) const override {
    Location loc = unstickOp.getLoc();
    Operation *op = unstickOp.getOperation();
    // stickifiedMemRef has affine layout, e.g. MemRef<1x3x5xf32, #map>
    Value stickifiedMemRef = unstickOp.getX();
    // cpuMemRef has no affine layout, e.g. MemRef<1x3x5xf32>
    Value cpuMemRef = unstickOp.getOut();

    // Common types.
    Type stickifiedElementType =
        stickifiedMemRef.getType().cast<MemRefType>().getElementType();
    Type cpuElementType =
        cpuMemRef.getType().cast<MemRefType>().getElementType();

    // Input must has affine layout to access elements in the stickified MemRef.
    if (auto type = dyn_cast<MemRefType>(stickifiedMemRef.getType())) {
      AffineMap m = type.getLayout().getAffineMap();
      if (m.isIdentity())
        return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
          diag << "Input has no affine layout ";
        });
    }

    // Do not support layout 1D and 2DS since their access index functions are
    // incorrect: https://github.com/onnx/onnx-mlir/issues/1940
    std::string layout = unstickOp.getLayout().value().str();
    if ((layout == LAYOUT_1D) || (layout == LAYOUT_2DS))
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "Unsupport layout 1D and 2DS";
      });

    // 1. Match pattern: unstick -> load -> store -> stick.

    // All users of cpuMemRef must be affine.load.
    SmallVector<AffineLoadOp, 4> loadOps;
    for (Operation *user : cpuMemRef.getUsers()) {
      if (user == op)
        continue;
      if (auto loadOp = llvm::dyn_cast<AffineLoadOp>(user))
        loadOps.emplace_back(loadOp);
      else
        return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
          diag << "There is non-affine-load op";
        });
    }
    if (loadOps.size() == 0)
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "There is no affine load op";
      });

    // All users of loadOps must be affine.store.
    // affine.store must store to a Memref allocated by memref.alloc.
    SmallVector<AffineStoreOp, 4> storeOps;
    for (AffineLoadOp loadOp : loadOps) {
      Value loadValue = loadOp.getValue();
      for (Operation *user : loadValue.getUsers()) {
        if (user == loadOp.getOperation())
          continue;
        if (auto storeOp = llvm::dyn_cast<AffineStoreOp>(user)) {
          // Store's input must be defined by a memref.alloc.
          Value storeMemref = storeOp.getMemref();
          if (storeMemref.isa<BlockArgument>())
            return rewriter.notifyMatchFailure(
                op, [&](::mlir::Diagnostic &diag) {
                  diag << "Store to a BlockArgument";
                });
          Operation *allocOp = storeMemref.getDefiningOp();
          if (!isa<memref::AllocOp>(allocOp))
            return rewriter.notifyMatchFailure(
                op, [&](::mlir::Diagnostic &diag) {
                  diag << "Store's destination was not allocated by AllocOp";
                });
          storeOps.emplace_back(storeOp);
        } else
          return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
            diag << "There is non-affine-store op";
          });
      }
    }
    if (storeOps.size() == 0)
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "There is no affine store op";
      });

    // All storeOps have ZLowStick as destination.
    // Each storeOp must have only single ZLowStick.
    std::map<AffineStoreOp, ZLowStickOp> StoreOpStickOpMap;
    SmallVector<ZLowStickOp, 4> stickOps;
    for (AffineStoreOp storeOp : storeOps) {
      ZLowStickOp myStickOp;
      Value destMemref = storeOp.getMemref();
      for (Operation *user : destMemref.getUsers()) {
        if (user == storeOp.getOperation())
          continue;
        if (auto storeOp = llvm::dyn_cast<AffineStoreOp>(user))
          continue;
        if (auto stick = llvm::dyn_cast<ZLowStickOp>(user)) {
          if (myStickOp)
            return rewriter.notifyMatchFailure(
                op, [&](::mlir::Diagnostic &diag) {
                  diag << "Two ZLowStickOp linked to an AffineStoreOp";
                });
          else
            myStickOp = stick;
        } else
          return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
            diag << "There is non-stick op";
          });
      }
      stickOps.emplace_back(myStickOp);
      StoreOpStickOpMap[storeOp] = myStickOp;
    }
    if (stickOps.size() == 0)
      return rewriter.notifyMatchFailure(op,
          [&](::mlir::Diagnostic &diag) { diag << "There is no stick op"; });

    // 2. Rewrite
    // - replace all source MemRefs of AffineLoadOp by unstick's MemRef.
    MultiDialectBuilder<AffineBuilder> create(rewriter, loc);
    for (AffineLoadOp loadOp : loadOps) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(loadOp);
      // Clone loadOp with new Memref and return type, which preserves the
      // access indices.
      IRMapping operandMap;
      operandMap.map(loadOp.getMemref(), stickifiedMemRef);
      Operation *clonedOp = rewriter.clone(*loadOp.getOperation(), operandMap);
      clonedOp->getResult(0).setType(stickifiedElementType);
      Value convertedVal = rewriter.create<ZLowDummyOp>(
          loc, cpuElementType, clonedOp->getResult(0));
      rewriter.replaceOp(loadOp, {convertedVal});
    }

    // - replace all target MemRefs of AffineStoreOp by stick's zMemRef.
    // TODO: get the ealiest AllocOp from storeOps to replace.
    for (AffineStoreOp storeOp : storeOps) {
      Value storeMemref = storeOp.getMemref();
      Value storeValue = storeOp.getValue();
      ZLowStickOp myStickOp = StoreOpStickOpMap[storeOp];
      Value stickMemref = myStickOp.getOut();
      // Get AllocOps that allocated storeMemref and stickMemref.
      Operation *storeAllocOp = storeMemref.getDefiningOp();
      Operation *stickAllocOp = stickMemref.getDefiningOp();
      // stickAllocOp should be after storeAllocOp, since dimensions come from
      // storeAllocOp according to the definition of zlow.stick.
      stickAllocOp->moveAfter(storeAllocOp);
      for (int i = 0; i < stickAllocOp->getNumOperands(); ++i) {
        Value oprd = stickAllocOp->getOperand(i);
        if (isa<BlockArgument>(oprd))
          continue;
        oprd.getDefiningOp()->moveBefore(stickAllocOp);
      }
      // Replace store's Memref and Value, and preserve the access indices.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(storeOp);
      // Convert the value to dlf16.
      Value dlf16 =
          rewriter.create<ZLowDummyOp>(loc, stickifiedElementType, storeValue);
      // Clone storeOp with new Memref and Value, which preserves the access
      // indices.
      IRMapping operandMap;
      operandMap.map(storeOp.getMemref(), stickMemref);
      operandMap.map(storeOp.getValue(), dlf16);
      Operation *clonedOp = rewriter.clone(*storeOp.getOperation(), operandMap);
      rewriter.eraseOp(storeOp);
    }

    rewriter.eraseOp(unstickOp);
    return success();
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

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<StickViewUnstickRemovalPattern>(&getContext());
    patterns.insert<UnstickLoadStoreStickRemovalPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createZLowRewritePass() {
  return std::make_unique<ZLowRewritePass>();
}

} // namespace zlow
} // namespace onnx_mlir
