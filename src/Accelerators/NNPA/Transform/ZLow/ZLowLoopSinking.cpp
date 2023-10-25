/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZLowLoopSinkingPass.cpp - ZLow Rewrite Patterns -------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This pass sinks zlow.attach_layout and zlow.detach operations into affine-for
// loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

// clang-format off

/// This pattern rewrites
/// ```mlir
/// %in = "zlow.detach_layout"(%arg0) : (memref<5x7xf16, #map>) -> memref<5x7xf32>)
/// %out = memref.alloc(%dim) {alignment = 16 : i64} : memref<5x7xf32>
/// affine.for %arg1 = 0 to 5 {
///   affine.for %arg2 = 0 to 7 {
///     %0 = affine.load %in[%arg1, %arg2] : memref<5x7xf32>
///     %1 = math.sqrt %0 : f32 
///     affine.store %1, %out[%arg1, %arg2] : memref<5x7xf32>
///   }
/// }
/// 
/// into
/// 
/// %out = memref.alloc(%dim) {alignment = 16 : i64} : memref<5x7xf32>
/// affine.for %arg1 = 0 to 5 {
///   affine.for %arg2 = 0 to 7 {
///     %0 = affine.load %arg0[%arg1, %arg2] : memref<5x7xf16, #map>
///     %1 = "zlow.dlf16_to_f32"(%0) : (f16) -> f32
///     %2 = math.sqrt %1 : f32 
///     affine.store %2, %out[%arg1, %arg2] : memref<5x7xf32>
///   }
/// }
/// ```

// clang-format on

class SinkDetachLayoutPattern : public OpRewritePattern<ZLowDetachLayoutOp> {
public:
  using OpRewritePattern<ZLowDetachLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowDetachLayoutOp detachOp, PatternRewriter &rewriter) const override {
    Operation *op = detachOp.getOperation();
    Value input = detachOp.getInput();
    Value output = detachOp.getOutput();
    std::string layoutName = detachOp.getLayoutNameAttr().str();

    // 1. Match

    // Do not support layout 1D and 2DS since their access index functions are
    // incorrect: https://github.com/onnx/onnx-mlir/issues/1940
    if ((layoutName == LAYOUT_1D) || (layoutName == LAYOUT_2DS))
      return failure();

    // Collect affine.load operations for sinking.
    // Condition for transformation: all affine.load, memref.dim and
    // zlow.detach_layout are the ONLY users of the input, so that we can safely
    // remove zlow.detach_layout.
    SmallVector<affine::AffineLoadOp, 4> affineLoads;
    for (Operation *user : output.getUsers()) {
      if (user == op)
        continue;
      if (llvm::isa<memref::DimOp>(user))
        continue;
      if (auto affineLoad = llvm::dyn_cast<affine::AffineLoadOp>(user))
        affineLoads.emplace_back(affineLoad);
      else
        return failure();
    }
    if (affineLoads.size() == 0)
      return failure();

    // 2. Rewrite
    // Replace all uses of the identity-layout MemRef by the non-identity-layout
    // MemRef.
    output.replaceAllUsesWith(input);

    return success();
  }
};

class SinkAttachLayoutPattern : public OpRewritePattern<ZLowAttachLayoutOp> {
public:
  using OpRewritePattern<ZLowAttachLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowAttachLayoutOp attachOp, PatternRewriter &rewriter) const override {
    Operation *op = attachOp.getOperation();
    Value input = attachOp.getInput();
    Value output = attachOp.getOutput();
    std::string layoutName = attachOp.getLayoutNameAttr().str();

    // 1. Match

    // Input is from memref.alloc.
    if (input.isa<BlockArgument>())
      return failure();
    Operation *allocOp = input.getDefiningOp();
    if (!isa<memref::AllocOp>(allocOp))
      return failure();

    // Do not support layout 1D and 2DS since their access index functions are
    // incorrect: https://github.com/onnx/onnx-mlir/issues/1940
    if ((layoutName == LAYOUT_1D) || (layoutName == LAYOUT_2DS))
      return failure();

    // Collect affine.store operations for sinking.
    // Condition for transformation: all affine.store and zlow.attach_layout are
    // the ONLY users of the input, so that we can safely remove
    // zlow.attach_layout.
    SmallVector<affine::AffineStoreOp, 4> affineStores;
    for (Operation *user : input.getUsers()) {
      if (user == op)
        continue;
      if (auto affineStore = llvm::dyn_cast<affine::AffineStoreOp>(user))
        affineStores.emplace_back(affineStore);
      else
        return failure();
    }
    if (affineStores.size() == 0)
      return failure();

    // 2. Rewrite
    // Attach layout directly to the input MemRef, then remove
    // zlow.attach_layout.
    allocOp->getResult(0).setType(output.getType());
    output.replaceAllUsesWith(allocOp->getResult(0));

    return success();
  }
};

/*!
 *  Function pass that optimizes ZLowIR.
 */
class ZLowLoopSinkingPass
    : public PassWrapper<ZLowLoopSinkingPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "zlow-loop-sinking"; }

  StringRef getDescription() const override {
    return "This pass sinks zlow.attach_layout and zlow.detach operations into "
           "affine-for loops";
  }

  void runOnOperation() override {
    Operation *function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<SinkDetachLayoutPattern>(&getContext());
    patterns.insert<SinkAttachLayoutPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createZLowLoopSinkingPass() {
  return std::make_unique<ZLowLoopSinkingPass>();
}

} // namespace zlow
} // namespace onnx_mlir
