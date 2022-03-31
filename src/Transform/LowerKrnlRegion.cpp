/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- LowerKrnlRegion.cpp ------------------------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This pass enables the lowering of the krnl.shape operation to use Shape
// dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;
using namespace onnx_mlir::krnl;

namespace {

/*!
 Move the ops in KrnlRegionOp out of its region and erase KrnlRegionOp
 */

class LowerKrnlRegion : public OpRewritePattern<KrnlRegionOp> {
public:
  using OpRewritePattern<KrnlRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlRegionOp krnlRegionOp, PatternRewriter &rewriter) const override {

    auto loc = krnlRegionOp.getLoc();
    MultiDialectBuilder<MathBuilder> create(rewriter,loc);
    
    Block &regionBlock = krnlRegionOp.bodyRegion().front();

    // use the special traversal because the op is modified in the sametime
    for (Operation &op : llvm::make_early_inc_range(regionBlock)) {
      op.moveBefore(krnlRegionOp);
    }

    rewriter.eraseOp(krnlRegionOp);
    return success();
  }
};

/*!
 *  Function pass that lowers KrnlRgionOp
 */
class LowerKrnlRegionPass
    : public PassWrapper<LowerKrnlRegionPass, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const override { return "lower-krnl-region"; }

  StringRef getDescription() const override {
    return "Move ops in krnl.region operation out and erase this op";
  }

  void runOnOperation() override {
    auto function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<LowerKrnlRegion>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace onnx_mlir{
namespace krnl{
std::unique_ptr<Pass> createLowerKrnlRegionPass() {
  return std::make_unique<LowerKrnlRegionPass>();
}
} // namespace krnl
} // namespace onnx_mlir
