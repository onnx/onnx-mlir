/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-----------LowerKrnlRegion.cpp ---------------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This pass enables the lowering of the krnl.region operation
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;
using namespace onnx_mlir::krnl;

namespace {

/*!
 Move the ops in KrnlRegionOp out of its region and then erase KrnlRegionOp
 */

class LowerKrnlRegion : public OpRewritePattern<KrnlRegionOp> {
public:
  using OpRewritePattern<KrnlRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlRegionOp krnlRegionOp, PatternRewriter &rewriter) const override {

    // Special traversal is used because the op being traversed is moved.
    Block &regionBlock = krnlRegionOp.getBodyRegion().front();

    for (Operation &op : llvm::make_early_inc_range(regionBlock)) {
      op.moveBefore(krnlRegionOp);
    }

    rewriter.eraseOp(krnlRegionOp);
    return success();
  }
};

/*!
 *  Function pass that lowers KrnlRegionOp
 */
class LowerKrnlRegionPass
    : public PassWrapper<LowerKrnlRegionPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKrnlRegionPass)

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

namespace onnx_mlir {
namespace krnl {
std::unique_ptr<Pass> createLowerKrnlRegionPass() {
  return std::make_unique<LowerKrnlRegionPass>();
}
} // namespace krnl
} // namespace onnx_mlir
