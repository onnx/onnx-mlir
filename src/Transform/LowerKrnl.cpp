//===-------------- LowerKrnl.cpp - Krnl Dialect Lowering -----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlIterate operation.
//===----------------------------------------------------------------------===//

struct KrnlIterateOpLowering : public OpRewritePattern<KrnlIterateOp> {
  using OpRewritePattern<KrnlIterateOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(KrnlIterateOp iterateOp,
                                     PatternRewriter &rewriter) const override {
    auto boundMapAttrs =
        iterateOp.getAttrOfType<ArrayAttr>(KrnlIterateOp::getBoundsAttrName())
            .getValue();
    auto operandItr =
        iterateOp.operand_begin() + iterateOp.getNumOptimizedLoops();
    SmallVector<AffineForOp, 4> nestedForOps;
    for (size_t boundIdx = 0; boundIdx < boundMapAttrs.size(); boundIdx += 2) {
      // Consume input loop operand, currently do not do anything with it.
      operandItr++;

      // Organize operands into lower/upper bounds in affine.for ready formats.
      SmallVector<Value, 4> lbOperands, ubOperands;
      AffineMap lbMap, ubMap;
      for (int boundType = 0; boundType < 2; boundType++) {
        auto &operands = boundType == 0 ? lbOperands : ubOperands;
        auto &map = boundType == 0 ? lbMap : ubMap;
        map = boundMapAttrs[boundIdx + boundType]
                  .cast<AffineMapAttr>()
                  .getValue();
        operands.insert(operands.end(), operandItr,
                        operandItr + map.getNumInputs());
        std::advance(operandItr, map.getNumInputs());
      }

      nestedForOps.emplace_back(rewriter.create<AffineForOp>(
          iterateOp.getLoc(), lbOperands, lbMap, ubOperands, ubMap));
      rewriter.setInsertionPoint(nestedForOps.back().getBody(),
                                 nestedForOps.back().getBody()->begin());
    }

    // Replace induction variable references from those introduced by a
    // single krnl.iterate to those introduced by multiple affine.for
    // operations.
    for (size_t i = 0; i < nestedForOps.size() - 1; i++) {
      auto iterateIV = iterateOp.bodyRegion().front().getArgument(0);
      auto forIV = nestedForOps[i].getBody()->getArgument(0);
      iterateIV.replaceAllUsesWith(forIV);
      iterateOp.bodyRegion().front().eraseArgument(0);
    }

    // Pop krnl.iterate body region block arguments, leave the last one
    // for convenience (it'll be taken care of by region inlining).
    while (iterateOp.bodyRegion().front().getNumArguments() > 1)
      iterateOp.bodyRegion().front().eraseArgument(0);

    // Transfer krnl.iterate region to innermost for op.
    auto innermostForOp = nestedForOps.back();
    innermostForOp.region().getBlocks().clear();
    rewriter.inlineRegionBefore(iterateOp.bodyRegion(), innermostForOp.region(),
                                innermostForOp.region().end());

    rewriter.eraseOp(iterateOp);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlTerminator operation.
//===----------------------------------------------------------------------===//

class KrnlTerminatorLowering : public OpRewritePattern<KrnlTerminatorOp> {
public:
  using OpRewritePattern<KrnlTerminatorOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(KrnlTerminatorOp op,
                                     PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AffineTerminatorOp>(op);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlDefineLoops operation.
//===----------------------------------------------------------------------===//

class KrnlDefineLoopsLowering : public OpRewritePattern<KrnlDefineLoopsOp> {
public:
  using OpRewritePattern<KrnlDefineLoopsOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(KrnlDefineLoopsOp op,
                                     PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlOptimizeLoops operation.
//===----------------------------------------------------------------------===//

class KrnlOptimizeLoopsLowering : public OpRewritePattern<KrnlOptimizeLoopsOp> {
public:
  using OpRewritePattern<KrnlOptimizeLoopsOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(KrnlOptimizeLoopsOp op,
                                     PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// KrnlToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the krnl dialect operations.
/// At this stage the dialect will contain standard operations as well like
/// add and multiply, this pass will leave these operations intact.
namespace {
struct KrnlToAffineLoweringPass
    : public FunctionPass<KrnlToAffineLoweringPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void KrnlToAffineLoweringPass::runOnFunction() {
  auto function = getFunction();

  ConversionTarget target(getContext());

  target.addLegalDialect<AffineOpsDialect, StandardOpsDialect>();
  // We expect IR to be free of Krnl Dialect Ops.
  target.addIllegalDialect<KrnlOpsDialect>();
  target.addLegalOp<KrnlMemcpyOp>();
  target.addLegalOp<KrnlEntryPointOp>();

  OwningRewritePatternList patterns;
  patterns.insert<KrnlIterateOpLowering, KrnlTerminatorLowering,
                  KrnlDefineLoopsLowering, KrnlOptimizeLoopsLowering>(
      &getContext());

  if (failed(applyPartialConversion(getFunction(), target, patterns))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createLowerKrnlPass() {
  return std::make_unique<KrnlToAffineLoweringPass>();
}

static PassRegistration<KrnlToAffineLoweringPass> pass("lower-krnl",
                                                       "Lower Krnl dialect.");
