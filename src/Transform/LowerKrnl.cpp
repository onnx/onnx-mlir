//===-------------- LowerKrnl.cpp - Krnl Dialect Lowering -----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#include <map>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

void lowerIterateOp(KrnlIterateOp &iterateOp, OpBuilder &rewriter,
    SmallVector<std::pair<Value, AffineForOp>, 4> &nestedForOps) {
  rewriter.setInsertionPointAfter(iterateOp);
  SmallVector<std::pair<Value, AffineForOp>, 4> currentNestedForOps;
  auto boundMapAttrs =
      iterateOp.getAttrOfType<ArrayAttr>(KrnlIterateOp::getBoundsAttrName())
          .getValue();
  auto operandItr =
      iterateOp.operand_begin() + iterateOp.getNumOptimizedLoops();
  for (size_t boundIdx = 0; boundIdx < boundMapAttrs.size(); boundIdx += 2) {
    // Consume input loop operand, currently do not do anything with it.
    auto unoptimizedLoopRef = *(operandItr++);

    // Organize operands into lower/upper bounds in affine.for ready formats.
    llvm::SmallVector<Value, 4> lbOperands, ubOperands;
    AffineMap lbMap, ubMap;
    for (int boundType = 0; boundType < 2; boundType++) {
      auto &operands = boundType == 0 ? lbOperands : ubOperands;
      auto &map = boundType == 0 ? lbMap : ubMap;
      map =
          boundMapAttrs[boundIdx + boundType].cast<AffineMapAttr>().getValue();
      operands.insert(
          operands.end(), operandItr, operandItr + map.getNumInputs());
      std::advance(operandItr, map.getNumInputs());
    }
    currentNestedForOps.emplace_back(std::make_pair(
        unoptimizedLoopRef, rewriter.create<AffineForOp>(iterateOp.getLoc(),
                                lbOperands, lbMap, ubOperands, ubMap)));

    rewriter.setInsertionPoint(currentNestedForOps.back().second.getBody(),
        currentNestedForOps.back().second.getBody()->begin());
  }

  // Replace induction variable references from those introduced by a
  // single krnl.iterate to those introduced by multiple affine.for
  // operations.
  for (int64_t i = 0; i < (int64_t)currentNestedForOps.size() - 1; i++) {
    auto iterateIV = iterateOp.bodyRegion().front().getArgument(0);
    auto forIV = currentNestedForOps[i].second.getBody()->getArgument(0);
    iterateIV.replaceAllUsesWith(forIV);
    iterateOp.bodyRegion().front().eraseArgument(0);
  }

  // Pop krnl.iterate body region block arguments, leave the last one
  // for convenience (it'll be taken care of by region inlining).
  while (iterateOp.bodyRegion().front().getNumArguments() > 1)
    iterateOp.bodyRegion().front().eraseArgument(0);

  if (currentNestedForOps.empty()) {
    // If no loops are involved, simply move operations from within iterateOp
    // body region to the parent region of iterateOp.
    rewriter.setInsertionPointAfter(iterateOp);
    iterateOp.bodyRegion().walk([&](Operation *op) {
      if (!op->isKnownTerminator())
        op->replaceAllUsesWith(rewriter.clone(*op));
    });
  } else {
    // Transfer krnl.iterate region to innermost for op.
    auto innermostForOp = currentNestedForOps.back().second;
    innermostForOp.region().getBlocks().clear();
    auto &innerMostRegion = innermostForOp.region();
    innerMostRegion.getBlocks().splice(
        innerMostRegion.end(), iterateOp.bodyRegion().getBlocks());
  }

  iterateOp.erase();
  nestedForOps.insert(nestedForOps.end(), currentNestedForOps.begin(),
      currentNestedForOps.end());
}

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlTerminator operation.
//===----------------------------------------------------------------------===//

class KrnlTerminatorLowering : public OpRewritePattern<KrnlTerminatorOp> {
public:
  using OpRewritePattern<KrnlTerminatorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlTerminatorOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AffineTerminatorOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlDefineLoops operation.
//===----------------------------------------------------------------------===//

class KrnlDefineLoopsLowering : public OpRewritePattern<KrnlDefineLoopsOp> {
public:
  using OpRewritePattern<KrnlDefineLoopsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlDefineLoopsOp op, PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlBlock operation.
//===----------------------------------------------------------------------===//

class KrnlBlockOpLowering : public OpRewritePattern<KrnlBlockOp> {
public:
  using OpRewritePattern<KrnlBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlBlockOp op, PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlPermute operation.
//===----------------------------------------------------------------------===//

class KrnlPermuteOpLowering : public OpRewritePattern<KrnlPermuteOp> {
public:
  using OpRewritePattern<KrnlPermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlPermuteOp op, PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
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
    : public PassWrapper<KrnlToAffineLoweringPass, FunctionPass> {
  void runOnFunction() final;
};

// Helper function to test if KrnlIterateOp is nested under another
// KrnlIterateOp.
bool isIterateOpNested(KrnlIterateOp iterateOp) {
  // krnl.iterate is dynamically legal, if and only if it is enclosed by
  // another krnl.iterate.
  Operation *op = iterateOp;
  while ((op = op->getParentOp()))
    if (auto parentOp = dyn_cast<KrnlIterateOp>(op))
      return true;
  return false;
}

Optional<KrnlIterateOp> nextIterateOp(FuncOp function) {
  Optional<KrnlIterateOp> nextIterateOp;
  function.walk([&](KrnlIterateOp op) {
    if (!isIterateOpNested(op))
      nextIterateOp = op;
  });
  return nextIterateOp;
}

bool hasOnePerfectlyNestedIterateOp(KrnlIterateOp op) {
  auto childrenOps = op.bodyRegion().getOps();
  auto childrenOpsIter = childrenOps.begin();
  if (childrenOpsIter == childrenOps.end() ||
      !isa<KrnlIterateOp>(*childrenOpsIter))
    return false;
  if (++childrenOpsIter == childrenOps.end() ||
      !(*childrenOpsIter).isKnownTerminator())
    return false;
  return true;
}
} // end anonymous namespace.

void KrnlToAffineLoweringPass::runOnFunction() {
  auto function = getFunction();
  ConversionTarget target(getContext());

  target.addLegalDialect<AffineDialect, StandardOpsDialect>();
  // We expect IR to be free of Krnl Dialect Ops.
  target.addIllegalDialect<KrnlOpsDialect>();

  // Operations that should be converted to LLVM IRs directly.
  target.addLegalOp<KrnlMemcpyOp>();
  target.addLegalOp<KrnlEntryPointOp>();
  target.addLegalOp<KrnlGlobalOp>();
  target.addLegalOp<KrnlGetRefOp>();
  target.addLegalOp<KrnlIterateOp>();

  OwningRewritePatternList patterns;
  patterns.insert<KrnlTerminatorLowering, KrnlDefineLoopsLowering,
      KrnlBlockOpLowering, KrnlPermuteOpLowering>(&getContext());

  // Do not lower operations that pertain to schedules just yet.
  target.addLegalOp<KrnlBlockOp>();
  target.addLegalOp<KrnlPermuteOp>();
  target.addLegalOp<KrnlDefineLoopsOp>();
  if (failed(applyPartialConversion(function, target, patterns)))
    return signalPassFailure();

  OpBuilder builder(&getContext());
  while (auto iterateOp = nextIterateOp(function)) {
    // Collect a maximal set of loop band to lower. They must be a perfectly
    // nested sequence of for loops (this limitation follows from the
    // precondition of current loop manupulation utility libraries).
    auto rootOp = iterateOp;
    SmallVector<KrnlIterateOp, 4> loopBand = {*rootOp};
    while (hasOnePerfectlyNestedIterateOp(*rootOp)) {
      auto nestedIterateOp =
          *rootOp->bodyRegion().getOps<KrnlIterateOp>().begin();
      loopBand.emplace_back(nestedIterateOp);
      rootOp = nestedIterateOp;
    }

    // Lower the band of iterateOps, initialize loopRefToLoop to be the list of
    // loop reference and the for loop being referenced.
    SmallVector<std::pair<Value, AffineForOp>, 4> loopRefToLoop;
    for (auto op : loopBand)
      lowerIterateOp(op, builder, loopRefToLoop);

    // Manually lower schedule ops.
    while (!loopRefToLoop.empty()) {
      Value loopRef;
      AffineForOp forOp;
      std::tie(loopRef, forOp) = loopRefToLoop.pop_back_val();

      // Get user operations excluding krnl.iterate. These operations will be
      // scheduling primitives.
      auto loopRefUsers = loopRef.getUsers();
      SmallVector<Operation *, 4> unfilteredUsers(
          loopRefUsers.begin(), loopRefUsers.end()),
          users;
      std::copy_if(unfilteredUsers.begin(), unfilteredUsers.end(),
          std::back_inserter(users),
          [](Operation *op) { return !isa<KrnlIterateOp>(op); });

      // No schedule primitives associated with this loop reference, move on.
      if (users.empty())
        continue;

      // Scheduling operations detected, transform loops as directed, while
      // keeping the loopRefToLoop mapping up-to-date.
      auto user = users.front();
      if (isa<KrnlBlockOp>(user)) {
        auto blockOp = cast<KrnlBlockOp>(user);
        SmallVector<AffineForOp, 2> tiledLoops;
        SmallVector<AffineForOp, 1> loopsToTile = {forOp};
        if (failed(tilePerfectlyNested(loopsToTile,
                cast<KrnlBlockOp>(user).tile_sizeAttr().getInt(),
                &tiledLoops))) {
          return signalPassFailure();
        }
        assert(tiledLoops.size() == 2);
        assert(blockOp.getNumResults() == 2);
        // Record the tiled loop references, and their corresponding tiled for
        // loops in loopRefToLoop.
        loopRefToLoop.emplace_back(
            std::make_pair(blockOp.getResult(0), tiledLoops[0]));
        loopRefToLoop.emplace_back(
            std::make_pair(blockOp.getResult(1), tiledLoops[1]));
      } else if (isa<KrnlPermuteOp>(user)) {
        auto permuteOp = cast<KrnlPermuteOp>(user);

        // Re-insert loop reference back for convenience.
        loopRefToLoop.insert(
            loopRefToLoop.begin(), std::make_pair(loopRef, forOp));
        llvm::SmallDenseMap<Value, AffineForOp, 4> map(
            loopRefToLoop.begin(), loopRefToLoop.end());

        bool allReferredLoopsScheduled =
            std::all_of(permuteOp.operand_begin(), permuteOp.operand_end(),
                [&](const Value &val) { return map.count(val) > 0; });

        if (!allReferredLoopsScheduled)
          continue;

        SmallVector<AffineForOp, 4> loopsToPermute;
        std::transform(permuteOp.operand_begin(), permuteOp.operand_end(),
            std::back_inserter(loopsToPermute),
            [&](const Value &val) { return map[val]; });

        SmallVector<unsigned int, 4> permuteMap;
        for (const auto &attr : permuteOp.map().getAsRange<IntegerAttr>())
          permuteMap.emplace_back(attr.getValue().getSExtValue());

        permuteLoops(loopsToPermute, permuteMap);
        auto remoteItr = std::remove_if(
            loopRefToLoop.begin(), loopRefToLoop.end(), [&](const auto &pair) {
              return std::find_if(permuteOp.operand_begin(),
                         permuteOp.operand_end(), [&](const Value &loopRef) {
                           return loopRef == pair.first;
                         }) != permuteOp.operand_end();
            });
        loopRefToLoop.erase(remoteItr, loopRefToLoop.end());
      }
    }
  }

  // KrnlIterateOp should be all gone by now.
  target.addIllegalOp<KrnlIterateOp>();

  // Remove/lower schedule related operations.
  target.addIllegalOp<KrnlDefineLoopsOp>();
  target.addIllegalOp<KrnlBlockOp>();
  target.addIllegalOp<KrnlPermuteOp>();
  if (failed(applyPartialConversion(function, target, patterns)))
    return signalPassFailure();
}
} // namespace

std::unique_ptr<Pass> mlir::createLowerKrnlPass() {
  return std::make_unique<KrnlToAffineLoweringPass>();
}
