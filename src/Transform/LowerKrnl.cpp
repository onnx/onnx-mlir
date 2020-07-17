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
#include <set>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

void lowerIterateOp(KrnlIterateOp &iterateOp, OpBuilder &builder,
    llvm::SmallDenseMap<Value, AffineForOp, 4> &refToOps) {
  builder.setInsertionPointAfter(iterateOp);
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
        unoptimizedLoopRef, builder.create<AffineForOp>(iterateOp.getLoc(),
                                lbOperands, lbMap, ubOperands, ubMap)));

    builder.setInsertionPoint(currentNestedForOps.back().second.getBody(),
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
    builder.setInsertionPointAfter(iterateOp);
    iterateOp.bodyRegion().walk([&](Operation *op) {
      if (!op->isKnownTerminator())
        op->replaceAllUsesWith(builder.clone(*op));
    });
  } else {
    // Transfer krnl.iterate region to innermost for op.
    auto innermostForOp = currentNestedForOps.back().second;
    innermostForOp.region().getBlocks().clear();
    auto &innerMostRegion = innermostForOp.region();

    // Convert Krnl Terminator to Affine Terminator.
    for (auto &block : iterateOp.bodyRegion().getBlocks()) {
      auto krnlTerm = block.getOps<KrnlTerminatorOp>();
      if (!krnlTerm.empty()) {
        KrnlTerminatorOp termOp = *krnlTerm.begin();
        //        builder.setInsertionPoint(termOp);
        //        builder.create<AffineTerminatorOp>(iterateOp.getLoc());
        termOp.erase();
      }
    }

    AffineForOp::ensureTerminator(
        iterateOp.bodyRegion(), builder, iterateOp.getLoc());
    innerMostRegion.getBlocks().splice(
        innerMostRegion.end(), iterateOp.bodyRegion().getBlocks());
  }

  iterateOp.erase();
  for (const auto &pair : currentNestedForOps)
    refToOps[pair.first] = pair.second;
}

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlTerminator operation.
//===----------------------------------------------------------------------===//

class KrnlTerminatorLowering : public OpRewritePattern<KrnlTerminatorOp> {
public:
  using OpRewritePattern<KrnlTerminatorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlTerminatorOp op, PatternRewriter &rewriter) const override {
    auto parentOp = op.getParentOp();
    op.erase();
    rewriter.eraseOp(op);
    //    rewriter.replaceOpWithNewOp<AffineTerminatorOp>(op);

    //    printf("replaced!");
    fprintf(stderr, "replaced!\n");
    parentOp->dump();
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

bool interpretOperation(Operation *op, OpBuilder &builder,
    llvm::SmallDenseMap<Value, AffineForOp, 4> &loopRefToOp) {
  if (isa<KrnlDefineLoopsOp>(op)) {
    auto usersItr = op->getUsers();
    std::set<Operation *> iterateOps;
    std::copy_if(usersItr.begin(), usersItr.end(),
        std::inserter(iterateOps, iterateOps.begin()),
        [](Operation *user) { return isa<KrnlIterateOp>(user); });

    if (!iterateOps.empty()) {
      for (auto iterateOp : iterateOps) {
        auto castedIterateOp = dyn_cast<KrnlIterateOp>(iterateOp);
        lowerIterateOp(castedIterateOp, builder, loopRefToOp);
      }
      return true;
    }
  } else if (auto blockOp = dyn_cast_or_null<KrnlBlockOp>(op)) {
    SmallVector<AffineForOp, 2> tiledLoops;
    SmallVector<AffineForOp, 1> loopsToTile = {loopRefToOp[blockOp.loop()]};
    if (failed(tilePerfectlyNested(
            loopsToTile, blockOp.tile_sizeAttr().getInt(), &tiledLoops))) {
      fprintf(stderr, "Error!\n");
      exit(1);
      //      return signalPassFailure();
    }
    assert(tiledLoops.size() == 2);
    assert(blockOp.getNumResults() == 2);
    // Record the tiled loop references, and their corresponding tiled
    // for loops in loopRefToLoop.
    loopRefToOp[blockOp.getResult(0)] = tiledLoops[0];
    loopRefToOp[blockOp.getResult(1)] = tiledLoops[1];
    blockOp.erase();
    return true;
  } else if (auto permuteOp = dyn_cast_or_null<KrnlPermuteOp>(op)) {
    SmallVector<AffineForOp, 4> loopsToPermute;
    std::transform(permuteOp.operand_begin(), permuteOp.operand_end(),
        std::back_inserter(loopsToPermute),
        [&](const Value &val) { return loopRefToOp[val]; });

    SmallVector<unsigned int, 4> permuteMap;
    for (const auto &attr : permuteOp.map().getAsRange<IntegerAttr>())
      permuteMap.emplace_back(attr.getValue().getSExtValue());

    permuteLoops(loopsToPermute, permuteMap);

    permuteOp.erase();
    return true;
  }

  for (auto &region : op->getRegions())
    for (auto &block : region.getBlocks()) {
      bool requireRevisit;
      do {
        requireRevisit = false;
        block.walk([&](Operation *next) {
          requireRevisit |= interpretOperation(next, builder, loopRefToOp);
          if (requireRevisit)
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
      } while (requireRevisit);
    }

  return false;
}

void KrnlToAffineLoweringPass::runOnFunction() {
  OpBuilder builder(&getContext());

  mlir::Operation *funcOp = getFunction();
  llvm::SmallDenseMap<Value, AffineForOp, 4> loopRefToOp;
  interpretOperation(funcOp, builder, loopRefToOp);
}
} // namespace

std::unique_ptr<Pass> mlir::createLowerKrnlPass() {
  return std::make_unique<KrnlToAffineLoweringPass>();
}
