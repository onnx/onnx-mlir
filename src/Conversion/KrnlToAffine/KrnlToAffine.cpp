/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- LowerKrnl.cpp - Krnl Dialect Lowering -----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;

// Since Krnl Dialect allows optimizations to be specified without being
// applied, some IR may be placed under Krnl loops that will not be materialized
// concretely until relevant optimizations have been applied. Consider the
// following example, where we specify the recepie for a 2-dimensional tiled
// loop, and insert memory allocation/deallocation for per-tile temporary
// buffer:
//
// %ii, %ij = krnl.define_loops 2
// %ib, %il = krnl.block %ii 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// %jb, %jl = krnl.block %ij 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// krnl.permute(%ib, %il, %jb, %jl) [0, 2, 1, 3] : !krnl.loop, !krnl.loop,
//     !krnl.loop, !krnl.loop
// krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 20) {
//   %alloc = alloc() : memref<10 x f32>
//   krnl.iterate(%il, %jl) with () {
//     %foo = addi %i, %j : index
//   }
//   dealloc %alloc : memref<10 x f32>
//  }
//
// The temporary buffer allocation/deallocation are placed within loops that
// have yet to be materialized because loop tiling and loop permutation are only
// specified as recipes without actually being applied at Krnl Dialect level.
// The need to move IR between loop bodies is ubiquitous in the lowering of Krnl
// Dialect because of its lazy nature, and thus the need for LoopBodyMover.
//
// LoopBody mover registers, for each Krnl loop reference, a list of operations
// that should be contained directly beneath it as the moving plan in the
// beginning of the Krnl Dialect lowering. And subsequently, when the Affine for
// loop corresponding to the Krnl loop reference is materialized, IRs will be
// moved to appropriate locations based on information recorded as moving plan.
//
// Thus, for the above IR, the following moving plan is registered:
// - For %ib, %jb, the list of operation nested directly under is:
//    - alloc() operation,
//    - materialized loops corresponding to %il, %jl,
//    - dealloc() operation.
// - For %il, %jl, the list of operations nested directly under is:
//    - addi operation.
//
// Subsequently, lowering will start with affine ops materialized corresponding
// to the reference to un-optimized loops:
//
// affine.for %i = 0 to 10 {
//   affine.for %j = 0 to 20 {
//     %foo = addi %i, %j : index
//   }
// }
//
// Since the tiling has not taken place yet, tile coordinat iteration loops have
// not been materialized, therefore the alloc and dealloc operations do not fit
// in the IR presently yet. Instead, they will be placed within a krnl.movable
// op region, to indicate that their positioning is subject to change.
//
// krnl.movable {
//   %alloc = alloc() : memref<10 x f32>;
// }
// krnl.movable {
//   dealloc %alloc : memref<10 x f32>
// }
//
// As we lower the optimization recipes, outer loops will eventually manifest as
// affine loops. When the destination loops emerge, content within the
// krnl.movable op will be transferred to appropriate locations, too, resulting
// in the following final lowered IR:
//
// affine.for ib = 0 to 10 step 5 {
//   affine.for jb = 0 to 20 step 4 {
//     %alloc = alloc() : memref<10xf32>
//     affine.for %il = ... {
//       affine.for %jl = ... {
//         %foo = addi %il, %jl : index
//       }
//     }
//     dealloc %alloc : memref<10xf32>
//   }
// }
//
// As specified by the high-level Krnl Dialect.
class LoopBodyMover {
public:
  /*!
   * Represents (a list of) operations to be moved, or a particular set of loop
   * nests expected in the destination loop body.
   */
  struct Movable {
    llvm::Optional<KrnlMovableOp> movableOp;
    llvm::Optional<llvm::SmallVector<mlir::Value, 4>> loopsToSkip;

    explicit Movable(KrnlMovableOp op) : movableOp(op) {}
    explicit Movable(KrnlIterateOp op) {
      auto operandRange = op->getOperands();
      loopsToSkip = llvm::SmallVector<Value, 4>(operandRange.begin(),
          operandRange.begin() + op.getNumOptimizedLoops());
    }
  };

  void toMoveUnder(const Movable &movable, KrnlIterateOp loop) {
    Value innerMostLoopHandler =
        loop.getOperand(loop.getNumOptimizedLoops() - 1);
    movingPlan[innerMostLoopHandler].push_back(movable);
  }

  void moveOne(Value loopRef,
      llvm::SmallDenseMap<Value, AffineForOp, 4> &loopRefToOp,
      bool erase = true) {
    assert(loopRefToOp.count(loopRef) >= 0 &&
           "Can't find affine for operation associated with .");
    AffineForOp forOp = loopRefToOp[loopRef];
    Block &loopBody = forOp.getLoopBody().front();
    auto insertPt = loopBody.begin();

    auto opsToTransfer = movingPlan[loopRef];
    if (erase)
      movingPlan.erase(loopRef);

    for (Movable transferPt : opsToTransfer) {
      assert(insertPt != loopBody.end());
      assert(
          transferPt.loopsToSkip.hasValue() != transferPt.movableOp.hasValue());
      if (transferPt.movableOp.hasValue()) {
        auto movableOp = transferPt.movableOp.getValue();

        loopBody.getOperations().splice(insertPt,
            movableOp.getBody()->getOperations(), movableOp.getBody()->begin(),
            movableOp.getBody()->getTerminator()->getIterator());

        // After insertion, the insertion point iterator will remain valid
        // and points to the operation before which new operations can be
        // inserted, unless it happens to point to the extraction point, too
        // (aka, the movable op from which operations are drawn). In this
        // case, we increment it to its next operation. Notably, this has to
        // be done after the movable op is disconnected from the basic block.
        // Otherwise the iterator is invalidated and iterator increment
        // doesn't work anymore.
        if (insertPt == movableOp->getIterator())
          insertPt++;
        movableOp->erase();
      } else if (transferPt.loopsToSkip.hasValue()) {
        llvm::Optional<AffineForOp> loopToSkip;
        loopToSkip =
            transferPt.loopsToSkip.getValue().empty()
                ? loopToSkip
                : loopRefToOp[transferPt.loopsToSkip.getValue().front()];

        // Move iterator to point to the next AffineFor Op.
        while (insertPt != loopBody.end() &&
               !dyn_cast_or_null<AffineForOp>(&*insertPt)) {
          assert(dyn_cast_or_null<KrnlMovableOp>(&*insertPt));
          insertPt++;
        }

        // Assert that now insertion point points to the loop to skip.
        if (loopToSkip)
          assert(insertPt == loopToSkip.getValue()->getIterator());

        // Skip loop by incrementing insertion point.
        insertPt++;
      }
    }
  }

  void moveAll(llvm::SmallDenseMap<Value, AffineForOp, 4> &loopRefToOp) {
    for (const auto &pair : movingPlan)
      moveOne(pair.first, loopRefToOp, /*erase=*/false);
  }

private:
  llvm::DenseMap<mlir::Value, llvm::SmallVector<Movable, 4>> movingPlan;
};

namespace {

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlTerminator operation.
//===----------------------------------------------------------------------===//

class KrnlTerminatorLowering : public OpRewritePattern<KrnlTerminatorOp> {
public:
  using OpRewritePattern<KrnlTerminatorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlTerminatorOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AffineYieldOp>(op);
    return success();
  }
};

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
    // Consume input loop operand, at this stage, do not do anything with it.
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
    auto forOp = builder.create<AffineForOp>(
        iterateOp.getLoc(), lbOperands, lbMap, ubOperands, ubMap);

    currentNestedForOps.emplace_back(std::make_pair(unoptimizedLoopRef, forOp));
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
    auto *parentBlock = iterateOp->getBlock();
    auto &iterateOpEntryBlock = iterateOp.bodyRegion().front();
    // Transfer body region operations to parent region, without the terminator
    // op.
    parentBlock->getOperations().splice(iterateOp->getIterator(),
        iterateOpEntryBlock.getOperations(),
        iterateOpEntryBlock.front().getIterator(),
        iterateOpEntryBlock.getTerminator()->getIterator());
  } else {
    // Transfer krnl.iterate region to innermost for op.
    auto innermostForOp = currentNestedForOps.back().second;
    innermostForOp.region().getBlocks().clear();
    auto &innerMostRegion = innermostForOp.region();
    innerMostRegion.getBlocks().splice(
        innerMostRegion.end(), iterateOp.bodyRegion().getBlocks());
  }

  for (const auto &pair : currentNestedForOps)
    refToOps.try_emplace(pair.first, pair.second);
}

//===----------------------------------------------------------------------===//
// ConvertKrnlToAffinePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the krnl dialect operations.
/// At this stage the dialect will contain standard operations as well like
/// add and multiply, this pass will leave these operations intact.
namespace {
struct ConvertKrnlToAffinePass
    : public PassWrapper<ConvertKrnlToAffinePass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

LogicalResult interpretOperation(Operation *op, OpBuilder &builder,
    llvm::SmallDenseMap<Value, AffineForOp, 4> &loopRefToOp,
    llvm::SmallPtrSetImpl<Operation *> &opsToErase, LoopBodyMover &mover) {
  // Recursively interpret nested operations.
  for (auto &region : op->getRegions())
    for (auto &block : region.getBlocks()) {
      auto &blockOps = block.getOperations();
      for (auto itr = blockOps.begin(); itr != blockOps.end();)
        if (failed(interpretOperation(
                &(*itr), builder, loopRefToOp, opsToErase, mover))) {
          return failure();
        } else {
          ++itr;
        }
    }

  if (auto defineOp = dyn_cast_or_null<KrnlDefineLoopsOp>(op)) {
    // Collect users of defineLoops operations that are iterate operations.
    std::vector<KrnlIterateOp> iterateOps;
    for (auto result : op->getResults())
      for (auto *user : result.getUsers())
        if (auto iterateOp = dyn_cast_or_null<KrnlIterateOp>(user))
          if (std::find(iterateOps.begin(), iterateOps.end(), iterateOp) ==
              iterateOps.end())
            iterateOps.push_back(dyn_cast<KrnlIterateOp>(user));

    // Lower iterate operations and record the mapping between loop references
    // and affine for loop operations in loopRefToOp map.
    if (!iterateOps.empty()) {
      for (auto opToLower : iterateOps) {
        if (opsToErase.count(opToLower) == 0) {
          lowerIterateOp(opToLower, builder, loopRefToOp);
          opsToErase.insert(opToLower);
        }
      }
    }
    opsToErase.insert(op);
    return success();
  } else if (auto iterateOp = dyn_cast_or_null<KrnlIterateOp>(op)) {
    // If an iterateOp has no unoptimized loop references, then we need to lower
    // them manually.
    if (opsToErase.count(op) == 0) {
      lowerIterateOp(iterateOp, builder, loopRefToOp);
      opsToErase.insert(iterateOp);
    }
    return success();
  } else if (auto blockOp = dyn_cast_or_null<KrnlBlockOp>(op)) {
    SmallVector<AffineForOp, 2> tiledLoops;
    SmallVector<AffineForOp, 1> loopsToTile = {loopRefToOp[blockOp.loop()]};

    if (failed(tilePerfectlyNested(
            loopsToTile, blockOp.tile_sizeAttr().getInt(), &tiledLoops))) {
      return failure();
    }
    assert(tiledLoops.size() == 2);
    assert(blockOp.getNumResults() == 2);

    // Record the tiled loop references, and their corresponding tiled
    // for loops in loopRefToLoop.
    loopRefToOp[blockOp.getResult(0)] = tiledLoops[0];
    loopRefToOp[blockOp.getResult(1)] = tiledLoops[1];

    opsToErase.insert(op);
    return success();
  } else if (auto permuteOp = dyn_cast_or_null<KrnlPermuteOp>(op)) {
    // Collect loops to permute.
    SmallVector<AffineForOp, 4> loopsToPermute;
    std::transform(permuteOp.operand_begin(), permuteOp.operand_end(),
        std::back_inserter(loopsToPermute),
        [&](const Value &val) { return loopRefToOp[val]; });

    // Construct permutation map from integer array attribute.
    SmallVector<unsigned int, 4> permuteMap;
    for (const auto &attr : permuteOp.map().getAsRange<IntegerAttr>())
      permuteMap.emplace_back(attr.getValue().getSExtValue());

    // Perform loop permutation.
    permuteLoops(loopsToPermute, permuteMap);

    opsToErase.insert(op);
    return success();
  } else if (auto unrollOp = dyn_cast_or_null<KrnlUnrollOp>(op)) {
    // Unroll the affine for loop fully.
    auto loopRef = unrollOp.loop();
    auto loopToUnroll = loopRefToOp[loopRef];
    mover.moveOne(loopRef, loopRefToOp);

    // Assert that there's no floating code within the loop to be unrolled.
    loopToUnroll.walk([](KrnlMovableOp op) {
      llvm_unreachable("Loop to unroll must not contain movable op.");
    });
    loopUnrollFull(loopToUnroll);

    opsToErase.insert(op);
    return success();
  } else if (auto convertOp =
                 dyn_cast_or_null<KrnlGetInductionVariableValueOp>(op)) {
    auto zippedOperandsResults = llvm::zip(op->getOperands(), op->getResults());
    for (const auto &operandAndResult : zippedOperandsResults) {
      auto operand = std::get<0>(operandAndResult);
      auto result = std::get<1>(operandAndResult);
      result.replaceAllUsesWith(loopRefToOp[operand].getInductionVar());
    }
    opsToErase.insert(op);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlLoad operation.
//===----------------------------------------------------------------------===//

/// KrnlLoad will be lowered to std.load or affine.load, depending on whether
/// the access indices are all affine maps or not.
class KrnlLoadLowering : public OpRewritePattern<KrnlLoadOp> {
public:
  using OpRewritePattern<KrnlLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlLoadOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    KrnlLoadOpAdaptor operandAdaptor = KrnlLoadOpAdaptor(op);

    // Prepare inputs.
    Value memref = operandAdaptor.memref();
    SmallVector<Value, 4> indices = operandAdaptor.indices();

    // Check whether all indices are affine maps or not.
    bool affineIndices =
        !llvm::any_of(indices, [](Value v) { return !isValidDim(v); });

    if (affineIndices)
      rewriter.replaceOpWithNewOp<AffineLoadOp>(op, memref, indices);
    else
      rewriter.replaceOpWithNewOp<LoadOp>(op, memref, indices);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlStore operation.
//===----------------------------------------------------------------------===//

/// KrnlStore will be lowered to std.store or affine.store, depending on whether
/// the access indices are all affine maps or not.
class KrnlStoreLowering : public OpRewritePattern<KrnlStoreOp> {
public:
  using OpRewritePattern<KrnlStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlStoreOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    KrnlStoreOpAdaptor operandAdaptor = KrnlStoreOpAdaptor(op);

    // Prepare inputs.
    Value value = operandAdaptor.value();
    Value memref = operandAdaptor.memref();
    SmallVector<Value, 4> indices = operandAdaptor.indices();

    // Check whether all indices are affine maps or not.
    bool affineIndices =
        !llvm::any_of(indices, [](Value v) { return !isValidDim(v); });

    if (affineIndices)
      rewriter.replaceOpWithNewOp<AffineStoreOp>(op, value, memref, indices);
    else
      rewriter.replaceOpWithNewOp<StoreOp>(op, value, memref, indices);

    return success();
  }
};

void makeBodyMovable(
    KrnlIterateOp root, OpBuilder builder, LoopBodyMover &mover) {
  auto &bodyRegion = root.bodyRegion();

  if (root.getNumOptimizedLoops() == 0)
    return;

  for (auto &block : bodyRegion.getBlocks()) {
    assert(!block.empty() && "IterateOp body block shouldn't be empty.");

    // Delimeter ops are delimeters of a movable chunk of code.
    llvm::SmallVector<Operation *> delimeterOps(block.getOps<KrnlIterateOp>());
    delimeterOps.push_back(block.getTerminator());
    Operation *movableBeginOp = &block.front();
    for (auto delimeterOp : delimeterOps) {
      Block::iterator movableBegin = movableBeginOp->getIterator();

      // If no op to extract, continue;
      if (movableBegin == delimeterOp->getIterator())
        continue;

      // Extract region & transfer them into a movable op.
      builder.setInsertionPoint(delimeterOp);
      auto movableOp = builder.create<KrnlMovableOp>(delimeterOp->getLoc());
      auto &movableRegion = movableOp.region();
      auto *entryBlock = new Block;
      movableRegion.push_back(entryBlock);
      entryBlock->getOperations().splice(entryBlock->end(),
          block.getOperations(), movableBegin, movableOp->getIterator());
      KrnlMovableOp::ensureTerminator(
          movableRegion, builder, delimeterOp->getLoc());

      mover.toMoveUnder(LoopBodyMover::Movable(movableOp), root);
      if (auto iterateOp = dyn_cast_or_null<KrnlIterateOp>(delimeterOp))
        mover.toMoveUnder(LoopBodyMover::Movable(iterateOp), root);

      movableBeginOp = delimeterOp->getNextNode();
    }
  }
}

void ConvertKrnlToAffinePass::runOnFunction() {
  OpBuilder builder(&getContext());
  FuncOp funcOp = getFunction();

  LoopBodyMover mover;
  funcOp.walk([&](KrnlIterateOp op) { makeBodyMovable(op, builder, mover); });

  // Interpret krnl dialect operations while looping recursively through
  // operations within the current function, note that erasing operations while
  // iterating is tricky because it can invalidate the iterator, so we collect
  // the operations to be erased in a small ptr set `opsToErase`, and only erase
  // after iteration completes.
  llvm::SmallDenseMap<Value, AffineForOp, 4> loopRefToOp;
  llvm::SmallPtrSet<Operation *, 4> opsToErase;
  if (failed(interpretOperation(
          funcOp, builder, loopRefToOp, opsToErase, mover))) {
    signalPassFailure();
    return;
  }

  // Remove lowered operations topologically; if ops are not removed
  // topologically, memory error will occur.
  size_t numOpsToRemove = opsToErase.size();
  // Given N operations to remove topologically, and that we remove
  // at least one operation during each pass through opsToErase, we
  // can only have a maximum of N passes through opsToErase.
  for (size_t i = 0; i < numOpsToRemove; i++) {
    for (auto op : opsToErase) {
      if (op->use_empty()) {
        op->erase();
        opsToErase.erase(op);
        // Restart, itr has been invalidated.
        break;
      }
    }
    if (opsToErase.empty())
      break;
  }
  assert(opsToErase.empty());

  // Move loop body under appropriate newly created affine loops.
  mover.moveAll(loopRefToOp);
  //  funcOp.dump();

  ConversionTarget target(getContext());
  target.addIllegalOp<KrnlTerminatorOp>();

  // krnl.dim operations must be lowered prior to this pass.
  target.addIllegalOp<KrnlDimOp>();
  target.addLegalOp<AffineYieldOp>();
  target.addLegalOp<AffineLoadOp>();
  target.addLegalOp<AffineStoreOp>();
  target.addLegalOp<LoadOp>();
  target.addLegalOp<StoreOp>();
  OwningRewritePatternList patterns;
  patterns.insert<KrnlTerminatorLowering>(&getContext());
  patterns.insert<KrnlLoadLowering>(&getContext());
  patterns.insert<KrnlStoreLowering>(&getContext());
  DenseSet<Operation *> unconverted;
  if (failed(applyPartialConversion(
          getFunction(), target, std::move(patterns), &unconverted)))
    signalPassFailure();
}
} // namespace

std::unique_ptr<Pass> mlir::createConvertKrnlToAffinePass() {
  return std::make_unique<ConvertKrnlToAffinePass>();
}
