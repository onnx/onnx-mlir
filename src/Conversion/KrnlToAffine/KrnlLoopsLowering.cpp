/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- LowerKrnl.cpp - Krnl Dialect Lowering -----------------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// Code for lowering Krnl loops.
//
//===----------------------------------------------------------------------===//
#include "KrnlLoopsLowering.h"

using namespace mlir;

namespace onnx_mlir {
namespace {

/*!
 * Helper function to separate the operations nested directly within a
 * krnl.iterate op `root` into two kinds:
 * - the first kind are anchors, which are Krnl loop operations. They will be
 *   replaced by the lowered loop operations (affine for loops); furthermore,
 *   they act as positional references for moving other operations around.
 *
 * - the second kind of operations are all the other operations. Contiguous
 *   sequence of them will be stored together for efficiency and
 *   clarity. Upon the materialization of the lowered loop corresponding to the
 *   `root` operation , they will be moved under the lowered loop operation.
 *   TODO(tjingrant): additional constraints apply, explain.
 *
 * And record the moving plans in mover.
 *
 * @param root root Krnl iterate operation.
 * @param builder operation builder.
 * @param mover loop body mover.
 */
void markLoopBodyAsMovable(mlir::KrnlIterateOp root, mlir::OpBuilder builder,
    LoopBodyMover &mover,
    DenseMap<Operation *, llvm::SmallVector<KrnlIterateOp, 4>> &structureMap) {
  auto &bodyRegion = root.bodyRegion();

  if (root.getNumOptimizedLoops() == 0)
    return;

  for (auto &block : bodyRegion.getBlocks()) {
    assert(!block.empty() && "IterateOp body block shouldn't be empty.");

    // Delimeter ops are delimeters of movable chunks of code. Movable chunks of
    // code are separated by krnl.iterate operations and terminator operations.
    llvm::SmallVector<Operation *> delimeterOps(block.getOps<KrnlIterateOp>());
    delimeterOps.push_back(block.getTerminator());

    mlir::Operation *movableChunkBegin = &block.front();
    for (auto delimeterOp : delimeterOps) {
      // If no op to extract, continue;
      if (movableChunkBegin->getIterator() == delimeterOp->getIterator())
        continue;
      auto loopStack = structureMap[root];

      std::string movingPlanStr;
      llvm::for_each(llvm::reverse(loopStack), [&](mlir::KrnlIterateOp op) {
        if (op->getLoc().isa<NameLoc>())
          movingPlanStr.append(
              op->getLoc().cast<NameLoc>().getName().str() + ",");
      });
      auto movableOp =
          builder.create<KrnlMovableOp>(delimeterOp->getLoc(), movingPlanStr);
      auto &movableRegion = movableOp.region();
      auto *entryBlock = new Block;
      movableRegion.push_back(entryBlock);
      entryBlock->getOperations().splice(entryBlock->end(),
          block.getOperations(), movableChunkBegin->getIterator(),
          delimeterOp->getIterator());
      mlir::KrnlMovableOp::ensureTerminator(
          movableRegion, builder, delimeterOp->getLoc());

      // Let mover know to move the content of movable operations under the
      // lowered loop corresponding to `root`. For the delimeter op, let mover
      // know to expect an iterate operation.
      mover.toMoveUnder(LoopBodyMover::Movable(movableOp), root, loopStack);
      if (auto iterateOp = dyn_cast_or_null<mlir::KrnlIterateOp>(delimeterOp))
        mover.toMoveUnder(LoopBodyMover::Movable(iterateOp), root);

      movableChunkBegin = delimeterOp->getNextNode();
    }
  }
}

// Pre-compute loop structure map, which maps from each krnl.iterate op to its
// surrounding krnl.iterate ops ordered from inner to outer.
void computeLoopStructureMap(KrnlIterateOp iterateOp,
    DenseMap<Operation *, llvm::SmallVector<KrnlIterateOp, 4>> &structureMap) {
  // Stuff identified chunks of operations into a krnl.movable operation.
  llvm::SmallVector<mlir::KrnlIterateOp, 4> loopStack = {iterateOp};
  while (auto parentOp = loopStack.back()->getParentOfType<KrnlIterateOp>())
    loopStack.push_back(parentOp);
  structureMap[iterateOp.getOperation()] = loopStack;
}

} // namespace

LoopBodyMover preprocessKrnlLoops(
    mlir::FuncOp funcOp, mlir::OpBuilder &builder, bool debug) {
  // Use the end of the function body as a staging area for movable ops.
  builder.setInsertionPoint(
      &funcOp.body().front(), funcOp.body().front().without_terminator().end());
  DenseMap<Operation *, llvm::SmallVector<KrnlIterateOp, 4>> structureMap;
  funcOp.walk(
      [&](KrnlIterateOp op) { computeLoopStructureMap(op, structureMap); });

  LoopBodyMover mover;
  funcOp.walk([&](KrnlIterateOp op) {
    markLoopBodyAsMovable(op, builder, mover, structureMap);
  });
  return mover;
}

void LoopBodyMover::toMoveUnder(const LoopBodyMover::Movable &movable,
    mlir::KrnlIterateOp iterateOp,
    llvm::SmallVector<mlir::KrnlIterateOp, 4> loopStack) {
  assert(iterateOp.getNumOptimizedLoops() > 0);

  mlir::Value innerMostLoopHandler =
      iterateOp.getOperand(iterateOp.getNumOptimizedLoops() - 1);
  movingPlan[innerMostLoopHandler].push_back(movable);

  llvm::SmallVector<mlir::Value, 4> enclosingLoopRefs;
  for (auto iterateOp : llvm::reverse(loopStack))
    for (int outIdx = 0; outIdx < iterateOp.getNumOptimizedLoops(); outIdx++)
      enclosingLoopRefs.emplace_back(iterateOp->getResult(outIdx));

  if (movable.movableOp.hasValue())
    structurePlan[innerMostLoopHandler] = enclosingLoopRefs;
}

void LoopBodyMover::moveOne(mlir::Value loopRef,
    llvm::SmallDenseMap<mlir::Value, mlir::AffineForOp, 4> &loopRefToOp,
    bool erase) {
  assert(loopRefToOp.count(loopRef) >= 0 &&
         "Can't find affine for operation associated with .");
  mlir::AffineForOp forOp = loopRefToOp[loopRef];
  mlir::Block &loopBody = forOp.getLoopBody().front();
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
      llvm::Optional<mlir::AffineForOp> loopToSkip;
      loopToSkip = transferPt.loopsToSkip.getValue().empty()
                       ? loopToSkip
                       : loopRefToOp[transferPt.loopsToSkip.getValue().front()];

      // Move iterator to point to the next AffineFor/KrnlIterate Op.
      while (insertPt != loopBody.end() &&
             !llvm::dyn_cast_or_null<mlir::AffineForOp>(&*insertPt) &&
             !llvm::dyn_cast_or_null<mlir::KrnlIterateOp>(&*insertPt)) {
        assert(llvm::dyn_cast_or_null<mlir::KrnlMovableOp>(&*insertPt));
        insertPt++;
      }

      // TODO(tjingrant): Fix this, sanity check needs to be done with
      // krnl.iterate pointers as well. Assert that now insertion point points
      // to the loop to skip.
      //      if (loopToSkip)
      //        assert(insertPt == loopToSkip.getValue()->getIterator());

      // Skip loop by incrementing insertion point.
      insertPt++;
    }
  }
}

void LoopBodyMover::moveAll(
    llvm::SmallDenseMap<mlir::Value, mlir::AffineForOp, 4> &loopRefToOp) {
  for (const auto &pair : movingPlan)
    moveOne(pair.first, loopRefToOp, /*erase=*/false);
  movingPlan.clear();
}

void LoopBodyMover::moveNext(
    llvm::SmallDenseMap<mlir::Value, mlir::AffineForOp, 4> &loopRefToOp) {
  llvm::SmallVector<mlir::Value, 4> readyLoopRefs;
  for (const auto &pair : movingPlan) {
    if (loopRefToOp.count(pair.first)) {
      auto loop = loopRefToOp[pair.first];
      llvm::SmallVector<mlir::AffineForOp, 4> actualEnclosingLoops = {loop};
      while (
          auto parent =
              actualEnclosingLoops.back()->getParentOfType<mlir::AffineForOp>())
        actualEnclosingLoops.push_back(parent);

      auto plannedEnclosingLoopRefs = structurePlan[pair.first];
      if (plannedEnclosingLoopRefs.size() != actualEnclosingLoops.size())
        continue;

      for (auto actualAndPlanned : llvm::zip(
               actualEnclosingLoops, llvm::reverse(plannedEnclosingLoopRefs))) {
        auto actual = std::get<0>(actualAndPlanned);
        auto planned = std::get<1>(actualAndPlanned);
        if (loopRefToOp.count(planned) == 0 || actual != loopRefToOp[planned])
          continue;
      }

      readyLoopRefs.emplace_back(pair.first);
    }
  }

  for (auto readyLoopRef : readyLoopRefs) {
    moveOne(readyLoopRef, loopRefToOp, /*erase=*/true);
  }
}

bool LoopBodyMover::empty() { return movingPlan.empty(); }
} // namespace onnx_mlir