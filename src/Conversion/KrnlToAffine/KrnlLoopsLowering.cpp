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
void markLoopBodyAsMovable(
    mlir::KrnlIterateOp root, mlir::OpBuilder builder, LoopBodyMover &mover) {
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

      // Stuff identified chunks of operations into a krnl.movable operation.
      auto movableOp = builder.create<KrnlMovableOp>(delimeterOp->getLoc());
      //      movableOp->setLoc(NameLoc::get("Random Name For Debug"));
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
      mover.toMoveUnder(LoopBodyMover::Movable(movableOp), root);
      if (auto iterateOp = dyn_cast_or_null<mlir::KrnlIterateOp>(delimeterOp))
        mover.toMoveUnder(LoopBodyMover::Movable(iterateOp), root);

      movableChunkBegin = delimeterOp->getNextNode();
    }
  }
}
} // namespace

LoopBodyMover preprocessKrnlLoops(
    mlir::FuncOp funcOp, mlir::OpBuilder &builder) {
  // Use the end of the function body as a staging area for movable ops.
  builder.setInsertionPoint(
      &funcOp.body().front(), funcOp.body().front().without_terminator().end());
  LoopBodyMover mover;
  funcOp.walk(
      [&](KrnlIterateOp op) { markLoopBodyAsMovable(op, builder, mover); });
  return mover;
}
} // namespace onnx_mlir