/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertKrnlToAffine.cpp - Krnl Dialect Lowering --------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of Krnl operations to the affine dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/Support/Debug.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

#include <functional>
#include <mutex>

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;
using namespace mlir::affine;

namespace onnx_mlir {
namespace krnl {

UnrollAndJamMap unrollAndJamMap;
std::mutex unrollAndJamMutex;

// Since Krnl Dialect allows optimizations to be specified in the form of
// recipes without being applied, some IR block may exist under Krnl loops
// corresponding to loops that will be materialized only after relevant
// optimization recipes are applied; these Krnl loops serve as anchors for IR
// placement as we progressively apply optimization recipes, creating new
// concrete loops that will correspond to these optimized loop references.
// Whenever a concrete loop gets materialized and is referred to by Krnl loop
// reference %loop_ref, we will need to maintain the relative positioning of IR
// block and their parent loop operations; we do so by moving IR blocks while
// Krnl Dialect lowering proceeds.
//
// Consider the following example, where we specify the recipe for a
// 2-dimensional tiled loop, and insert memory allocation/deallocation aimed to
// set up and clean up per-tile temporary buffer:
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
// Therefore as we proceed to lower Krnl Dialect, there will be no place for
// these (blocks of) operations to exist until the corresponding concrete outer
// loops emerge, as a result of optimizations being applied. Upon materializing
// such a loop, we will move these (blocks of) operations to the corresponding
// regions in the newly created loops.
//
// We use LoopBody mover to:
// - register, for each Krnl loop reference, blocks of operations
//   that should be contained directly beneath the corresponding concrete loops
//   as the moving plan in the beginning of the Krnl Dialect lowering.
// - subsequently, when the concrete loops corresponding to the Krnl loop
//   reference is materialized, IR blocks will be moved to appropriate locations
//   based on information recorded as moving plan.
//
// Thus, for the above IR, the following moving plan will be registered:
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
// Since the tiling has not taken place yet, tile coordinate iteration loops
// have not been materialized, therefore the alloc and dealloc operations do not
// fit in the IR presently yet. Instead, they will be placed within a
// krnl.movable op region, to indicate that their positioning is subject to
// change.
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
   * Represents either:
   * - a list of operations to be moved, or
   * - a particular set of loop nests expected in the destination loop body.
   *     This is helpful because we're only adjusting the relative positioning
   *     of IR blocks with respect to the concrete loops as we lowering the Krnl
   *     Dialect by applying the optimization recipes. Therefore, clearly
   *     moving IR blocks alone is sufficient to achieve our goal, and recording
   *     the position of expected loop nests in the destination loop body simply
   *     helps determine the correct relative position of IR blocks with respect
   *     to inner loops.
   */
  struct Movable {
    std::optional<KrnlMovableOp> movableOp;
    std::optional<llvm::SmallVector<Value, 4>> loopsToSkip;

    // Movable that stores a KrnlMovableOp.
    explicit Movable(KrnlMovableOp op) : movableOp(op) {}

    // Alternate Movable that stores a list of loopRefs for all its
    // optimized loops (except if that optimized loop is an KrnlUnrollOp),
    explicit Movable(KrnlIterateOp op) {
      auto operandRange = op->getOperands();
      SmallVector<Value, 4> values;
      for (int64_t i = 0; i < op.getNumOptimizedLoops(); ++i) {
        // Note, KrnlIterateOp have their loopRef for optimized loops as
        // first operands [0..getNumOptimizedLoops).
        Value val = operandRange[i];
        // Only skip non-unroll loops.  Loops that are unrolled are by
        // definitions a loop whose loopRef is used by a KrnlUnrollOp.
        if (llvm::all_of(val.getUsers(), [&](Operation *user) {
              return mlir::dyn_cast_or_null<KrnlUnrollOp>(user);
            }))
          values.emplace_back(val);
      }
      loopsToSkip = values;
    }
  };

  /*!
   * Register in our moving plan that content in the movable op should be moved
   * under the concrete loops corresponding to loop.
   * @param movable IR blocks enclosed in krnl.movable op to move around.
   * @param loop The Krnl Loop referring to the concrete loop surrounding the
   * content of the movable op in the lowered IR.
   */
  void toMoveUnder(const Movable &movable, KrnlIterateOp loop) {
    // Set movable in the moving plan of the innermost optimized loop.
    Value innerMostLoopHandler =
        loop.getOperand(loop.getNumOptimizedLoops() - 1);
    movingPlan[innerMostLoopHandler].push_back(movable);
  }

  /*!
   * Signal that the concrete loop corresponding to loopRef has been
   * materialized, and therefore we can transfer operations to its loop body as
   * specified by moving plan.
   * @param loopRef Krnl loop ref corresponding to the concrete loop being
   * materialized.
   * @param loopRefToOp A dictionary keeping track of the correspondence between
   * Krnl loop references and concrete loops.
   * @param erase whether to erase entries in the moving plan corresponding to
   * this action.
   */
  void moveOne(Value loopRef,
      llvm::SmallDenseMap<Value, Operation *, 4> &loopRefToOp,
      bool erase = true) {
    // Find the forOp associated with loopRef, get ready to insert into
    // forOp body.
    // Cast to affine.forOp or affine.parallelOp
    Block &loopBody =
        dyn_cast_or_null<AffineForOp>(loopRefToOp[loopRef])
            ? llvm::cast<AffineForOp>(loopRefToOp[loopRef]).getRegion().front()
            : llvm::cast<AffineParallelOp>(loopRefToOp[loopRef])
                  .getRegion()
                  .front();
    auto insertPt = loopBody.begin();
    // If the first operation is not a loop, it must be inserted at the end of
    // the block. This situation arises when the loop of the first operation has
    // been unrolled.
    if (!isa<AffineForOp, AffineParallelOp>(loopBody.getOperations().front()))
      insertPt = loopBody.getTerminator()->getIterator();

    // Find the ops to transfer (saved into a Movable) associated with
    // loopRef.
    auto opsToTransfer = movingPlan[loopRef];
    if (erase)
      movingPlan.erase(loopRef);

    for (const Movable &transferPt : opsToTransfer) {
      assert(insertPt != loopBody.end() && "Expecting insertPt in the loop");
      assert(transferPt.loopsToSkip.has_value() !=
                 transferPt.movableOp.has_value() &&
             "Expecting non-equal values");
      if (transferPt.movableOp.has_value()) {
        // This Movable is the kind that record one MovableOp.
        KrnlMovableOp movableOp = transferPt.movableOp.value();

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
      } else if (transferPt.loopsToSkip.has_value()) {
        // This Movable is the kind that record a list of loopRefs
        // associated with a KrnlIterate.
        std::optional<AffineForOp> loopToSkip;
        loopToSkip =
            transferPt.loopsToSkip.value().empty()
                ? loopToSkip
                : llvm::cast<AffineForOp>(
                      loopRefToOp[transferPt.loopsToSkip.value().front()]);

        // Move iterator to point to the next AffineFor Op.
        while (insertPt != loopBody.end() &&
               (!mlir::dyn_cast_or_null<AffineForOp>(&*insertPt) ||
                   !mlir::dyn_cast_or_null<AffineParallelOp>(&*insertPt)) &&
               loopToSkip) {
          assert(mlir::dyn_cast_or_null<KrnlMovableOp>(&*insertPt) &&
                 "Expecting a KrnlMovableOp");
          insertPt++;
        }

        // Assert that now insertion point points to the loop to skip.
        if (loopToSkip)
          assert(insertPt == loopToSkip.value()->getIterator());

        // Skip loop by incrementing insertion point.
        insertPt++;
      }
    }
  }

  void moveAll(llvm::SmallDenseMap<Value, Operation *, 4> &loopRefToOp) {
    for (const auto &pair : movingPlan)
      moveOne(pair.first, loopRefToOp, /*erase=*/false);
  }

private:
  llvm::DenseMap<Value, llvm::SmallVector<Movable, 4>> movingPlan;
};

/*!
 * Helper function to separate the operations nested directly within a
 * Krnl.iterate op into two kinds:
 * - the first kind is contiguous sequence of operations that will need to be
 *     moved to a concrete loop when it materializes.
 * - the second kind is anchors, which are Krnl loop operations. They need not
 *     be moved because they are the references, and IR blocks will be
 *     positioned relative to these anchors.
 *
 * And record the moving plans in mover.
 *
 * @param root root Krnl iterate operation.
 * @param builder operation builder.
 * @param mover loop body mover.
 */
static void markLoopBodyAsMovable(
    KrnlIterateOp root, OpBuilder builder, LoopBodyMover &mover) {
  Region &bodyRegion = root.getBodyRegion();
  if (root.getNumOptimizedLoops() == 0)
    return;

  for (auto &block : bodyRegion.getBlocks()) {
    assert(!block.empty() && "IterateOp body block shouldn't be empty.");

    // Delimeter ops are delimeter of a movable chunk of code.
    llvm::SmallVector<Operation *> delimeterOps(block.getOps<KrnlIterateOp>());
    delimeterOps.push_back(block.getTerminator());
    Operation *movableBeginOp = &block.front();
    for (Operation *delimeterOp : delimeterOps) {
      Block::iterator movableBegin = movableBeginOp->getIterator();

      // If no op to extract, continue;
      if (movableBegin == delimeterOp->getIterator())
        continue;

      MultiDialectBuilder<KrnlBuilder> create(builder, delimeterOp->getLoc());
      KrnlMovableOp movableOp = create.krnl.movable();
      Region &movableRegion = movableOp.getRegion();
      Block *entryBlock = new Block();
      movableRegion.push_back(entryBlock);
      entryBlock->getOperations().splice(entryBlock->end(),
          block.getOperations(), movableBegin, delimeterOp->getIterator());
      KrnlMovableOp::ensureTerminator(
          movableRegion, builder, delimeterOp->getLoc());

      mover.toMoveUnder(LoopBodyMover::Movable(movableOp), root);
      if (auto iterateOp = mlir::dyn_cast_or_null<KrnlIterateOp>(delimeterOp))
        mover.toMoveUnder(LoopBodyMover::Movable(iterateOp), root);

      movableBeginOp = delimeterOp->getNextNode();
    }
  }
}

static void lowerGetInductionVariableValueOp(
    KrnlGetInductionVariableValueOp &getIVOp,
    llvm::SmallDenseMap<Value, Operation *, 4> &loopRefToOp) {
  auto zippedOperandsResults =
      llvm::zip(getIVOp->getOperands(), getIVOp->getResults());
  for (const auto &operandAndResult : zippedOperandsResults) {
    auto operand = std::get<0>(operandAndResult);
    auto result = std::get<1>(operandAndResult);
    if (auto forOp =
            mlir::dyn_cast_or_null<AffineForOp>(loopRefToOp[operand])) {
      result.replaceAllUsesWith(forOp.getInductionVar());
    } else {
      auto parallelOp =
          mlir::dyn_cast_or_null<AffineParallelOp>(loopRefToOp[operand]);
      assert(parallelOp && "expected affine.parallelOp only");
      result.replaceAllUsesWith(parallelOp.getIVs()[0]);
    }
  }
}

static void lowerIterateOp(KrnlIterateOp &iterateOp, OpBuilder &builder,
    llvm::SmallDenseMap<Value, Operation *, 4> &refToOps) {
  builder.setInsertionPointAfter(iterateOp);
  // Map from unoptimizedLoopRef to the (original, unoptimized) AffineForOp.
  SmallVector<std::pair<Value, Operation *>, 4> currentNestedForOps;
  ArrayRef<Attribute> boundMapAttrs =
      iterateOp->getAttrOfType<ArrayAttr>(KrnlIterateOp::getBoundsAttrName())
          .getValue();
  auto operandItr =
      iterateOp.operand_begin() + iterateOp.getNumOptimizedLoops();

  ValueRange inits = iterateOp.getIterArgInits();

  // For each bounds, create an original loop with its original bounds using
  // an affine.for. This affine.for will be transformed if any optimizations are
  // present on the loop nest (aka permute, tile, ...).
  for (size_t boundIdx = 0; boundIdx < boundMapAttrs.size(); boundIdx += 2) {
    // Consume input loop operand, at this stage, do not do anything with it.
    auto unoptimizedLoopRef = *(operandItr++);

    // Organize operands into lower/upper bounds in affine.for ready formats.
    llvm::SmallVector<Value, 4> lbOperands, ubOperands;
    AffineMap lbMap, ubMap;
    for (int boundType = 0; boundType < 2; boundType++) {
      auto &operands = boundType == 0 ? lbOperands : ubOperands;
      auto &map = boundType == 0 ? lbMap : ubMap;
      map = mlir::cast<AffineMapAttr>(boundMapAttrs[boundIdx + boundType])
                .getValue();
      operands.insert(
          operands.end(), operandItr, operandItr + map.getNumInputs());
      std::advance(operandItr, map.getNumInputs());
    }

    auto forOp = builder.create<AffineForOp>(iterateOp.getLoc(), lbOperands,
        lbMap, ubOperands, ubMap, /*step*/ 1, inits,
        /*bodyBuilder=*/[](OpBuilder &, Location, Value, ValueRange) {
          // Make sure we don't create a default terminator in the loop body as
          // the proper terminator will be added later.
        });

    currentNestedForOps.emplace_back(std::make_pair(unoptimizedLoopRef, forOp));
    builder.setInsertionPoint(
        llvm::cast<AffineForOp>(currentNestedForOps.back().second).getBody(),
        llvm::cast<AffineForOp>(currentNestedForOps.back().second)
            .getBody()
            ->begin());
    // Update inits to iterArgs of forOp.
    inits = ValueRange(forOp.getRegionIterArgs());
  }

  // add yield for each affine.for created with result of inner affine.for
  // until last optimized loop.
  for (int64_t i = 0; i < (int64_t)currentNestedForOps.size() - 1; i++) {
    auto forOp = llvm::cast<AffineForOp>(currentNestedForOps[i].second);
    if ((iterateOp.getNumOptimizedLoops() - 1) == i) {
      // For last optimized loop.
      // yield the iterateOp yield value.
      builder.setInsertionPointToEnd(forOp.getBody());
      auto Yield =
          mlir::cast<KrnlYieldOp>(iterateOp.getBody()->getTerminator());
      builder.create<AffineYieldOp>(iterateOp.getLoc(), Yield.getOperands());

      // replace use of iterateOp iterArgs with forOp iterArgs.
      for (auto [newIterArg, oldItArg] :
          llvm::zip(forOp.getRegionIterArgs(), iterateOp.getRegionIterArgs())) {
        oldItArg.replaceAllUsesWith(newIterArg);
      }
      // No need to add yield for rest nested loops.
      // These nested loops will be replaced when lower nested iterateOp.
      break;
    }
    auto innerForOp =
        llvm::cast<AffineForOp>(currentNestedForOps[i + 1].second);
    builder.setInsertionPointToEnd(forOp.getBody());
    if (forOp.getNumResults() > 0)
      builder.create<AffineYieldOp>(
          iterateOp.getLoc(), innerForOp.getResults());
    else
      builder.create<AffineYieldOp>(iterateOp.getLoc());
  }

  // Replace induction variable references from those introduced by a
  // single krnl.iterate to those introduced by multiple affine.for
  // operations.
  for (int64_t i = 0; i < (int64_t)currentNestedForOps.size() - 1; i++) {
    auto iterateIV = iterateOp.getBodyRegion().front().getArgument(0);
    BlockArgument forIV = llvm::cast<AffineForOp>(currentNestedForOps[i].second)
                              .getBody()
                              ->getArgument(0);
    iterateIV.replaceAllUsesWith(forIV);
    iterateOp.getBodyRegion().front().eraseArgument(0);
  }

  // Pop krnl.iterate body region block arguments which is not iterArgs, leave
  // the last one for convenience (it'll be taken care of by region inlining).
  unsigned int numIterArgs = iterateOp.getNumIterArgs();
  while (
      iterateOp.getBodyRegion().front().getNumArguments() > (numIterArgs + 1))
    iterateOp.getBodyRegion().front().eraseArgument(0);

  if (currentNestedForOps.empty()) {
    // Collect information about nested loop.
    bool isLoop = iterateOp.getNumOptimizedLoops() > 0;
    bool outerLoopHasResult = false;
    bool iterateHasResult = iterateOp.getNumResults() > 0;
    if (isLoop) {
      Value loopRef =
          iterateOp.getOperand(iterateOp.getNumOptimizedLoops() - 1);
      auto it = refToOps.find(loopRef);
      assert(it != refToOps.end());
      auto outerLoop = llvm::cast<AffineForOp>(it->second);
      outerLoopHasResult = outerLoop.getNumResults() > 0;
    }

    // When there's loop and iterateOp/outerLoop has result.
    if (isLoop && (iterateHasResult || outerLoopHasResult)) {
      // Recreate forOps for iterate with iterateOp inits.
      // The old forOps are using outer iterateOp inits.
      std::vector<AffineForOp> newForOps;
      std::vector<AffineForOp> oldForOps;
      for (int i = 0; i < iterateOp.getNumOptimizedLoops(); ++i) {
        Value LoopRef = iterateOp.getOperand(i);
        auto it = refToOps.find(LoopRef);
        assert(it != refToOps.end());

        auto oldForOp = llvm::cast<AffineForOp>(it->second);
        builder.setInsertionPointAfter(oldForOp);
        oldForOps.emplace_back(oldForOp);
        auto forOp = builder.create<AffineForOp>(iterateOp.getLoc(),
            oldForOp.getLowerBoundOperands(), oldForOp.getLowerBoundMap(),
            oldForOp.getUpperBoundOperands(), oldForOp.getUpperBoundMap(),
            /*step*/ 1, inits,
            /*bodyBuilder=*/[](OpBuilder &, Location, Value, ValueRange) {
              // Make sure we don't create a default terminator in the loop body
              // as the proper terminator will be added later.
            });
        newForOps.emplace_back(forOp);
        refToOps[LoopRef] = forOp;
        // Update inits to iterArgs of forOp.
        inits = ValueRange(forOp.getRegionIterArgs());
      }

      // Move the body of oldForOp to newForOp.
      auto innermostNewForOp = newForOps.back();
      auto oldForOp = oldForOps.back();
      Region &innerMostRegion = innermostNewForOp.getRegion();

      innerMostRegion.getBlocks().clear();
      innerMostRegion.getBlocks().splice(
          innerMostRegion.end(), oldForOp.getBodyRegion().getBlocks());

      // After the splice, newForOp get entry arguments of oldForOp.
      // Remove oldForOp iter arguments.
      Block *loopEntry = innermostNewForOp.getBody();
      int oldForOpResNum = oldForOp.getResults().size();
      for (int i = 0; i < oldForOpResNum; ++i) {
        int lastArgIdx = loopEntry->getNumArguments() - 1;
        loopEntry->eraseArgument(lastArgIdx);
      }
      // Add newForOp iter arguments. Then replace iterateOp iterArgs with
      // newForOp iter arguments.
      auto iterLoopArgs = iterateOp.getRegionIterArgs();
      for (auto iterArg : iterLoopArgs) {
        auto NewArg =
            loopEntry->addArgument(iterArg.getType(), iterArg.getLoc());
        iterArg.replaceAllUsesWith(NewArg);
      }

      // Remove old ForOps.
      for (auto it = oldForOps.rbegin(); it != oldForOps.rend(); ++it) {
        auto forOp = *it;
        forOp.erase();
      }

      // add yield for each affine.for created with result of inner affine.for
      // except innermost affine.for.
      for (int64_t i = 0; i < (int64_t)newForOps.size() - 1; i++) {
        auto forOp = newForOps[i];
        auto innerForOp = newForOps[i + 1];
        builder.setInsertionPointToEnd(forOp.getBody());
        if (forOp.getNumResults() > 0)
          builder.create<AffineYieldOp>(
              iterateOp.getLoc(), innerForOp.getResults());
        else
          builder.create<AffineYieldOp>(iterateOp.getLoc());
      }
      // Add yield for innermost affine.for with iterateOp yield value.
      auto innerForOp = newForOps.back();
      auto prevTerm = innerForOp.getBody()->getTerminator();
      builder.setInsertionPointToEnd(innerForOp.getBody());
      auto iterTerm =
          mlir::cast<KrnlYieldOp>(iterateOp.getBody()->getTerminator());
      builder.create<AffineYieldOp>(iterateOp.getLoc(), iterTerm.getOperands());
      // Remove the old terminator.
      prevTerm->erase();

      // replace use of iterateOp result with outer affine.for result.
      auto outermostForOp = llvm::cast<AffineForOp>(newForOps.front());
      for (auto [result, newResult] :
          llvm::zip(iterateOp.getResults(), outermostForOp.getResults())) {
        result.replaceAllUsesWith(newResult);
      }
    }
    // When there's no loop but iterateOp has result.
    else if (!isLoop && iterateHasResult) {
      // Replace use of iteratedOp with the yield value.
      auto Yield =
          mlir::cast<KrnlYieldOp>(iterateOp.getBody()->getTerminator());
      for (auto [result, yieldValue] :
          llvm::zip(iterateOp.getResults(), Yield.getOperands())) {
        result.replaceAllUsesWith(yieldValue);
      }
      // Replace iterArg with iterInit.
      auto iterLoopArgs = iterateOp.getRegionIterArgs();
      auto iterInits = iterateOp.getIterArgInits();
      // Add iterLoopArgs to outer affine.for region iterArgs.
      for (auto [arg, init] : llvm::zip(iterLoopArgs, iterInits)) {
        arg.replaceAllUsesWith(init);
      }
    }

    // Move operations from within iterateOp body region to the parent region of
    // iterateOp.
    Block *parentBlock = iterateOp->getBlock();
    Block &iterateOpEntryBlock = iterateOp.getBodyRegion().front();
    // Transfer body region operations to parent region, without the
    // terminator op.
    parentBlock->getOperations().splice(iterateOp->getIterator(),
        iterateOpEntryBlock.getOperations(),
        iterateOpEntryBlock.front().getIterator(),
        iterateOpEntryBlock.getTerminator()->getIterator());

  } else {
    // Transfer krnl.iterate region to innermost for op.
    auto innermostForOp =
        llvm::cast<AffineForOp>(currentNestedForOps.back().second);
    innermostForOp.getRegion().getBlocks().clear();
    Region &innerMostRegion = innermostForOp.getRegion();
    innerMostRegion.getBlocks().splice(
        innerMostRegion.end(), iterateOp.getBodyRegion().getBlocks());

    // replace iterateOp result with outer affine.for result.
    auto outermostForOp =
        llvm::cast<AffineForOp>(currentNestedForOps.front().second);
    for (auto [result, newResult] :
        llvm::zip(iterateOp.getResults(), outermostForOp.getResults())) {
      result.replaceAllUsesWith(newResult);
    }
  }

  for (const auto &pair : currentNestedForOps)
    refToOps.try_emplace(pair.first, pair.second);
}

static void removeOps(llvm::SmallPtrSetImpl<Operation *> &opsToErase) {
  // Remove lowered operations topologically; if ops are not removed
  // topologically, memory error will occur.
  size_t numOpsToRemove = opsToErase.size();
  // Given N operations to remove topologically, and that we remove
  // at least one operation during each pass through opsToErase, we
  // can only have a maximum of N passes through opsToErase.
  for (size_t i = 0; i < numOpsToRemove; i++) {
    for (Operation *op : opsToErase) {
      bool safeToDelete = op->use_empty();
      safeToDelete &= llvm::all_of(op->getRegions(), [](Region &region) {
        return llvm::all_of(region.getBlocks(), [](Block &block) {
          return (block.getOperations().size() == 0) ||
                 (block.getOperations().size() == 1 &&
                     block.getOperations()
                         .front()
                         .hasTrait<OpTrait::IsTerminator>());
        });
      });

      if (safeToDelete) {
        op->erase();
        opsToErase.erase(op);
        // Restart, itr has been invalidated.
        break;
      }
    }
    if (opsToErase.empty())
      break;
  }
}

static LogicalResult interpretOperation(Operation *op, OpBuilder &builder,
    llvm::SmallDenseMap<Value, Operation *, 4> &loopRefToOp,
    llvm::SmallPtrSetImpl<Operation *> &opsToErase, LoopBodyMover &mover) {

  // Recursively interpret nested operations.
  for (auto &region : op->getRegions())
    for (auto &block : region.getBlocks()) {
      auto &blockOps = block.getOperations();
      for (auto itr = blockOps.begin(); itr != blockOps.end();) {
        LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << " Call interpretOperation \n");
        if (failed(interpretOperation(
                &(*itr), builder, loopRefToOp, opsToErase, mover)))
          return failure();
        else
          ++itr;
      }
    }

  if (auto iterateOp = mlir::dyn_cast_or_null<KrnlIterateOp>(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret iterate op " << iterateOp << "\n");
    // If an iterateOp has no unoptimized loop references, then we need to lower
    // them manually.
    if (opsToErase.count(op) == 0) {
      lowerIterateOp(iterateOp, builder, loopRefToOp);
      opsToErase.insert(iterateOp);
    }
    return success();
  } else if (auto blockOp = mlir::dyn_cast_or_null<KrnlBlockOp>(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret block op " << blockOp << "\n");
    SmallVector<AffineForOp, 2> tiledLoops;
    SmallVector<AffineForOp, 1> loopsToTile = {
        llvm::cast<AffineForOp>(loopRefToOp[blockOp.getLoop()])};

    int64_t step = blockOp.getTileSizeAttr().getInt();
    if (failed(tilePerfectlyNested(loopsToTile, step, &tiledLoops))) {
      return failure();
    }

    if (blockOp.getResult(1).use_empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << DEBUG_TYPE << " inner block loop unused, trivialize\n";
        tiledLoops[1].dump();
      });
      tiledLoops[1].setConstantLowerBound(0);
      tiledLoops[1].setConstantUpperBound(1);
      tiledLoops[1].setStep(1);
      LLVM_DEBUG(tiledLoops[1].dump());
    }
    assert(tiledLoops.size() == 2);
    assert(blockOp.getNumResults() == 2);

    // Record the tiled loop references, and their corresponding tiled
    // for loops in loopRefToLoop.
    loopRefToOp.erase(loopRefToOp.find_as(blockOp.getLoop()));
    loopRefToOp[blockOp.getResult(0)] = tiledLoops[0];
    loopRefToOp[blockOp.getResult(1)] = tiledLoops[1];

    opsToErase.insert(op);
    return success();
  } else if (auto permuteOp = mlir::dyn_cast_or_null<KrnlPermuteOp>(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret permute op " << permuteOp << "\n");
    // TODO(tjingrant): call it whenever an operation lowering completes.
    removeOps(opsToErase);
    // Collect loops to permute.
    SmallVector<AffineForOp, 4> loopsToPermute;
    std::transform(permuteOp.operand_begin(), permuteOp.operand_end(),
        std::back_inserter(loopsToPermute), [&](const Value &val) {
          return llvm::cast<AffineForOp>(loopRefToOp[val]);
        });

    // Construct permutation map from integer array attribute.
    SmallVector<unsigned int, 4> permuteMap;
    for (const auto &attr : permuteOp.getMap().getAsRange<IntegerAttr>())
      permuteMap.emplace_back(attr.getValue().getSExtValue());

    // Perform loop permutation.
    permuteLoops(loopsToPermute, permuteMap);

    opsToErase.insert(op);
    return success();
  } else if (auto parallelOp = mlir::dyn_cast_or_null<KrnlParallelOp>(op)) {
    // Parallelism the given loop by transform the tagged affine.for op to
    // affine.parallel
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << " interpret parallel op "
                            << parallelOp << "\n");
    // ToFix handle multiple parallel loop
    ValueRange loopRefs = parallelOp.getLoops();
    Value numThreads = parallelOp.getNumThreads();
    StringAttr procBind = parallelOp.getProcBindAttr();
    bool needParallelClause =
        numThreads || (procBind && procBind.getValue().size() > 0);

    // Obtain the the reference the loop that needs to be parallelized
    for (Value loopRef : loopRefs) {
      // Value loopRef = parallelOp.getLoops()[0];
      //  Obtain the lowered affine.forOp
      AffineForOp loopToParallel =
          llvm::cast<AffineForOp>(loopRefToOp[loopRef]);
      OpBuilder opBuilder(loopToParallel);

      // Extract the metadata from the original affine.forOp and then create a
      // affine.parallelOp
      Location loc = loopToParallel.getLoc();
      AffineMap lbsMap = loopToParallel.getLowerBoundMap();
      ValueRange lbsOperands = loopToParallel.getLowerBoundOperands();
      AffineMap ubsMap = loopToParallel.getUpperBoundMap();
      ValueRange ubsOperands = loopToParallel.getUpperBoundOperands();

      // Current: parallel reduction is not used. Parallel reduction can be
      // enabled after the Ops have been lowered to Affine. Please check
      // Dialect/Affine/Transforms/AffineParallelize.cpp in MLIR repo to see how
      // to enable parallel reduction.
      SmallVector<LoopReduction> parallelReductions;
      auto reducedValues =
          llvm::to_vector<4>(llvm::map_range(parallelReductions,
              [](const LoopReduction &red) { return red.value; }));
      auto reductionKinds =
          llvm::to_vector<4>(llvm::map_range(parallelReductions,
              [](const LoopReduction &red) { return red.kind; }));

      AffineParallelOp parallelLoop = opBuilder.create<AffineParallelOp>(loc,
          ValueRange(reducedValues).getTypes(), reductionKinds,
          ArrayRef(lbsMap), lbsOperands, ArrayRef(ubsMap), ubsOperands,
          ArrayRef(loopToParallel.getStepAsInt()));
      parallelLoop.getRegion().takeBody(loopToParallel.getRegion());
      Operation *yieldOp = &parallelLoop.getBody()->back();
      yieldOp->setOperands(reducedValues);
      if (needParallelClause) {
        // Use clause only for the first one (expected the outermost one).
        // Ideally, we would generate here a single, multi-dimensional
        // AffineParallelOp, and we would not need to reset the flag.
        needParallelClause = false;
        // Currently approach: insert after yield and then move before it.
        PatternRewriter::InsertionGuard insertGuard(builder);
        builder.setInsertionPointAfter(yieldOp);
        // Get induction variable.
        ValueRange optionalLoopIndices = parallelLoop.getIVs();
        assert(optionalLoopIndices.size() >= 1 &&
               "expected at least one loop index");
        Value parallelLoopIndex = optionalLoopIndices[0];
        Operation *newOp = opBuilder.create<KrnlParallelClauseOp>(
            loc, parallelLoopIndex, numThreads, procBind);
        newOp->moveBefore(yieldOp);
      }
      // Replace the affine.forOp with affine.parallelOp in loopRefToTop
      loopRefToOp[loopRef] = parallelLoop;
      loopToParallel.erase();
    }
    opsToErase.insert(parallelOp);
    return success();
  }
  return success();
}

AffineTypeConverter::AffineTypeConverter() {
  // The order of type conversion is important: later ones are tried earlier.
  addConversion([](Type type) { return type; });

  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

//
//===----------------------------------------------------------------------===//
// ConvertKrnlToAffinePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the krnl dialect operations.
/// At this stage the dialect will contain standard operations as well like
/// add and multiply, this pass will leave these operations intact.
struct ConvertKrnlToAffinePass
    : public PassWrapper<ConvertKrnlToAffinePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertKrnlToAffinePass);

  ConvertKrnlToAffinePass() = default;
  ConvertKrnlToAffinePass(const ConvertKrnlToAffinePass &pass)
      : PassWrapper<ConvertKrnlToAffinePass, OperationPass<func::FuncOp>>() {}
  ConvertKrnlToAffinePass(bool parallelEnabled) {
    this->parallelEnabled = parallelEnabled;
  }

  StringRef getArgument() const override { return "convert-krnl-to-affine"; }

  StringRef getDescription() const override { return "Lower Krnl dialect."; }

  void runOnOperation() final;

  Option<bool> parallelEnabled{*this, "parallel-enabled",
      llvm::cl::desc("Enable parallelization"), llvm::cl::init(false)};
};

void ConvertKrnlToAffinePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.getBody().empty()) // external function: nothing to do
    return;

  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(
      &getContext(), dataLayoutAnalysis.getAtOrAbove(funcOp));
  // Request C wrapper emission via attribute.
  funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
      UnitAttr::get(&getContext()));

  // Move invariant instructions outside of the loops as many as possible. This
  // helps make loops perfectly nested, which facilitates transformations.
  funcOp.walk([&](KrnlIterateOp loopOp) {
    moveLoopInvariantCode(
        mlir::cast<LoopLikeOpInterface>(loopOp.getOperation()));
  });

  // We use the end of the function body as a staging area for movable ops.
  builder.setInsertionPoint(&funcOp.getBody().front(),
      funcOp.getBody().front().without_terminator().end());
  LoopBodyMover mover;
  funcOp.walk(
      [&](KrnlIterateOp op) { markLoopBodyAsMovable(op, builder, mover); });

  // Interpret krnl dialect operations while looping recursively through
  // operations within the current function, note that erasing operations
  // while iterating is tricky because it can invalidate the iterator, so we
  // collect the operations to be erased in a small ptr set `opsToErase`, and
  // only erase after iteration completes.
  llvm::SmallDenseMap<Value, Operation *, 4> loopRefToOp;
  llvm::SmallPtrSet<Operation *, 4> opsToErase;

  // Lower `define_loops` first.
  // This is will make sure affine.for created for all the defined loops first.
  // Later when lower things like nested iteratorOp and blockOp, these
  // affine.for will be ready to use.
  funcOp->walk([&](KrnlDefineLoopsOp defineOp) {
    // Make sure define loop lowered first, so the iterateOp which create
    // affine.for can be lowered first.
    // This is because the affine.for created by iterateOp will be used by
    // the blockOp and permuteOp and the nested iterateOp.
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret define op " << defineOp << "\n");
    // Collect users of defineLoops operations that are iterate operations.
    std::vector<KrnlIterateOp> iterateOps;
    for (auto result : defineOp.getResults())
      for (auto *user : result.getUsers())
        if (auto iterateOp = mlir::dyn_cast_or_null<KrnlIterateOp>(user))
          if (std::find(iterateOps.begin(), iterateOps.end(), iterateOp) ==
              iterateOps.end())
            iterateOps.push_back(mlir::dyn_cast<KrnlIterateOp>(user));

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
    opsToErase.insert(defineOp);
  });

  if (failed(interpretOperation(
          funcOp, builder, loopRefToOp, opsToErase, mover))) {
    signalPassFailure();
    return;
  }
  // Lower `unrollOp` after all `iterateOps` have been lowered.
  // This is necessary because `unrollOp` may reference a loop created by an
  // outer `iterateOp`, which will be updated after lowering an inner
  // `iterateOp`. If `unrollOp` is lowered before `iterateOp`, the loop may end
  // up in an incorrect state during unrolling.
  auto unrolls = funcOp.getOps<KrnlUnrollOp>();
  for (KrnlUnrollOp unrollOp : unrolls) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret unroll op " << unrollOp << "\n");
    // Unroll the affine for loop fully.
    Value loopRef = unrollOp.getLoop();
    auto loopToUnroll = llvm::cast<AffineForOp>(loopRefToOp[loopRef]);

    mover.moveOne(loopRef, loopRefToOp);

    // Interpret and remove 'krnl.get_induction_var' inside the unrolling loop
    // if any. Otherwise, we lost the trace of the loop induction variables.
    for (auto &region : loopToUnroll->getRegions())
      for (auto &block : region.getBlocks()) {
        auto &blockOps = block.getOperations();
        for (auto itr = blockOps.begin(); itr != blockOps.end(); ++itr) {
          Operation *genericOp = &(*itr);
          if (auto getIVOp =
                  mlir::dyn_cast_or_null<KrnlGetInductionVariableValueOp>(
                      genericOp)) {
            lowerGetInductionVariableValueOp(getIVOp, loopRefToOp);
            opsToErase.insert(genericOp);
          }
        }
      }
    removeOps(opsToErase);

    // Assert that there's no floating code within the loop to be unrolled.
    loopToUnroll.walk([](KrnlMovableOp op) {
      llvm_unreachable("Loop to unroll must not contain movable op.");
    });
    LogicalResult res = loopUnrollFull(loopToUnroll);
    assert(succeeded(res) && "failed to unroll");
    opsToErase.insert(unrollOp);
  }

  funcOp->walk([&](Operation *op) {
    if (SpecializedKernelOpInterface kernelOp =
            mlir::dyn_cast<SpecializedKernelOpInterface>(op)) {
      OperandRange loopRefs = kernelOp.getLoopRefs();
      for (auto loopRef : loopRefs)
        opsToErase.insert(loopRefToOp[loopRef]);
      kernelOp.getLoopRefs().clear();
    }
    if (auto getIVOp =
            mlir::dyn_cast_or_null<KrnlGetInductionVariableValueOp>(op)) {
      lowerGetInductionVariableValueOp(getIVOp, loopRefToOp);
      opsToErase.insert(op);
    }
  });
  removeOps(opsToErase);
  assert(opsToErase.empty());

  // Move loop body under appropriate newly created affine loops.
  mover.moveAll(loopRefToOp);

  ConversionTarget target(*ctx);
  // Legal/illegal ops.
  target.addIllegalOp<KrnlTerminatorOp>();
  target.addIllegalOp<KrnlMatMulOp>();
  target.addIllegalOp<KrnlCopyToBufferOp>();
  target.addIllegalOp<KrnlCopyFromBufferOp>();
  target.addIllegalOp<KrnlPrefetchOp>();
  target.addLegalOp<KrnlParallelClauseOp>();
  target.addLegalOp<AffineYieldOp>();
  target.addLegalOp<AffineLoadOp>();
  target.addLegalOp<AffineStoreOp>();
  target.addLegalOp<KrnlVectorTypeCastOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::memref::MemRefDialect, mlir::func::FuncDialect,
      mlir::vector::VectorDialect>();

  // Patterns.
  RewritePatternSet patterns(ctx);
  AffineTypeConverter typeConverter;

  populateKrnlToAffineConversion(typeConverter, patterns, ctx, parallelEnabled);

  // Create list for recording the <loop, unroll factor> pairs associated with
  // this function.
  UnrollAndJamList *currUnrollAndJamList = new UnrollAndJamList();
  Operation *currFuncOp = funcOp.getOperation();
  {
    const std::lock_guard<std::mutex> lock(unrollAndJamMutex);
    unrollAndJamMap[currFuncOp] = currUnrollAndJamList;
  }
  if (failed(applyPartialConversion(
          getOperation(), target, std::move(patterns)))) {
    {
      const std::lock_guard<std::mutex> lock(unrollAndJamMutex);
      unrollAndJamMap.erase(currFuncOp);
      delete currUnrollAndJamList;
    }
    signalPassFailure();
    return;
  }

  for (auto record : *currUnrollAndJamList) {
    LogicalResult res = loopUnrollJamUpToFactor(record.first, record.second);
    assert(succeeded(res) && "failed to optimize");
  }

  {
    const std::lock_guard<std::mutex> lock(unrollAndJamMutex);
    unrollAndJamMap.erase(currFuncOp);
  }

  delete currUnrollAndJamList;
}

std::unique_ptr<Pass> createConvertKrnlToAffinePass() {
  return std::make_unique<ConvertKrnlToAffinePass>();
}

std::unique_ptr<Pass> createConvertKrnlToAffinePass(bool parallelEnabled) {
  return std::make_unique<ConvertKrnlToAffinePass>(parallelEnabled);
}

void populateKrnlToAffineConversion(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx, bool parallelEnabled) {
  krnl::populateLoweringKrnlCopyFromBufferOpPattern(
      typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlCopyToBufferOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlLoadOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlStoreOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlGetLinearOffsetIndexOpPattern(
      typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlMatmultOpPattern(
      typeConverter, patterns, ctx, parallelEnabled);
  krnl::populateLoweringKrnlMemsetOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlPrefetchOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlTerminatorOpPattern(typeConverter, patterns, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
