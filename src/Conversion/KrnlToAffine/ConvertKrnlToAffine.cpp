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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

// Per-dimension trip counts (outer-to-inner) of the loops fused by a
// krnl.collapse, keyed by the induction variable of the resulting merged loop.
// That induction variable is the value krnl.get_induction_var_value substitutes
// for such a loop reference, so presence in this table is exactly how that op
// recognizes a collapsed loop and learns how many dimensions to recover -- no
// walk back to the krnl.collapse op is needed.
//
// This is a plain C++ side-table, in the spirit of the loopRefToOp/opsToErase
// tables already threaded through this file, rather than an IR op: unlike
// krnl.parallel_clause, whose consumer is a genuinely later pass,
// krnl.collapse is created and fully consumed within one run of this pass.
using CollapseTripCounts =
    llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 4>>;

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
    // The moving plan positions movable blocks relative to the loops nested in
    // this body, so the insertion point is the first such loop -- which is not
    // necessarily the first operation of the block. A krnl.collapse'd band
    // leaves the affine.apply ops that compute the merged loop's fused bound in
    // front of it (affine::coalesceLoops materializes them there, and they have
    // to stay above the loop whose bound they feed), so a body can well open
    // with something other than its loop.
    auto insertPt = llvm::find_if(loopBody,
        [](Operation &op) { return isa<AffineForOp, AffineParallelOp>(op); });
    // With no loop at all to position against, the content goes at the end of
    // the block. This situation arises when the loop of the first operation has
    // been unrolled.
    if (insertPt == loopBody.end())
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

        // Move iterator to point to the next AffineFor Op, when there still is
        // one: an unrolled loop leaves none, and then the increment below just
        // steps over whatever the unrolled body put here, as it always has.
        // Anything sitting in front of that loop is not what this Movable
        // stands for -- a krnl.movable still awaiting its own destination, or
        // the affine.apply/affine.min ops computing the fused bound of a
        // krnl.collapse'd band, which belong above the loop they bound.
        auto loopIt = std::find_if(insertPt, loopBody.end(), [](Operation &op) {
          return isa<AffineForOp, AffineParallelOp>(op);
        });
        if (loopIt != loopBody.end())
          insertPt = loopIt;

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

// Emit the running-quotient chain that recovers the per-dimension indices of a
// collapsed loop nest from its fused index, same algorithm as
// affine::coalesceLoops:
//   iv_i = floordiv(iv_fused, product of the trip counts nested in i) mod tc_i
// built from the innermost dimension outwards.
//
// Degenerates correctly at rank 1: no op is emitted and the fused index is
// returned as is, since a single dimension's index *is* the fused index.
static void emitCollapsedIndices(OpBuilder &builder, Location loc,
    Value fusedIndex, ArrayRef<Value> tripCounts,
    SmallVectorImpl<Value> &indices) {
  size_t rank = tripCounts.size();
  indices.assign(rank, nullptr);
  Value previous = fusedIndex;
  for (size_t idx = rank; idx > 0; --idx) {
    if (idx != rank)
      previous = AffineApplyOp::create(builder, loc,
          AffineMap::get(/*dimCount=*/1, /*symbolCount=*/1,
              builder.getAffineDimExpr(0).floorDiv(
                  builder.getAffineSymbolExpr(0))),
          ValueRange{previous, tripCounts[idx]});
    if (idx == 1) {
      // The outermost dimension needs no modulo: nothing remains above it.
      indices[idx - 1] = previous;
    } else {
      indices[idx - 1] = AffineApplyOp::create(builder, loc,
          AffineMap::get(/*dimCount=*/1, /*symbolCount=*/1,
              builder.getAffineDimExpr(0) % builder.getAffineSymbolExpr(0)),
          ValueRange{previous, tripCounts[idx - 1]});
    }
  }
}

// Resolve a krnl.get_induction_var_value into the induction variables of the
// loops its operands now denote.
//
// Every operand yields one result, except a krnl.collapse'd loop queried
// without the fusedIndex attribute: that yields one result per collapsed
// dimension, recovered from the fused index by the chain above. A collapsed
// loop is recognized by having an entry in collapseTripCounts, which
// resolveCollapseOps keyed on the merged loop's induction variable -- the very
// value being substituted here, so no walk back to the krnl.collapse op is
// needed.
static LogicalResult lowerGetInductionVariableValueOp(
    KrnlGetInductionVariableValueOp &getIVOp,
    llvm::SmallDenseMap<Value, Operation *, 4> &loopRefToOp,
    const CollapseTripCounts &collapseTripCounts) {
  bool fusedIndex = getIVOp.getFusedIndex();
  // Pair each operand with the induction variable it resolves to, and with the
  // trip counts of its collapsed dimensions when it has any to expand.
  SmallVector<std::pair<Value, ArrayRef<Value>>, 4> resolved;
  size_t numIndices = 0;
  for (Value loopRef : getIVOp.getLoops()) {
    Operation *loopOp = loopRefToOp[loopRef];
    Value iv;
    if (auto forOp = mlir::dyn_cast_or_null<AffineForOp>(loopOp)) {
      iv = forOp.getInductionVar();
    } else {
      auto parallelOp = mlir::dyn_cast_or_null<AffineParallelOp>(loopOp);
      assert(parallelOp && "expected affine.parallelOp only");
      iv = parallelOp.getIVs()[0];
    }
    ArrayRef<Value> tripCounts;
    if (!fusedIndex) {
      auto it = collapseTripCounts.find(iv);
      if (it != collapseTripCounts.end())
        tripCounts = it->second;
    }
    numIndices += tripCounts.empty() ? 1 : tripCounts.size();
    resolved.emplace_back(iv, tripCounts);
  }

  // The verifier counted the results against the krnl.collapse ops; this counts
  // them against the loops those collapses actually produced. A disagreement
  // means a collapse was never resolved, which would be a bug in this pass
  // rather than bad input -- report it rather than writing out of bounds.
  if (numIndices != getIVOp.getNumResults())
    return getIVOp.emitOpError("resolves to ")
           << numIndices << " induction variable values but has "
           << getIVOp.getNumResults() << " results";

  OpBuilder builder(getIVOp);
  Location loc = getIVOp.getLoc();
  unsigned resIdx = 0;
  for (auto &[iv, tripCounts] : resolved) {
    if (tripCounts.empty()) {
      getIVOp.getResult(resIdx++).replaceAllUsesWith(iv);
      continue;
    }
    SmallVector<Value, 4> indices;
    emitCollapsedIndices(builder, loc, iv, tripCounts, indices);
    for (Value index : indices)
      getIVOp.getResult(resIdx++).replaceAllUsesWith(index);
  }
  return success();
}

// Report whether this loop's lower bound is a compile-time zero, rewriting it
// into a literal constant 0 when it is.
//
// A zero lower bound reaches here spelled in more than one way. `0 to N` gives a
// genuine constant bound, but onnx-mlir also emits `%c0 to N`, which becomes a
// single-result map over an operand instead -- canonicalization would normally
// fold the two together, and it has not run at this point in the pass. Detecting
// that is not sufficient on its own: affine::coalesceLoops tests the bound
// syntactically with hasConstantLowerBound(), so the map form has to be rewritten
// into the constant form or coalescing simply fails.
//
// There is no single upstream call for this. canonicalizeLoopBounds() would do it
// but is file-static in AffineOps.cpp, its composeAffineMapAndOperands() helper is
// in no public header, and no affine utility exposes a zero-lower-bound predicate
// (normalizeAffineFor inline-checks the same thing). So compose two public APIs:
// getConstantIntValue per operand, and AffineMap::constantFold on the bound map.
static bool normalizeZeroLowerBound(AffineForOp forOp) {
  if (forOp.hasConstantLowerBound())
    return forOp.getConstantLowerBound() == 0;
  AffineMap lbMap = forOp.getLowerBoundMap();
  // A multi-result lower bound is a max, never a plain zero.
  if (lbMap.getNumResults() != 1)
    return false;
  // The fold only succeeds if every operand is itself a compile-time constant.
  // getConstantIntValue matches any op folding to an integer attribute, so this
  // is not tied to arith.constant specifically.
  Builder builder(forOp.getContext());
  SmallVector<Attribute, 4> operandConsts;
  for (Value operand : forOp.getLowerBoundOperands()) {
    std::optional<int64_t> cst = getConstantIntValue(operand);
    if (!cst)
      return false;
    operandConsts.emplace_back(builder.getIndexAttr(*cst));
  }
  SmallVector<Attribute, 1> folded;
  if (failed(lbMap.constantFold(operandConsts, folded)) || folded.empty())
    return false;
  auto intAttr = mlir::dyn_cast<IntegerAttr>(folded[0]);
  if (!intAttr || intAttr.getInt() != 0)
    return false;
  // Semantics-preserving now that the bound is known to evaluate to zero. This is
  // the form coalesceLoops requires, and it drops the now-dead bound operand.
  forOp.setConstantLowerBound(0);
  return true;
}

// Resolve every krnl.collapse listed among iterateOp's optimized loops.
//
// This is called from lowerIterateOp, at the one point where the loop nest is
// exactly what mlir::affine::coalesceLoops expects: a band of one plain
// affine.for per original loop dimension, each nested directly inside the
// previous one with nothing interspersed between the headers, and with the
// iterate body already spliced into the innermost one. No block, permute or
// unroll recipe has been applied yet.
//
// On success, each run of nestedForOps entries fused by a krnl.collapse is
// replaced by a single entry mapping the collapse result to the merged loop, so
// that the caller registers exactly the loop refs that survive and never the
// erased ones.
static LogicalResult resolveCollapseOps(KrnlIterateOp &iterateOp,
    SmallVector<std::pair<Value, Operation *>, 4> &nestedForOps,
    CollapseTripCounts &collapseTripCounts) {
  // Gather the collapse ops among the optimized loops, in operand order.
  SmallVector<KrnlCollapseOp, 2> collapseOps;
  for (int64_t i = 0; i < iterateOp.getNumOptimizedLoops(); ++i)
    if (auto collapseOp =
            iterateOp.getOperand(i).getDefiningOp<KrnlCollapseOp>())
      collapseOps.emplace_back(collapseOp);
  if (collapseOps.empty())
    return success();

  // coalesceLoops discards the iterArgs of the loops it erases (it forwards
  // them to their inits), so reductions cannot be collapsed correctly.
  if (iterateOp.getNumIterArgs() > 0)
    return iterateOp.emitOpError(
        "krnl.collapse is not supported on an iterate with iterArgs");

  for (KrnlCollapseOp collapseOp : collapseOps) {
    ValueRange loopsToFuse = collapseOp.getLoops();

    // Locate the affine.for ops built for the loop refs being fused. Indices
    // are recomputed per collapse op because an earlier collapse may already
    // have shortened nestedForOps.
    SmallVector<size_t, 4> indices;
    for (Value loopRef : loopsToFuse) {
      auto it = llvm::find_if(nestedForOps,
          [&](const std::pair<Value, Operation *> &pair) {
            return pair.first == loopRef;
          });
      if (it == nestedForOps.end())
        return collapseOp.emitOpError("collapses a loop that is not an "
                                      "original loop of its krnl.iterate");
      indices.emplace_back(std::distance(nestedForOps.begin(), it));
    }
    // coalesceLoops needs a perfectly nested band, which only holds for
    // adjacent dimensions taken outer-to-inner.
    for (size_t i = 1; i < indices.size(); ++i)
      if (indices[i] != indices[i - 1] + 1)
        return collapseOp.emitOpError(
            "collapses loops that are not adjacent dimensions of its "
            "krnl.iterate, listed outer-to-inner");

    // The consumed-ref rule: a collapsed loop ref has no affine.for of its own
    // once fused, so nothing outside the collapse may refer to it. This is what
    // rejects sharing an operand with krnl.block/krnl.unroll/krnl.permute, and
    // krnl.get_induction_var_value on a collapsed-away ref (query the collapse
    // result instead, which yields every collapsed dimension's index).
    for (Value loopRef : loopsToFuse)
      for (Operation *user : loopRef.getUsers())
        if (user != collapseOp.getOperation() && !isa<KrnlIterateOp>(user))
          return collapseOp.emitOpError("collapses a loop reference that is "
                                        "also used by '")
                 << user->getName()
                 << "'; a collapsed loop reference may not be used elsewhere";

    SmallVector<AffineForOp, 4> band;
    for (size_t idx : indices)
      band.emplace_back(llvm::cast<AffineForOp>(nestedForOps[idx].second));

    // coalesceLoops only handles normalized loops, and the trip count recorded
    // below is the upper bound, which is only the trip count when the lower bound
    // is 0 and the step is 1. Check both here, with separate diagnostics, so that
    // this reports what is actually wrong instead of a bare pass failure.
    //
    // These cannot live in KrnlCollapseOp's verifier: the bounds belong to the
    // krnl.iterate, which the collapse op cannot see from its own operands.
    for (AffineForOp forOp : band) {
      if (forOp.getStepAsInt() != 1)
        return collapseOp.emitOpError("can only collapse loops with a step of 1");
      if (!normalizeZeroLowerBound(forOp))
        return collapseOp.emitOpError(
            "can only collapse loops whose lower bound is a compile-time "
            "constant 0 (a literal 0, or a constant that folds to 0); a "
            "non-zero lower bound would need its per-dimension offset added "
            "back when recovering the indices, which is not supported");
    }

    // coalesceLoops rewrites uses of the collapsed induction variables only
    // within the innermost loop's own region. By this point markLoopBodyAsMovable
    // has typically parked the iterate body in a krnl.movable elsewhere in the
    // function, so a use out there would survive that rewrite and then dangle
    // when the loop defining it is erased -- an assertion failure inside
    // coalesceLoops, or worse without assertions. Such uses come from naming a
    // collapsed dimension in the `with (%ii -> %i = ...)` clause of the
    // krnl.iterate; a collapsed nest has to obtain its indices from
    // krnl.get_induction_var_value on the collapse result instead.
    Region &innermostRegion = band.back().getRegion();
    for (AffineForOp forOp : band)
      for (OpOperand &use : forOp.getInductionVar().getUses())
        if (!innermostRegion.isAncestor(use.getOwner()->getParentRegion()))
          return collapseOp.emitOpError(
              "the induction variable of a collapsed dimension cannot be used "
              "directly, as named by the 'with' clause of its krnl.iterate; "
              "query krnl.get_induction_var_value on the collapse result "
              "instead");

    // Materialize each dimension's trip count before coalescing, since
    // coalesceLoops rewrites the outermost bound and erases the inner loops.
    // With a zero lower bound and unit step the trip count is just the upper
    // bound. These bound operands are operands of the krnl.iterate, so they are
    // defined above the band and dominate this insertion point. coalesceLoops
    // recomputes the very same values internally; the duplicates fold away
    // under CSE/canonicalization.
    AffineForOp outermost = band.front();
    OpBuilder tcBuilder(outermost);
    Location loc = outermost.getLoc();
    SmallVector<Value, 4> tripCounts;
    for (AffineForOp forOp : band) {
      AffineMap ubMap = forOp.getUpperBoundMap();
      ValueRange ubOperands = forOp.getUpperBoundOperands();
      // A multi-result upper bound map denotes the min over its results.
      if (llvm::hasSingleElement(ubMap.getResults()))
        tripCounts.emplace_back(
            AffineApplyOp::create(tcBuilder, loc, ubMap, ubOperands));
      else
        tripCounts.emplace_back(
            AffineMinOp::create(tcBuilder, loc, ubMap, ubOperands));
    }

    if (failed(coalesceLoops(band)))
      return collapseOp.emitOpError("failed to collapse the loop nest");

    // coalesceLoops always materializes the per-dimension index recovery at the
    // top of the merged loop, so that it can rewrite uses of the original
    // induction variables -- namely the `%i` in `with (%ii -> %i = ...)`, whose
    // use is rejected above. The recovery this pass emits instead lands at the
    // krnl.get_induction_var_value query inside the body, which leaves
    // coalesceLoops' own copy dead on arrival, and dead ops between the merged
    // loop's
    // header and its nested loop make the nest imperfect, which later recipes
    // such as krnl.permute assert against. Drop them now instead of waiting for
    // canonicalization. Walked back-to-front because the recovery is a chain:
    // erasing a use can be what makes its producer dead.
    SmallVector<Operation *, 8> maybeDead;
    for (Operation &op : *outermost.getBody())
      if (isa<AffineApplyOp>(op))
        maybeDead.emplace_back(&op);
    for (Operation *op : llvm::reverse(maybeDead))
      if (op->use_empty())
        op->erase();

    // coalesceLoops turns the outermost loop into the merged loop in place and
    // erases the rest, so the outermost's induction variable is the fused
    // index. It stays valid across a later krnl.parallel too: AffineParallelOp
    // takes over the body region, so that block argument object is unchanged.
    collapseTripCounts[outermost.getInductionVar()] = tripCounts;

    // Replace the fused run with a single entry for the collapse result. The
    // erased affine.for ops must not be left behind for the caller to register.
    nestedForOps[indices.front()] =
        std::make_pair(collapseOp.getResult(), outermost.getOperation());
    nestedForOps.erase(nestedForOps.begin() + indices.front() + 1,
        nestedForOps.begin() + indices.back() + 1);
  }
  return success();
}

static LogicalResult lowerIterateOp(KrnlIterateOp &iterateOp, OpBuilder &builder,
    llvm::SmallDenseMap<Value, Operation *, 4> &refToOps,
    CollapseTripCounts &collapseTripCounts) {
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

    auto forOp = AffineForOp::create(builder, iterateOp.getLoc(), lbOperands,
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
      AffineYieldOp::create(builder, iterateOp.getLoc(), Yield.getOperands());

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
      AffineYieldOp::create(
          builder, iterateOp.getLoc(), innerForOp.getResults());
    else
      AffineYieldOp::create(builder, iterateOp.getLoc());
  }

  // The loop above stops at the "last optimized loop", leaving every loop after
  // it without a terminator. Its index `i` counts *original* dimensions, though,
  // while getNumOptimizedLoops() counts optimized loop references, so the break
  // lands in the right place only when there is one optimized loop per original
  // dimension. krnl.collapse breaks that correspondence: N original dimensions
  // become a single optimized loop, so the loops at index
  // [getNumOptimizedLoops(), rank - 2] are skipped.
  //
  // Being skipped is harmless for most of them. The innermost loop always
  // receives the iterate body's own terminator when the body is spliced in
  // below, and a skipped loop that sits strictly *inside* a collapsed band is
  // erased by affine::coalesceLoops. But the loop skipped at index
  // getNumOptimizedLoops() is the outermost of the *second* collapsed band when
  // there are two, and coalesceLoops keeps that one -- turning it into the merged
  // loop with an empty body, which LoopBodyMover::moveOne then walks into
  // (front() on an empty block).
  //
  // So fill in what the loop above skipped. A loop that already has a terminator
  // is left exactly as it made it, which keeps the iterArgs handling above and
  // every non-collapse nest bit-for-bit unchanged.
  for (int64_t i = 0; i < (int64_t)currentNestedForOps.size() - 1; i++) {
    auto forOp = llvm::cast<AffineForOp>(currentNestedForOps[i].second);
    if (forOp.getBody()->mightHaveTerminator())
      continue;
    auto innerForOp =
        llvm::cast<AffineForOp>(currentNestedForOps[i + 1].second);
    builder.setInsertionPointToEnd(forOp.getBody());
    if (forOp.getNumResults() > 0)
      AffineYieldOp::create(
          builder, iterateOp.getLoc(), innerForOp.getResults());
    else
      AffineYieldOp::create(builder, iterateOp.getLoc());
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
        auto forOp = AffineForOp::create(builder, iterateOp.getLoc(),
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
          AffineYieldOp::create(
              builder, iterateOp.getLoc(), innerForOp.getResults());
        else
          AffineYieldOp::create(builder, iterateOp.getLoc());
      }
      // Add yield for innermost affine.for with iterateOp yield value.
      auto innerForOp = newForOps.back();
      auto prevTerm = innerForOp.getBody()->getTerminator();
      builder.setInsertionPointToEnd(innerForOp.getBody());
      auto iterTerm =
          mlir::cast<KrnlYieldOp>(iterateOp.getBody()->getTerminator());
      AffineYieldOp::create(
          builder, iterateOp.getLoc(), iterTerm.getOperands());
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

  // Fuse any collapsed dimensions now, while the band built above is still a
  // pristine perfect nest, and before any loop ref is registered.
  if (failed(resolveCollapseOps(
          iterateOp, currentNestedForOps, collapseTripCounts)))
    return failure();

  for (const auto &pair : currentNestedForOps)
    refToOps.try_emplace(pair.first, pair.second);
  return success();
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
    llvm::SmallPtrSetImpl<Operation *> &opsToErase, LoopBodyMover &mover,
    CollapseTripCounts &collapseTripCounts) {

  // Recursively interpret nested operations.
  for (auto &region : op->getRegions())
    for (auto &block : region.getBlocks()) {
      auto &blockOps = block.getOperations();
      for (auto itr = blockOps.begin(); itr != blockOps.end();) {
        LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << " Call interpretOperation \n");
        if (failed(interpretOperation(&(*itr), builder, loopRefToOp, opsToErase,
                mover, collapseTripCounts)))
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
      if (failed(lowerIterateOp(
              iterateOp, builder, loopRefToOp, collapseTripCounts)))
        return failure();
      opsToErase.insert(iterateOp);
    }
    return success();
  } else if (auto collapseOp = mlir::dyn_cast_or_null<KrnlCollapseOp>(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret collapse op " << collapseOp << "\n");
    // krnl.collapse is resolved eagerly inside lowerIterateOp, where the naive
    // per-dimension band is still pristine, so there is nothing left to do here
    // beyond dropping the recipe op.
    opsToErase.insert(op);
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
    // Each listed loop ref is parallelized on its own, giving one 1-D
    // affine.parallel per ref, nested as the loops were. This does not fuse
    // their iteration spaces: to spread a single fused space over the threads,
    // krnl.collapse the dimensions first and parallelize the resulting fused
    // loop ref, which yields one affine.parallel over the product range.
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

      AffineParallelOp parallelLoop = AffineParallelOp::create(opBuilder, loc,
          ValueRange(reducedValues).getTypes(), reductionKinds,
          ArrayRef(lbsMap), lbsOperands, ArrayRef(ubsMap), ubsOperands,
          ArrayRef(loopToParallel.getStepAsInt()));
      parallelLoop.getRegion().takeBody(loopToParallel.getRegion());
      Operation *yieldOp = &parallelLoop.getBody()->back();
      yieldOp->setOperands(reducedValues);
      if (needParallelClause) {
        // The num_threads/proc_bind clause describes one parallel region, so it
        // is attached to the first loop only (expected to be the outermost one)
        // and the flag is reset for the remaining ones. Parallelizing a single
        // krnl.collapse'd loop ref sidesteps this altogether: there is then one
        // affine.parallel to carry the clause.
        needParallelClause = false;
        // Currently approach: insert after yield and then move before it.
        PatternRewriter::InsertionGuard insertGuard(builder);
        builder.setInsertionPointAfter(yieldOp);
        // Get induction variable.
        ValueRange optionalLoopIndices = parallelLoop.getIVs();
        assert(optionalLoopIndices.size() >= 1 &&
               "expected at least one loop index");
        Value parallelLoopIndex = optionalLoopIndices[0];
        Operation *newOp = KrnlParallelClauseOp::create(
            opBuilder, loc, parallelLoopIndex, numThreads, procBind);
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

    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });

  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
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
  // Connects a krnl.collapse to the krnl.get_induction_var_value queries that
  // recover its per-dimension indices.
  CollapseTripCounts collapseTripCounts;

  // Lower `define_loops` first.
  // This is will make sure affine.for created for all the defined loops first.
  // Later when lower things like nested iteratorOp and blockOp, these
  // affine.for will be ready to use.
  bool loweringFailed = false;
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
          if (failed(lowerIterateOp(
                  opToLower, builder, loopRefToOp, collapseTripCounts)))
            loweringFailed = true;
          opsToErase.insert(opToLower);
        }
      }
    }
    opsToErase.insert(defineOp);
  });
  if (loweringFailed) {
    signalPassFailure();
    return;
  }

  if (failed(interpretOperation(funcOp, builder, loopRefToOp, opsToErase, mover,
          collapseTripCounts))) {
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
            // A collapsed loop cannot itself be unrolled (KrnlUnrollOp's
            // verifier rejects that), but a query *inside* the unrolled loop may
            // still name an enclosing collapsed one, so this needs the trip
            // counts just as the final walk does.
            if (failed(lowerGetInductionVariableValueOp(
                    getIVOp, loopRefToOp, collapseTripCounts))) {
              signalPassFailure();
              return;
            }
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
      if (failed(lowerGetInductionVariableValueOp(
              getIVOp, loopRefToOp, collapseTripCounts)))
        loweringFailed = true;
      opsToErase.insert(op);
    }
  });
  if (loweringFailed) {
    signalPassFailure();
    return;
  }
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
