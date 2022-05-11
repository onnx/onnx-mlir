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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

#include "llvm/Support/Debug.h"

#include <functional>
#include <mutex>

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;

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
    llvm::Optional<KrnlMovableOp> movableOp;
    llvm::Optional<llvm::SmallVector<mlir::Value, 4>> loopsToSkip;

    explicit Movable(KrnlMovableOp op) : movableOp(op) {}
    explicit Movable(KrnlIterateOp op) {
      auto operandRange = op->getOperands();
      SmallVector<Value, 4> values;
      for (int64_t i = 0; i < op.getNumOptimizedLoops(); ++i) {
        Value val = operandRange[i];
        // Only skip non-unroll loops.
        if (llvm::all_of(val.getUsers(), [&](Operation *user) {
              return dyn_cast_or_null<KrnlUnrollOp>(user);
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
   * @param loop The Krnl Loop referring to the concrete loop sourrounding the
   * content of the movable op in the lowered IR.
   */
  void toMoveUnder(const Movable &movable, KrnlIterateOp loop) {
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
      llvm::SmallDenseMap<Value, AffineForOp, 4> &loopRefToOp,
      bool erase = true) {
    // Commented out because count is an unsigned int, and its by def >= 0.
    // assert(loopRefToOp.count(loopRef) >= 0 &&
    //       "Can't find affine for operation associated with .");
    AffineForOp forOp = loopRefToOp[loopRef];
    Block &loopBody = forOp.getLoopBody().front();
    auto insertPt = loopBody.begin();

    auto opsToTransfer = movingPlan[loopRef];
    if (erase)
      movingPlan.erase(loopRef);

    for (const Movable &transferPt : opsToTransfer) {
      assert(insertPt != loopBody.end() && "Expecting insertPt in the loop");
      assert(transferPt.loopsToSkip.hasValue() !=
                 transferPt.movableOp.hasValue() &&
             "Expecting non-equal values");
      if (transferPt.movableOp.hasValue()) {
        KrnlMovableOp movableOp = transferPt.movableOp.getValue();

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
               !dyn_cast_or_null<AffineForOp>(&*insertPt) && loopToSkip) {
          assert(dyn_cast_or_null<KrnlMovableOp>(&*insertPt) &&
                 "Expecting a KrnlMovableOp");
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
  Region &bodyRegion = root.bodyRegion();
  if (root.getNumOptimizedLoops() == 0)
    return;

  for (auto &block : bodyRegion.getBlocks()) {
    assert(!block.empty() && "IterateOp body block shouldn't be empty.");

    // Delimeter ops are delimeters of a movable chunk of code.
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
      Region &movableRegion = movableOp.region();
      Block *entryBlock = new Block();
      movableRegion.push_back(entryBlock);
      entryBlock->getOperations().splice(entryBlock->end(),
          block.getOperations(), movableBegin, delimeterOp->getIterator());
      KrnlMovableOp::ensureTerminator(
          movableRegion, builder, delimeterOp->getLoc());

      mover.toMoveUnder(LoopBodyMover::Movable(movableOp), root);
      if (auto iterateOp = dyn_cast_or_null<KrnlIterateOp>(delimeterOp))
        mover.toMoveUnder(LoopBodyMover::Movable(iterateOp), root);

      movableBeginOp = delimeterOp->getNextNode();
    }
  }
}

static void lowerGetInductionVariableValueOp(
    KrnlGetInductionVariableValueOp &getIVOp,
    llvm::SmallDenseMap<Value, AffineForOp, 4> &loopRefToOp) {
  auto zippedOperandsResults =
      llvm::zip(getIVOp->getOperands(), getIVOp->getResults());
  for (const auto &operandAndResult : zippedOperandsResults) {
    auto operand = std::get<0>(operandAndResult);
    auto result = std::get<1>(operandAndResult);
    result.replaceAllUsesWith(loopRefToOp[operand].getInductionVar());
  }
}

static void lowerIterateOp(KrnlIterateOp &iterateOp, OpBuilder &builder,
    llvm::SmallDenseMap<Value, AffineForOp, 4> &refToOps) {
  builder.setInsertionPointAfter(iterateOp);
  SmallVector<std::pair<Value, AffineForOp>, 4> currentNestedForOps;
  ArrayRef<Attribute> boundMapAttrs =
      iterateOp->getAttrOfType<ArrayAttr>(KrnlIterateOp::getBoundsAttrName())
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
    BlockArgument forIV =
        currentNestedForOps[i].second.getBody()->getArgument(0);
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
    Block *parentBlock = iterateOp->getBlock();
    Block &iterateOpEntryBlock = iterateOp.bodyRegion().front();
    // Transfer body region operations to parent region, without the terminator
    // op.
    parentBlock->getOperations().splice(iterateOp->getIterator(),
        iterateOpEntryBlock.getOperations(),
        iterateOpEntryBlock.front().getIterator(),
        iterateOpEntryBlock.getTerminator()->getIterator());
  } else {
    // Transfer krnl.iterate region to innermost for op.
    AffineForOp innermostForOp = currentNestedForOps.back().second;
    innermostForOp.region().getBlocks().clear();
    Region &innerMostRegion = innermostForOp.region();
    innerMostRegion.getBlocks().splice(
        innerMostRegion.end(), iterateOp.bodyRegion().getBlocks());
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
    llvm::SmallDenseMap<Value, AffineForOp, 4> &loopRefToOp,
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

  if (auto defineOp = dyn_cast_or_null<KrnlDefineLoopsOp>(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret define op " << defineOp << "\n");
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
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret iterate op " << iterateOp << "\n");
    // If an iterateOp has no unoptimized loop references, then we need to lower
    // them manually.
    if (opsToErase.count(op) == 0) {
      lowerIterateOp(iterateOp, builder, loopRefToOp);
      opsToErase.insert(iterateOp);
    }
    return success();
  } else if (auto blockOp = dyn_cast_or_null<KrnlBlockOp>(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret block op " << blockOp << "\n");
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
    loopRefToOp.erase(loopRefToOp.find_as(blockOp.loop()));
    loopRefToOp[blockOp.getResult(0)] = tiledLoops[0];
    loopRefToOp[blockOp.getResult(1)] = tiledLoops[1];

    opsToErase.insert(op);
    return success();
  } else if (auto permuteOp = dyn_cast_or_null<KrnlPermuteOp>(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret permute op " << permuteOp << "\n");
    // TODO(tjingrant): call it whenever an operation lowering completes.
    removeOps(opsToErase);
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
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " interpret unroll op " << unrollOp << "\n");
    // Unroll the affine for loop fully.
    Value loopRef = unrollOp.loop();
    auto loopToUnroll = loopRefToOp[loopRef];

    mover.moveOne(loopRef, loopRefToOp);

    // Interpret and remove 'krnl.get_induction_var' inside the unrolling loop
    // if any. Otherwise, we lost the trace of the loop induction variables.
    for (auto &region : loopToUnroll->getRegions())
      for (auto &block : region.getBlocks()) {
        auto &blockOps = block.getOperations();
        for (auto itr = blockOps.begin(); itr != blockOps.end(); ++itr) {
          Operation *genericOp = &(*itr);
          if (auto getIVOp = dyn_cast_or_null<KrnlGetInductionVariableValueOp>(
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
    opsToErase.insert(op);
    return success();
  }

  return success();
}

AffineTypeConverter::AffineTypeConverter() {
  // The order of type conversion is important: later ones are tried earlier.
  addConversion([](Type type) { return type; });

  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return llvm::None;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return llvm::None;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

//===----------------------------------------------------------------------===//
// ConvertKrnlToAffinePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the krnl dialect operations.
/// At this stage the dialect will contain standard operations as well like
/// add and multiply, this pass will leave these operations intact.
struct ConvertKrnlToAffinePass
    : public PassWrapper<ConvertKrnlToAffinePass, OperationPass<FuncOp>> {

  StringRef getArgument() const override { return "convert-krnl-to-affine"; }

  StringRef getDescription() const override { return "Lower Krnl dialect."; }

  void runOnOperation() final;
};

void ConvertKrnlToAffinePass::runOnOperation() {
  FuncOp funcOp = getOperation();
  if (funcOp.body().empty()) // external function: nothing to do
    return;

  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(
      &getContext(), dataLayoutAnalysis.getAtOrAbove(funcOp));
  options.emitCWrappers = true;

  // Move invariant instructions outside of the loops as many as possible. This
  // helps make loops perfectly nested, which facilitates transformations.
  funcOp.walk([&](KrnlIterateOp loopOp) {
    LogicalResult res =
        moveLoopInvariantCode(cast<LoopLikeOpInterface>(loopOp.getOperation()));
    assert(succeeded(res) && "failed to move loop invariant code");
  });

  // We use the end of the function body as a staging area for movable ops.
  builder.setInsertionPoint(
      &funcOp.body().front(), funcOp.body().front().without_terminator().end());
  LoopBodyMover mover;
  funcOp.walk(
      [&](KrnlIterateOp op) { markLoopBodyAsMovable(op, builder, mover); });

  // Interpret krnl dialect operations while looping recursively through
  // operations within the current function, note that erasing operations
  // while iterating is tricky because it can invalidate the iterator, so we
  // collect the operations to be erased in a small ptr set `opsToErase`, and
  // only erase after iteration completes.
  llvm::SmallDenseMap<Value, AffineForOp, 4> loopRefToOp;
  llvm::SmallPtrSet<Operation *, 4> opsToErase;
  if (failed(interpretOperation(
          funcOp, builder, loopRefToOp, opsToErase, mover))) {
    signalPassFailure();
    return;
  }

  funcOp->walk([&](Operation *op) {
    if (SpecializedKernelOpInterface kernelOp =
            dyn_cast<SpecializedKernelOpInterface>(op)) {
      OperandRange loopRefs = kernelOp.getLoopRefs();
      for (auto loopRef : loopRefs)
        opsToErase.insert(loopRefToOp[loopRef]);
      kernelOp.getLoopRefs().clear();
    }
    if (auto getIVOp = dyn_cast_or_null<KrnlGetInductionVariableValueOp>(op)) {
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

  // krnl.dim operations must be lowered prior to this pass.
  target.addIllegalOp<KrnlDimOp>();
  target.addIllegalOp<KrnlMatMulOp>();
  target.addIllegalOp<KrnlCopyToBufferOp>();
  target.addIllegalOp<KrnlCopyFromBufferOp>();
  target.addIllegalOp<KrnlMemsetOp>();
  target.addLegalOp<AffineYieldOp>();
  target.addLegalOp<AffineLoadOp>();
  target.addLegalOp<AffineStoreOp>();
  target.addLegalOp<KrnlVectorTypeCastOp>();
  target.addLegalDialect<mlir::AffineDialect, mlir::arith::ArithmeticDialect,
      mlir::memref::MemRefDialect, mlir::StandardOpsDialect,
      mlir::vector::VectorDialect>();

  // Patterns.
  RewritePatternSet patterns(ctx);
  AffineTypeConverter typeConverter;

  populateKrnlToAffineConversion(typeConverter, patterns, ctx);

  // Create list for recording the <loop, unroll factor> pairs associated with
  // this function.
  UnrollAndJamList *currUnrollAndJamList = new UnrollAndJamList();
  Operation *currFuncOp = funcOp.getOperation();
  {
    const std::lock_guard<std::mutex> lock(unrollAndJamMutex);
    unrollAndJamMap[currFuncOp] = currUnrollAndJamList;
  }

  DenseSet<Operation *> unconverted;
  if (failed(applyPartialConversion(
          getOperation(), target, std::move(patterns), &unconverted))) {
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

void populateKrnlToAffineConversion(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  krnl::populateLoweringKrnlCopyFromBufferOpPattern(
      typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlCopyToBufferOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlLoadOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlStoreOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlMatmultOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlMemsetOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlTerminatorOpPattern(typeConverter, patterns, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
