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
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/MLIRDialectBuilder.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"
#include "src/Support/KrnlSupport.hpp"

#include "llvm/Support/Debug.h"

#include <functional>
#include <mutex>

static constexpr int BUFFER_ALIGN = 64;

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;

// We use here a Affine builder that generates Krnl Load and Store ops instead
// of the affine memory ops directly. This is because we can still generrate
// Krnl Ops while lowring the dialect, and the big advantage of the Krnl memory
// operations is that they distinguish themselves if they are affine or not.
using AffineBuilderKrnlMem = GenericAffineBuilder<KrnlLoadOp, KrnlStoreOp>;

namespace {
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
               !dyn_cast_or_null<AffineForOp>(&*insertPt) && loopToSkip) {
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

// To assist unroll and jam
using UnrollAndJamRecord = std::pair<AffineForOp, int64_t>;
using UnrollAndJamList = SmallVector<UnrollAndJamRecord, 4>;
using UnrollAndJamMap = std::map<Operation *, UnrollAndJamList *>;
UnrollAndJamMap unrollAndJamMap;
std::mutex unrollAndJamMutex;

/// Retrieve function which contains the current operation.
Operation *getContainingFunction(Operation *op) {
  Operation *parentFuncOp = op->getParentOp();
  // While parent is not a FuncOp and its cast to a FuncOp is null.
  while (!llvm::dyn_cast_or_null<FuncOp>(parentFuncOp))
    parentFuncOp = parentFuncOp->getParentOp();
  return parentFuncOp;
}

UnrollAndJamList *getUnrollAndJamList(Operation *op) {
  Operation *currFuncOp = getContainingFunction(op);
  assert(currFuncOp && "function expected");
  const std::lock_guard<std::mutex> lock(unrollAndJamMutex);
  UnrollAndJamList *currUnrollAndJamList = unrollAndJamMap[currFuncOp];
  assert(currUnrollAndJamList && "expected list for function");
  return currUnrollAndJamList;
}

void removeOps(llvm::SmallPtrSetImpl<Operation *> &opsToErase) {
  // Remove lowered operations topologically; if ops are not removed
  // topologically, memory error will occur.
  size_t numOpsToRemove = opsToErase.size();
  // Given N operations to remove topologically, and that we remove
  // at least one operation during each pass through opsToErase, we
  // can only have a maximum of N passes through opsToErase.
  for (size_t i = 0; i < numOpsToRemove; i++) {
    for (auto op : opsToErase) {
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

void lowerGetInductionVariableValueOp(KrnlGetInductionVariableValueOp &getIVOp,
    llvm::SmallDenseMap<Value, AffineForOp, 4> &loopRefToOp) {
  auto zippedOperandsResults =
      llvm::zip(getIVOp->getOperands(), getIVOp->getResults());
  for (const auto &operandAndResult : zippedOperandsResults) {
    auto operand = std::get<0>(operandAndResult);
    auto result = std::get<1>(operandAndResult);
    result.replaceAllUsesWith(loopRefToOp[operand].getInductionVar());
  }
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

LogicalResult interpretOperation(Operation *op, OpBuilder &builder,
    llvm::SmallDenseMap<Value, AffineForOp, 4> &loopRefToOp,
    llvm::SmallPtrSetImpl<Operation *> &opsToErase, LoopBodyMover &mover) {
  // Recursively interpret nested operations.
  for (auto &region : op->getRegions())
    for (auto &block : region.getBlocks()) {
      auto &blockOps = block.getOperations();
      for (auto itr = blockOps.begin(); itr != blockOps.end();) {
        LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << " Call interpretOperation \n");
        if (failed(interpretOperation(
                &(*itr), builder, loopRefToOp, opsToErase, mover))) {
          return failure();
        } else {
          ++itr;
        }
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
    auto loopRef = unrollOp.loop();
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
      rewriter.replaceOpWithNewOp<memref::LoadOp>(op, memref, indices);

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
      rewriter.replaceOpWithNewOp<memref::StoreOp>(op, value, memref, indices);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: Krnl MatMul operation.
//===----------------------------------------------------------------------===//

// Affine expressions compared to >= 0
static IndexExpr isFullTile(IndexExpr UB, IndexExpr block, IndexExpr GI) {
  // Determine if the current tile is full. It is full if the begining of
  // the tile (nGI) is smaller or equal to UB - bloc, namely
  //   PredicateIndexExpr nIsFullTile = (nGI <= (nUB - nBlock));
  // However, if UB is divisible by Block, then its full no matter what.
  if (UB.isLiteral() && (UB.getLiteral() % block.getLiteral() == 0)) {
    // Last tile is guaranteed to be full because UB is divisable by block.
    return LiteralIndexExpr(1); // 1 >= 0 is true
  }
  // true if GI <= (UB - block), namely UB - block - GI >= 0
  IndexExpr res = UB - block - GI;
  return res;
}

static IndexExpr trip(IndexExpr UB, IndexExpr block, IndexExpr GI) {
  // Trip count in general: min(UB - GI, Block).
  UB.debugPrint("trip UB");
  block.debugPrint("trip block");
  GI.debugPrint("trip GI");
  //   IndexExpr nTrip = IndexExpr::min(nUB - nGI, nBlock);
  if (UB.isLiteral() && (UB.getLiteral() % block.getLiteral() == 0)) {
    // Last tile is guaranteed to be full, so trip is always full.
    return block;
  }
  return IndexExpr::min(UB - GI, block);
}

static IndexExpr partialTrip(IndexExpr UB, IndexExpr block, IndexExpr GI) {
  // Trip count for partial tiles: leftover = UB - GI in general. If UB is
  // known at compile time, then without loss of generality, leftover = (UB-
  // GI) % Block, and since GI is by definition a multiple of Block (GI is
  // index at begining of tile), then leftover = UB % Block.
  //   IndexExpr nPartialTrip = nUB.isLiteral() ? nUB % nBlock : nUB - nGI;
  if (UB.isLiteral()) {
    IndexExpr partialTrip = UB % block;
    assert(partialTrip.isLiteral() && "op on 2 literals has to be literal");
    return partialTrip;
  }
  // don't have to take the mod since we know we have a partial tile already.
  return UB - GI;
}

ATTRIBUTE(unused)
static IndexExpr startInBuffer(
    IndexExpr globalStart, IndexExpr tileSize, IndexExpr globalUB) {
  if (tileSize.isLiteral() && globalUB.isLiteral() &&
      tileSize.getLiteral() == globalUB.getLiteral()) {
    // No need to take the mod when the tile size is the entire data.
    return globalStart;
  }
  return globalStart % tileSize;
}

// KrnlMatmul will be lowered to vector and affine expressions
class KrnlMatmulLowering : public OpRewritePattern<KrnlMatMulOp> {

public:
  using OpRewritePattern<KrnlMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlMatMulOp op, PatternRewriter &rewriter) const override {
    KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);
    // Option.
    bool fullUnrollAndJam = op.unroll();

    // Operands and types.
    Type elementType =
        operandAdaptor.A().getType().cast<MemRefType>().getElementType();
    bool simdize = op.simdize();
    // Init scope and emit constants.
    Location loc = op.getLoc();
    AffineBuilderKrnlMem createAffine(rewriter, loc);
    IndexExprScope indexScope(createAffine);

    // Gather A, B, C tile sizes.
    SmallVector<IndexExpr, 2> aTileSize, bTileSize, cTileSize;
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(operandAdaptor.C());
    MemRefBoundsIndexCapture aBounds(A), bBounds(B), cBounds(C);
    int64_t aRank(aBounds.getRank()), bRank(bBounds.getRank()),
        cRank(cBounds.getRank());
    // Tile sizes for A/B/C are determined by their memref unless explicitly
    // specified by an optional argument. That allows A/B/C memrefs to be
    // padded if needed for SIMD/unroll and jam, for example.
    ArrayAttributeIndexCapture aSizeCapture(op.aTileSizeAttr());
    if (aSizeCapture.size())
      aTileSize = {aSizeCapture.getLiteral(0), aSizeCapture.getLiteral(1)};
    else
      aTileSize = {aBounds.getSymbol(aRank - 2), aBounds.getSymbol(aRank - 1)};
    ArrayAttributeIndexCapture bSizeCapture(op.bTileSizeAttr());
    if (bSizeCapture.size())
      bTileSize = {bSizeCapture.getLiteral(0), bSizeCapture.getLiteral(1)};
    else
      bTileSize = {bBounds.getSymbol(bRank - 2), bBounds.getSymbol(bRank - 1)};
    ArrayAttributeIndexCapture cSizeCapture(op.cTileSizeAttr());
    if (cSizeCapture.size())
      cTileSize = {cSizeCapture.getLiteral(0), cSizeCapture.getLiteral(1)};
    else
      cTileSize = {cBounds.getSymbol(cRank - 2), cBounds.getSymbol(cRank - 1)};

    // Gather N, M, K compute tile size. This is the size of the computations,
    // if the tile is full. Because computation in the buffers could be further
    // subtiled, the default size can be overridden from the tile sizes using
    // the computeTileSize attribute. Tiles may not be full if they are at the
    // outer boundaries of the original data.
    IndexExpr iComputeTileSize = cTileSize[0];
    IndexExpr jComputeTileSize = cTileSize[1];
    IndexExpr kComputeTileSize = aTileSize[1];
    ArrayAttributeIndexCapture computeSizeCapture(op.computeTileSizeAttr());
    if (computeSizeCapture.size()) {
      iComputeTileSize = computeSizeCapture.getLiteral(0);
      jComputeTileSize = computeSizeCapture.getLiteral(1);
      kComputeTileSize = computeSizeCapture.getLiteral(2);
    }

    // If we simdize, its along M for the full compute tile.
    IndexExpr vectorLen = jComputeTileSize;
    if (!vectorLen.isLiteral()) {
      // Cannot simdize if the vector length is not a compile time constant.
      simdize = false;
      LLVM_DEBUG(llvm::dbgs() << "Matmul: No simd due to vl not a literal\n");
    }
    if (!bBounds.isLiteral(bRank - 1) || !cBounds.isLiteral(cRank - 1)) {
      // Cannot simdize if the last dim of B or C are not constant.
      simdize = false;
      LLVM_DEBUG(llvm::dbgs()
                 << "Matmul: No simd due to B & C last dim not a literal\n");
    }
    if (simdize) {
      int64_t VL = vectorLen.getLiteral();
      if (bBounds.getShape(bRank - 1) % VL != 0 ||
          cBounds.getShape(cRank - 1) % VL != 0) {
        // If the memref of B and C are not multiple of the vector length in
        // their last dim, then we cannot simdize either.
        simdize = false;
        LLVM_DEBUG(
            llvm::dbgs()
            << "Matmul: No simd due to B & C last dim not a multiple of VL\n");
      }
    }
    if (!simdize)
      vectorLen = LiteralIndexExpr(1);

    // Now get global start indices, which would define the first element of the
    // tiles in the original computations.
    DimIndexExpr iComputeStart(operandAdaptor.iComputeStart()),
        jComputeStart(operandAdaptor.jComputeStart()),
        kComputeStart(operandAdaptor.kComputeStart());
    // And get the global upper bound of the original computations.
    SymbolIndexExpr iGlobalUB(operandAdaptor.iGlobalUB()),
        jGlobalUB(operandAdaptor.jGlobalUB()),
        kGlobalUB(operandAdaptor.kGlobalUB());
    // A[i, k];
    SmallVector<IndexExpr, 4> aStart, bStart, cStart;
    for (int t = 0; t < aRank - 2; t++)
      aStart.emplace_back(SymbolIndexExpr(operandAdaptor.aMemStart()[t]));
    aStart.emplace_back(
        iComputeStart - DimIndexExpr(operandAdaptor.aMemStart()[aRank - 2]));
    aStart.emplace_back(
        kComputeStart - DimIndexExpr(operandAdaptor.aMemStart()[aRank - 1]));
    // B[k, j];
    for (int t = 0; t < bRank - 2; t++)
      bStart.emplace_back(SymbolIndexExpr(operandAdaptor.bMemStart()[t]));
    bStart.emplace_back(
        kComputeStart - DimIndexExpr(operandAdaptor.bMemStart()[bRank - 2]));
    bStart.emplace_back(
        jComputeStart - DimIndexExpr(operandAdaptor.bMemStart()[bRank - 1]));
    // C[i, j]
    for (int t = 0; t < cRank - 2; t++)
      cStart.emplace_back(SymbolIndexExpr(operandAdaptor.cMemStart()[t]));
    cStart.emplace_back(
        iComputeStart - DimIndexExpr(operandAdaptor.cMemStart()[cRank - 2]));
    cStart.emplace_back(
        jComputeStart - DimIndexExpr(operandAdaptor.cMemStart()[cRank - 1]));

    SmallVector<IndexExpr, 4> bVecStart(bStart), cVecStart(cStart);
    bVecStart[bRank - 1] = bStart[bRank - 1].floorDiv(vectorLen);
    cVecStart[cRank - 1] = cStart[cRank - 1].floorDiv(vectorLen);

    // Now determine if we have full/partial tiles. This is determined by the
    // outer dimensions of the original computations, as by definition tiling
    // within the buffer always results in full tiles. In other words, partial
    // tiles only occurs because of "runing out" of the original data.
    IndexExpr iIsFullTile =
        isFullTile(iGlobalUB, iComputeTileSize, iComputeStart);
    IndexExpr jIsFullTile =
        isFullTile(jGlobalUB, jComputeTileSize, jComputeStart);
    IndexExpr kIsFullTile =
        isFullTile(kGlobalUB, kComputeTileSize, kComputeStart);
    SmallVector<IndexExpr, 3> allFullTiles = {
        iIsFullTile, jIsFullTile, kIsFullTile};

    SmallVector<IndexExpr, 1> jFullTiles = {jIsFullTile};
    // And if the tiles are not full, determine how many elements to compute.
    // With overcompute, this could be relaxed.
    IndexExpr iTrip = trip(
        iGlobalUB, iComputeTileSize, iComputeStart); // May or may not be full.
    IndexExpr jTrip = trip(
        jGlobalUB, jComputeTileSize, jComputeStart); // May or may not be full.
    IndexExpr kTrip = trip(
        kGlobalUB, kComputeTileSize, kComputeStart); // May or may not be full.
    IndexExpr jPartialTrip =
        partialTrip(jGlobalUB, jComputeTileSize, jComputeStart);

    if (simdize) {
      // SIMD code generator.
      // clang-format off
      createAffine.ifThenElse(indexScope, allFullTiles,
        /* then full */ [&](AffineBuilderKrnlMem &createAffine) {
        genSimd(rewriter, loc, op, elementType, aStart, bVecStart, cVecStart,
          iComputeTileSize, jComputeTileSize, kComputeTileSize,
          vectorLen, fullUnrollAndJam); 
      }, /* has some partial tiles */ [&](AffineBuilderKrnlMem &createAffine) {
        // Trip regardless of full/partial for N & K
        // Test if SIMD dim (M) is full.
        createAffine.ifThenElse(indexScope, jFullTiles,
          /* full SIMD */ [&](AffineBuilderKrnlMem &createAffine) {
          genSimd(rewriter, loc, op, elementType, aStart, bVecStart, cVecStart,
            iTrip, jComputeTileSize, kTrip, vectorLen, false);
        }, /* else partial SIMD */ [&](AffineBuilderKrnlMem &createAffine) {
          if (false && jPartialTrip.isLiteral() && jPartialTrip.getLiteral() >=2) {
            // has a known trip count along the simd dimension of at least 2
            // elements, use simd again.
            genSimd(rewriter, loc, op, elementType, aStart, bVecStart, cVecStart,
              iTrip, jPartialTrip, kTrip, vectorLen, false);
          } else {
            genScalar(rewriter, op, elementType, aStart, bStart,  cStart,
              iTrip, jPartialTrip, kTrip, false);
          }
        });
      });
      // clang-format on
    } else {
      // Scalar code generator.
      // clang-format off
      createAffine.ifThenElse(indexScope, allFullTiles,
        /* then full */ [&](AffineBuilderKrnlMem &createAffine) {
        genScalar(rewriter, op, elementType, aStart, bStart, cStart,
          iComputeTileSize, jComputeTileSize, kComputeTileSize,
          fullUnrollAndJam); 
      }, /* else partial */ [&](AffineBuilderKrnlMem &createAffine) {
        genScalar(rewriter, op, elementType, aStart, bStart, cStart,
          iTrip, jTrip, kTrip, false);
      });
      // clang-format on
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  void genScalar(PatternRewriter &rewriter, KrnlMatMulOp op, Type elementType,
      ArrayRef<IndexExpr> aStart, ArrayRef<IndexExpr> bStart,
      ArrayRef<IndexExpr> cStart, IndexExpr I, IndexExpr J, IndexExpr K,
      bool unrollJam) const {
    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor(op);
    Location loc = op.getLoc();
    AffineBuilderKrnlMem createAffine(rewriter, loc);
    MemRefBuilder createMemRef(createAffine);

    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(operandAdaptor.C());
    int64_t aRank(aStart.size()), bRank(bStart.size()), cRank(cStart.size());
    int64_t unrollFactor = (unrollJam && J.isLiteral()) ? J.getLiteral() : 1;
    // Have to privatize CTmpType by unroll factor (1 if none).
    MemRefType CTmpType = MemRefType::get({unrollFactor}, elementType);
    assert(BUFFER_ALIGN >= gDefaultAllocAlign);
    Value TmpC = createMemRef.alignedAlloc(CTmpType, BUFFER_ALIGN);

    // For i, j loops.
    LiteralIndexExpr zero(0);
    Value jSaved;
    createAffine.forIE(
        zero, I, 1, [&](AffineBuilderKrnlMem &createAffine, Value i) {
          createAffine.forIE(
              zero, J, 1, [&](AffineBuilderKrnlMem &createAffine, Value j) {
                MathBuilder createMath(createAffine);
                // Defines induction variables, and possibly initialize C.
                jSaved = j;
                // Alloc and init temp c storage.
                SmallVector<Value, 4> cAccess;
                // CC(i + cStart0.getValue(), j + cStart1.getValue());
                IndexExpr::getValues(cStart, cAccess);
                cAccess[cRank - 2] = createMath.add(i, cAccess[cRank - 2]);
                cAccess[cRank - 1] = createMath.add(j, cAccess[cRank - 1]);
                Value initVal = createAffine.load(C, cAccess);
                Value tmpCAccess = (unrollFactor > 1) ? j : zero.getValue();
                createAffine.store(initVal, TmpC, tmpCAccess);
                // TTmpC() = affine_load(C, cAccess);
                // Sum over k.
                createAffine.forIE(zero, K, 1,
                    [&](AffineBuilderKrnlMem &createAffine, Value k) {
                      MathBuilder createMath(createAffine);
                      SmallVector<Value, 4> aAccess, bAccess;
                      // AA(i + aStart0.getValue(), k + aStart1.getValue())
                      IndexExpr::getValues(aStart, aAccess);
                      aAccess[aRank - 2] =
                          createMath.add(i, aAccess[aRank - 2]);
                      aAccess[aRank - 1] =
                          createMath.add(k, aAccess[aRank - 1]);
                      Value a = createAffine.load(A, aAccess);
                      // BB(k + bStart0.getValue(), j + bStart1.getValue())
                      IndexExpr::getValues(bStart, bAccess);
                      bAccess[bRank - 2] =
                          createMath.add(k, bAccess[bRank - 2]);
                      bAccess[bRank - 1] =
                          createMath.add(j, bAccess[bRank - 1]);
                      Value b = createAffine.load(B, bAccess);
                      Value res = createMath.mul(a, b);
                      res = createMath.add(
                          res, createAffine.load(TmpC, tmpCAccess));
                      createAffine.store(res, TmpC, tmpCAccess);
                      // TTmpC() = a * b + TTmpC();
                    });
                // Store temp result into C(i, j)
                Value finalVal = createAffine.load(TmpC, tmpCAccess);
                createAffine.store(finalVal, C, cAccess);
                // affine_store(TTmpC(), C, cAccess);
              });
        });
    if (unrollJam && J.isLiteral()) {
      UnrollAndJamRecord record(
          getForInductionVarOwner(jSaved), J.getLiteral());
      getUnrollAndJamList(op.getOperation())->emplace_back(record);
    }
  }

  void genSimd(PatternRewriter &rewriter, Location loc, KrnlMatMulOp op,
      Type elementType, ArrayRef<IndexExpr> aStart, ArrayRef<IndexExpr> bStart,
      ArrayRef<IndexExpr> cStart, IndexExpr I, IndexExpr J, IndexExpr K,
      IndexExpr vectorLen, bool unrollJam) const {
    // can simdize only if K is compile time
    assert(J.isLiteral() &&
           "can only simdize with compile time blocking factor on simd axis");
    AffineBuilderKrnlMem createAffine(rewriter, loc);
    MemRefBuilder createMemRef(rewriter, loc);
    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(operandAdaptor.C());
    int64_t aRank(aStart.size()), bRank(bStart.size()), cRank(cStart.size());

    // Generate the vector type conversions.
    int64_t VL = vectorLen.getLiteral();
    VectorType vecType = VectorType::get({VL}, elementType);
    int64_t unrollFactor = (unrollJam && I.isLiteral()) ? I.getLiteral() : 1;
    // Have to privatize CTmpType by unroll factor (1 if none).
    MemRefType CTmpType = MemRefType::get({unrollFactor}, vecType);
    KrnlBuilder createKrnl(rewriter, loc);
    Value vecB = createKrnl.vectorTypeCast(B, VL);
    Value vecC = createKrnl.vectorTypeCast(C, VL);
    assert(BUFFER_ALIGN >= gDefaultAllocAlign);
    Value TmpC = createMemRef.alignedAlloca(CTmpType, BUFFER_ALIGN);

    // Iterates over the I indices (j are simd dim).
    Value iSaved, kSaved;
    LiteralIndexExpr zero(0);
    createAffine.forIE(
        zero, I, 1, [&](AffineBuilderKrnlMem &createAffine, Value i) {
          MathBuilder createMath(createAffine);
          iSaved = i; // Saved for unroll and jam.
          // Alloca temp vector TmpC and save C(i)/0.0 into it.
          SmallVector<Value, 4> cAccess;
          // cAccess = {i + cStart0.getValue(), cStart1.getValue()};
          IndexExpr::getValues(cStart, cAccess);
          cAccess[cRank - 2] = createMath.add(i, cAccess[cRank - 2]);
          Value initVal = createAffine.load(vecC, cAccess);
          Value tmpCAccess = (unrollFactor > 1) ? i : zero.getValue();
          createAffine.store(initVal, TmpC, tmpCAccess);
          // Sum over k.
          createAffine.forIE(
              zero, K, 1, [&](AffineBuilderKrnlMem &createAffine, Value k) {
                MathBuilder createMath(createAffine);
                kSaved = k;
                // Value a = AA(i + aStart0.getValue(), k + aStart1.getValue());
                SmallVector<Value, 4> aAccess, bAccess;
                IndexExpr::getValues(aStart, aAccess);
                aAccess[aRank - 2] = createMath.add(i, aAccess[aRank - 2]);
                aAccess[aRank - 1] = createMath.add(k, aAccess[aRank - 1]);
                Value a = createAffine.load(A, aAccess);
                // Value va = vector_broadcast(vecType, a);
                Value va =
                    createAffine.getBuilder().create<vector::BroadcastOp>(
                        createAffine.getLoc(), vecType, a);
                // bAccess = {k + bStart0.getValue(), bStart1.getValue()};
                IndexExpr::getValues(bStart, bAccess);
                bAccess[bRank - 2] = createMath.add(k, bAccess[bRank - 2]);
                Value vb = createAffine.load(vecB, bAccess);
                // TTmpC() = vector_fma(va, vb, TTmpC());
                Value tmpVal = createAffine.load(TmpC, tmpCAccess);
                Value res = createAffine.getBuilder().create<vector::FMAOp>(
                    createAffine.getLoc(), va, vb, tmpVal);
                createAffine.store(res, TmpC, tmpCAccess);
              });
          // Store temp result into C(i)
          Value tmpResults = createAffine.load(TmpC, tmpCAccess);
          int64_t JLit = J.getLiteral();
          if (JLit != VL) {
            // create vector constant
            SmallVector<int64_t, 8> mask;
            for (int64_t i = 0; i < VL; i++)
              mask.emplace_back((i < JLit) ? i : VL + i);
            // permute
            Value originalCvec = createAffine.load(vecC, cAccess);
            tmpResults = createAffine.getBuilder().create<vector::ShuffleOp>(
                createAffine.getLoc(), tmpResults, originalCvec, mask);
          }
          // CCvec(i + CStart0.getValue(), CStart1.getValue()) = tmpResults;
          createAffine.store(tmpResults, vecC, cAccess);
        });

    if (unrollJam && (I.isLiteral() || K.isLiteral())) {
      auto list = getUnrollAndJamList(op.getOperation());
      if (K.isLiteral()) {
        int64_t kUnroll = K.getLiteral();
        // We know there is no unrolling along I, make a bigger cutoff.
        int64_t cutoff = (!I.isLiteral() || I.getLiteral() < 2) ? 8 : 4;
        if (kUnroll >= cutoff) {
          // When kUnroll is too big, reduce it by a divisor.
          for (int64_t m = cutoff; m >= 1; --m) {
            if (kUnroll % m == 0) {
              kUnroll = m;
              break;
            }
          }
        }
        if (kUnroll > 1) {
          LLVM_DEBUG(
              llvm::dbgs() << "Matmul: unroll k by " << kUnroll << "\n";);
          UnrollAndJamRecord record(getForInductionVarOwner(kSaved), kUnroll);
          list->emplace_back(record);
        }
      }
      if (I.isLiteral() && I.getLiteral() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Matmul: unroll i by " << (int)I.getLiteral() << "\n");
        UnrollAndJamRecord record(
            getForInductionVarOwner(iSaved), I.getLiteral());
        list->emplace_back(record);
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: Krnl Copy to Buffer operation.
//===----------------------------------------------------------------------===//

class KrnlCopyToBufferLowering : public OpRewritePattern<KrnlCopyToBufferOp> {
public:
  using OpRewritePattern<KrnlCopyToBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlCopyToBufferOp op, PatternRewriter &rewriter) const override {
    // Get info from operands.
    KrnlCopyToBufferOpAdaptor operandAdaptor = KrnlCopyToBufferOpAdaptor(op);
    Value buffMemref(operandAdaptor.buffer());
    Value sourceMemref(operandAdaptor.source());
    ValueRange startVals(operandAdaptor.starts());
    Value padVal(operandAdaptor.padValue());
    int64_t srcRank =
        sourceMemref.getType().cast<MemRefType>().getShape().size();
    int64_t buffRank =
        buffMemref.getType().cast<MemRefType>().getShape().size();
    int64_t srcOffset = srcRank - buffRank;
    assert(srcOffset >= 0 && "offset expected non negative");
    Location loc = op.getLoc();
    AffineBuilderKrnlMem createAffine(rewriter, loc);
    IndexExprScope indexScope(createAffine);
    SmallVector<IndexExpr, 4> starts, bufferReadUBs, bufferPadUBs;
    MemRefBoundsIndexCapture buffBounds(buffMemref);
    MemRefBoundsIndexCapture sourceBounds(sourceMemref);
    getIndexExprList<DimIndexExpr>(startVals, starts);
    ArrayAttributeIndexCapture padCapture(op.padToNextAttr(), 1);
    ArrayAttributeIndexCapture readSizeCapture(op.tileSizeAttr());
    // Handle possible transpose by having an indirect array for indices
    // used in conjunction with source.
    SmallVector<int64_t, 4> srcIndexMap, srcLoopMap;
    generateIndexMap(srcIndexMap, srcRank, op.transpose());
    generateIndexMap(srcLoopMap, buffRank, op.transpose());

    // Overread not currently used, will if we simdize reads or
    // unroll and jam loops.
    // ArrayAttributeIndexCapture overCapture(op.overreadToNextAttr(), 1);

    // Determine here bufferReadUBs, which determine how many values of source
    // memeref to copy into the buffer. Also determine bufferPadUBs, which is
    // the upper bound past bufferReadUBs that must be padded.
    // This is only done on the dimensions shared between src memref and buffer.
    LiteralIndexExpr zero(0);
    for (long buffIndex = 0; buffIndex < buffRank; ++buffIndex) {
      long srcIndex = srcIndexMap[srcOffset + buffIndex];
      // Compute how many values to read.
      IndexExpr sourceBound =
          sourceBounds.getSymbol(srcIndex); // Source memref size.
      IndexExpr blockSize =
          buffBounds.getSymbol(buffIndex); // Buffer memref size.
      if (readSizeCapture.size()) {
        int64_t memSize = blockSize.getLiteral();
        blockSize = readSizeCapture.getLiteral(buffIndex); // Size from param.
        assert(blockSize.getLiteral() <= memSize &&
               "readTileSize cannot be larger than the buffer size");
      }
      IndexExpr startGI =
          starts[srcIndex]; // Global index in source memref of tile.
      IndexExpr bufferRead = trip(sourceBound, blockSize, startGI);
      bufferRead.debugPrint("buffer read");
      bufferReadUBs.emplace_back(bufferRead);
      // Determine the UB until which to pad
      IndexExpr padToNext = padCapture.getLiteral(buffIndex);
      int64_t padToNextLit =
          padToNext.getLiteral(); // Will assert if undefined.
      int64_t blockSizeLit = blockSize.getLiteral(); // Will assert if not lit.
      if (bufferRead.isLiteralAndIdenticalTo(blockSizeLit)) {
        // Read the full buffer already, nothing to do.
        bufferPadUBs.emplace_back(zero);
      } else if (bufferRead.isLiteral() &&
                 bufferRead.getLiteral() % padToNextLit == 0) {
        // We are already reading to the end of a line.
        bufferPadUBs.emplace_back(zero);
      } else if (padToNextLit == 1) {
        // Add pad % 1... namely no pad, nothing to do.
        bufferPadUBs.emplace_back(zero);
      } else if (padToNextLit == blockSizeLit) {
        // Pad to end.
        bufferPadUBs.emplace_back(blockSize);
      } else {
        assert(padToNextLit > 1 && padToNextLit < blockSizeLit &&
               "out of range padToLit");
        IndexExpr newPadUB = (bufferRead.ceilDiv(padToNext)) * padToNext;
        bufferPadUBs.emplace_back(newPadUB);
      }
    }
    SmallVector<Value, 4> loopIndices;
    genCopyLoops(createAffine, &indexScope, buffMemref, sourceMemref,
        srcLoopMap, padVal, zero, starts, bufferReadUBs, bufferPadUBs,
        loopIndices, 0, buffRank, false);
    rewriter.eraseOp(op);
    return success();
  }

  void genCopyLoops(AffineBuilderKrnlMem &createAffine,
      IndexExprScope *enclosingScope, Value buffMemref, Value sourceMemref,
      SmallVectorImpl<int64_t> &srcLoopMap, Value padVal, IndexExpr zero,
      SmallVectorImpl<IndexExpr> &starts, SmallVectorImpl<IndexExpr> &readUBs,
      SmallVectorImpl<IndexExpr> &padUBs, SmallVectorImpl<Value> &loopIndices,
      int64_t i, int64_t buffRank, bool padPhase) const {
    if (i == buffRank) {
      // create new scope and import index expressions
      IndexExprScope currScope(createAffine, enclosingScope);
      KrnlBuilder createKrnl(createAffine);
      SmallVector<IndexExpr, 4> currLoopIndices, currStarts;
      getIndexExprList<DimIndexExpr>(loopIndices, currLoopIndices);
      if (!padPhase) {
        SmallVector<IndexExpr, 4> currLoadIndices;
        getIndexExprList<DimIndexExpr>(starts, currStarts);
        int64_t srcRank = starts.size();
        int64_t srcOffset = srcRank - buffRank;
        for (long srcIndex = 0; srcIndex < srcRank; ++srcIndex) {
          if (srcIndex < srcOffset) {
            // Dimensions that are unique to source memref, just use starts.
            currLoadIndices.emplace_back(currStarts[srcIndex]);
          } else {
            // Dimensions that are shared by source memref & buffer, add loop
            // indices to starts.
            int64_t buffIndex = srcIndex - srcOffset;
            currLoadIndices.emplace_back(
                currLoopIndices[srcLoopMap[buffIndex]] + currStarts[srcIndex]);
          }
        }
        Value sourceVal = createKrnl.loadIE(sourceMemref, currLoadIndices);
        createKrnl.storeIE(sourceVal, buffMemref, currLoopIndices);
      } else {
        createKrnl.storeIE(padVal, buffMemref, currLoopIndices);
      }
    } else {
      readUBs[i].getValue();
      if (readUBs[i].isLiteralAndIdenticalTo(0)) {
        // Nothing to read, skip.
      } else {
        createAffine.forIE(zero, readUBs[i], 1,
            [&](AffineBuilderKrnlMem &createAffine, Value index) {
              loopIndices.emplace_back(index);
              genCopyLoops(createAffine, enclosingScope, buffMemref,
                  sourceMemref, srcLoopMap, padVal, zero, starts, readUBs,
                  padUBs, loopIndices, i + 1, buffRank,
                  /*no pad phase*/ false);
              loopIndices.pop_back_n(1);
            });
      }
      if (padUBs[i].isLiteralAndIdenticalTo(0)) {
        // No padding needed.
      } else {
        createAffine.forIE(readUBs[i], padUBs[i], 1,
            [&](AffineBuilderKrnlMem &createAffine, Value index) {
              loopIndices.emplace_back(index);
              genCopyLoops(createAffine, enclosingScope, buffMemref,
                  sourceMemref, srcLoopMap, padVal, zero, starts, readUBs,
                  padUBs, loopIndices, i + 1, buffRank,
                  /*pad phase*/ true);
              loopIndices.pop_back_n(1);
            });
      }
      // For next level up of padding, if any, will not copy data anymore
      readUBs[i] = zero;
    }
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: Krnl Copy from Buffer operation.
//===----------------------------------------------------------------------===//

class KrnlCopyFromBufferLowering
    : public OpRewritePattern<KrnlCopyFromBufferOp> {
public:
  using OpRewritePattern<KrnlCopyFromBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlCopyFromBufferOp op, PatternRewriter &rewriter) const override {
    KrnlCopyFromBufferOpAdaptor operandAdaptor =
        KrnlCopyFromBufferOpAdaptor(op);
    Value buffMemref(operandAdaptor.buffer());
    Value destMemref(operandAdaptor.dest());
    ValueRange startVals(operandAdaptor.starts());
    int64_t destRank =
        destMemref.getType().cast<MemRefType>().getShape().size();
    int64_t buffRank =
        buffMemref.getType().cast<MemRefType>().getShape().size();
    int64_t destOffset = destRank - buffRank;
    assert(destOffset >= 0 && "offset expected non negative");
    ArrayAttributeIndexCapture writeSizeCapture(op.tileSizeAttr());

    Location loc = op.getLoc();
    AffineBuilderKrnlMem createAffine(rewriter, loc);
    IndexExprScope indexScope(createAffine);

    SmallVector<IndexExpr, 4> starts, bufferWriteUBs;
    MemRefBoundsIndexCapture buffBounds(buffMemref);
    MemRefBoundsIndexCapture destBounds(destMemref);
    getIndexExprList<DimIndexExpr>(startVals, starts);
    SmallVector<Value, 4> loopIndices;
    LiteralIndexExpr zero(0);

    for (long buffIndex = 0; buffIndex < buffRank; ++buffIndex) {
      long destIndex = destOffset + buffIndex;
      // Compute how many values to read.
      IndexExpr destBound =
          destBounds.getSymbol(destIndex); // Source memref size.
      IndexExpr blockSize =
          buffBounds.getSymbol(buffIndex); // Buffer memref size.
      if (writeSizeCapture.size()) {
        int64_t memSize = blockSize.getLiteral();
        blockSize = writeSizeCapture.getLiteral(buffIndex); // Size from param.
        assert(blockSize.getLiteral() <= memSize &&
               "writeTileSize cannot be larger than the buffer size");
      }
      IndexExpr startGI =
          starts[destIndex]; // Global index in dest memref of tile.
      IndexExpr bufferWrite = trip(destBound, blockSize, startGI);
      bufferWrite.debugPrint("buffer wrote");
      bufferWriteUBs.emplace_back(bufferWrite);
    }
    genCopyLoops(createAffine, &indexScope, buffMemref, destMemref, zero,
        starts, bufferWriteUBs, loopIndices, 0, buffRank);
    rewriter.eraseOp(op);
    return success();
  }

  void genCopyLoops(AffineBuilderKrnlMem &createAffine,
      IndexExprScope *enclosingScope, Value buffMemref, Value destMemref,
      IndexExpr zero, SmallVectorImpl<IndexExpr> &starts,
      SmallVectorImpl<IndexExpr> &writeUBs, SmallVectorImpl<Value> &loopIndices,
      int64_t i, int64_t buffRank) const {
    if (i == buffRank) {
      // create new scope and import index expressions
      IndexExprScope currScope(createAffine, enclosingScope);
      KrnlBuilder createKrnl(createAffine);
      SmallVector<IndexExpr, 4> currLoopIndices, currStarts;
      getIndexExprList<DimIndexExpr>(loopIndices, currLoopIndices);
      getIndexExprList<SymbolIndexExpr>(starts, currStarts);
      int64_t destRank = starts.size();
      int64_t destOffset = destRank - buffRank;
      SmallVector<IndexExpr, 4> currStoreIndices;
      for (long destIndex = 0; destIndex < destRank; ++destIndex) {
        if (destIndex < destOffset) {
          // Dimensions that are unique to source memref, just use starts.
          currStoreIndices.emplace_back(currStarts[destIndex]);
        } else {
          // Dimensions that are shared by source memref & buffer, add loop
          // indices to starts.
          int64_t buffIndex = destIndex - destOffset;
          currStoreIndices.emplace_back(
              currLoopIndices[buffIndex] + currStarts[destIndex]);
        }
      }
      Value destVal = createKrnl.loadIE(buffMemref, currLoopIndices);
      createKrnl.storeIE(destVal, destMemref, currStoreIndices);
    } else {
      if (writeUBs[i].isLiteralAndIdenticalTo(0)) {
        // Nothing to write.
      } else {
        // Loop to copy the data.
        createAffine.forIE(zero, writeUBs[i], 1,
            [&](AffineBuilderKrnlMem &createAffine, Value index) {
              loopIndices.emplace_back(index);
              genCopyLoops(createAffine, enclosingScope, buffMemref, destMemref,
                  zero, starts, writeUBs, loopIndices, i + 1, buffRank);
              loopIndices.pop_back_n(1);
            });
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: Memset op.
//===----------------------------------------------------------------------===//

class KrnlMemsetLowering : public OpRewritePattern<KrnlMemsetOp> {
public:
  using OpRewritePattern<KrnlMemsetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlMemsetOp op, PatternRewriter &rewriter) const override {
    // Get info from operands.
    KrnlMemsetOpAdaptor operandAdaptor = KrnlMemsetOpAdaptor(op);
    Value destMemRef(operandAdaptor.dest());
    Value destVal(operandAdaptor.value());
    Location loc = op.getLoc();
    AffineBuilderKrnlMem createAffine(rewriter, loc);
    IndexExprScope indexScope(createAffine);
    MemRefBoundsIndexCapture destBounds(destMemRef);

    int rank = destBounds.getRank();
    SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> ubs;
    destBounds.getDimList(ubs);
    SmallVector<int64_t, 4> steps(rank, 1);
    // Copy data,
    createAffine.forIE(lbs, ubs, steps,
        [&](AffineBuilderKrnlMem &createAffine, ValueRange indices) {
          createAffine.store(destVal, destMemRef, indices);
        });
    rewriter.eraseOp(op);
    return success();
  }
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
void markLoopBodyAsMovable(
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

      auto movableOp = builder.create<KrnlMovableOp>(delimeterOp->getLoc());
      auto &movableRegion = movableOp.region();
      auto *entryBlock = new Block;
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

void ConvertKrnlToAffinePass::runOnOperation() {
  OpBuilder builder(&getContext());
  FuncOp funcOp = getOperation();

  // external function: nothing to do
  if (funcOp.body().empty()) {
    return;
  }

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

  ConversionTarget target(getContext());
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
  RewritePatternSet patterns(&getContext());
  patterns.insert<KrnlTerminatorLowering>(&getContext());
  patterns.insert<KrnlLoadLowering>(&getContext());
  patterns.insert<KrnlStoreLowering>(&getContext());
  patterns.insert<KrnlMatmulLowering>(&getContext());
  patterns.insert<KrnlCopyToBufferLowering>(&getContext());
  patterns.insert<KrnlCopyFromBufferLowering>(&getContext());
  patterns.insert<KrnlMemsetLowering>(&getContext());

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
    }
    free(currUnrollAndJamList);
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
  free(currUnrollAndJamList);
}

} // namespace

std::unique_ptr<Pass> mlir::createConvertKrnlToAffinePass() {
  return std::make_unique<ConvertKrnlToAffinePass>();
}
