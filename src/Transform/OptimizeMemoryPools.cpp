/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- OptimizeMemoryPools.cpp - Optimize Memory Pools -------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// For certain cases the number of individual memory allocations required for
// all internal tensors is large and needs to be mitigated. This pass optimizes
// the internal MemRef static and dynamic memory pools emitted by the
// BundleMemoryPool pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;
namespace {

// Handling of static memory pool on a block-basis in each function. For each
// block we need to keep track of the memory pools which have been compacted
// already. There can be several such memory pools, one for each alignment
// present in the program.
typedef std::map<Block *, llvm::SmallSet<int64_t, 16>>
    BlockToCompactedAlignments;

typedef std::map<Block *, llvm::SmallSet<KrnlGetRefOp, 16>>
    BlockToDiscardedGetRefs;

/// Get the total size in bytes used by the getref operations associated
/// with a given memory pool.
int64_t getAllocGetRefTotalSize(memref::AllocOp *allocOp) {
  auto parentBlock = allocOp->getOperation()->getBlock();

  int64_t totalSize = 0;
  int64_t alignment = getAllocAlignment(*allocOp);
  SmallVector<KrnlGetRefOp, 4> seenGetRefs;
  parentBlock->walk(
      [&totalSize, &seenGetRefs, &alignment, allocOp](KrnlGetRefOp op) {
        // If krnl.getref operation uses a different mempool then exit.
        if (op.mempool() != allocOp->getResult())
          return;

        // Check that the krnl.getref operation has not already been counted.
        // We must make sure we count the memory footprint of getref operations
        // sharing a slot only once.
        for (auto getRef : seenGetRefs)
          if (op.offset() == getRef.offset())
            return;

        // Footprint has not been counted yet. Add it to totalSize.
        int64_t memrefSize = getMemRefSizeInBytes(op.getResult());
        totalSize += memrefSize;
        if (alignment > 0) {
          int64_t misalignment = memrefSize % alignment;
          if (misalignment > 0)
            totalSize += alignment - misalignment;
        }

        // Act krnl.getref operation as seen.
        seenGetRefs.emplace_back(op);
      });

  return totalSize;
}

/// Returns a list of operations in the current block that use the getref.
std::vector<Operation *> getGetRefStores(KrnlGetRefOp *getRef) {
  auto parentBlock = getRef->getOperation()->getBlock();
  std::vector<Operation *> stores;

  parentBlock->walk([&stores, getRef](KrnlStoreOp op) {
    for (const auto &operand : op.getOperands())
      if (operand == getRef->getResult())
        stores.emplace_back(op);
  });

  // The list contains at least one use.
  return stores;
}

/// Returns a list of operations in the current block that *view* the getref.
std::vector<Operation *> getGetRefViews(KrnlGetRefOp *getRef) {
  auto parentBlock = getRef->getOperation()->getBlock();
  std::vector<Operation *> views;

  parentBlock->walk([&views, getRef](Operation *op) {
    if (dyn_cast<ViewLikeOpInterface>(op)) {
      if (op->getOperands()[0] == getRef->getResult())
        views.emplace_back(op);
    }
  });

  // The list contains at least one use.
  return views;
}

/// Returns a list of distinct krnl.getref operations in the current
/// block that use the memory pool.
SmallVector<KrnlGetRefOp, 4> getAllDistinctGetRefsForAlloc(
    memref::AllocOp *allocOp) {
  auto parentBlock = allocOp->getOperation()->getBlock();
  SmallVector<KrnlGetRefOp, 4> getRefs;

  parentBlock->walk([&getRefs, allocOp](KrnlGetRefOp op) {
    if (op.mempool() != allocOp->getResult())
      return;

    // If a getRef with the same memory pool and offset has
    // already been added, skip it.
    for (auto getRef : getRefs)
      if (op.offset() == getRef.offset())
        return;

    getRefs.emplace_back(op);
  });

  // The list contains at least one use.
  return getRefs;
}

/// Returns a list of krnl.getref operations in the current block
/// that share the same offset and memory pool.
SmallVector<KrnlGetRefOp, 4> getAllGetRefWithSameOffset(KrnlGetRefOp *getRef) {
  auto parentBlock = getRef->getOperation()->getBlock();
  SmallVector<KrnlGetRefOp, 4> sameOffsetGetRefs;

  parentBlock->walk([&sameOffsetGetRefs, getRef](KrnlGetRefOp op) {
    if (op.mempool() == getRef->mempool() && op.offset() == getRef->offset())
      sameOffsetGetRefs.emplace_back(op);
  });

  // The list contains at least one entry, the input krnl.getref.
  return sameOffsetGetRefs;
}

/// Returns a list of krnl.getref operations in the current block
/// that share the same offset and memory pool but are not part
/// of the exception list.
SmallVector<KrnlGetRefOp, 4> getAllGetRefWithSameOffsetExcept(
    KrnlGetRefOp *getRef, SmallVectorImpl<KrnlGetRefOp> &exceptionList) {
  auto parentBlock = getRef->getOperation()->getBlock();
  SmallVector<KrnlGetRefOp, 4> sameOffsetGetRefs;

  parentBlock->walk([&sameOffsetGetRefs, getRef, &exceptionList](
                        KrnlGetRefOp op) {
    for (auto exception : exceptionList)
      if (op == exception)
        return;

    if (op.mempool() == getRef->mempool() && op.offset() == getRef->offset())
      sameOffsetGetRefs.emplace_back(op);
  });

  // The list contains at least one entry, the input krnl.getref.
  return sameOffsetGetRefs;
}

bool getRefUsesAreDisjoint(
    SmallVectorImpl<KrnlGetRefOp> &firstGetRefList, KrnlGetRefOp secondGetRef) {
  // Return variable.
  bool refsUseIsDisjoint = true;

  // TODO: support memref view ops.
  if (!getGetRefViews(&secondGetRef).empty())
    return false;

  // Compute all the stores into the second getref.
  std::vector<Operation *> allStores = getGetRefStores(&secondGetRef);

  // For each store, analyze the list of dependendent operations that
  // contributes to the computation of the value being stored. The leaf
  // values are going to be represented by: load operations and constants.
  for (const auto &store : allStores) {
    // Initialize work queue data structure.
    std::vector<Value> operandList;
    operandList.emplace_back(store->getOperands()[0]);

    // Construct the list of Values on which the current AllocOp depends on.
    llvm::SetVector<Operation *> dependentOps;
    while (operandList.size() > 0 && refsUseIsDisjoint) {
      Value currentElement = operandList[0];
      Operation *definingOperation = currentElement.getDefiningOp();

      // If this value has not been seen before, process it.
      if (definingOperation && dependentOps.count(definingOperation) == 0) {
        // Add value to dependent values list.
        dependentOps.insert(definingOperation);

        if (llvm::dyn_cast<KrnlLoadOp>(definingOperation)) {
          // Check that the MemRef operand of this load operation is
          // not in the firstGetRefList.
          Value loadOperand = definingOperation->getOperands()[0];
          if (!isBlockArgument(definingOperation, loadOperand)) {
            Operation *loadOperandDefinition = loadOperand.getDefiningOp();
            KrnlGetRefOp loadGetRefOperand =
                llvm::dyn_cast_or_null<KrnlGetRefOp>(loadOperandDefinition);

            // If the load operand is valid, compare it with all the entries
            // in the firstGetRefList. If it matches any one of them then the
            // secondGetRef cannot share the same memory pool slot with the
            // rest of the getref operations in the firstGetRefList.
            if (loadGetRefOperand)
              for (auto firstGetRef : firstGetRefList)
                if (firstGetRef == loadGetRefOperand) {
                  refsUseIsDisjoint = false;
                  break;
                }
          }
        } else {
          // Add operands to work queue.
          for (const auto &operand : definingOperation->getOperands())
            if (!isBlockArgument(secondGetRef, operand))
              operandList.emplace_back(operand);
        }
      }

      // Erase first element from work queue.
      operandList.erase(operandList.begin());
    }

    // Exit if use is not disjoint.
    if (!refsUseIsDisjoint)
      break;
  }

  return refsUseIsDisjoint;
}

/// Check that the incoming secondGetRef is not in any operation with existing
/// slot reusers.
bool getRefUsesAreNotUsedBySameOp(
    SmallVectorImpl<KrnlGetRefOp> &firstGetRefList, KrnlGetRefOp secondGetRef) {
  bool usedBySameOperation = false;
  for (auto firstGetRef : firstGetRefList)
    if (usedBySameOp(&firstGetRef, &secondGetRef)) {
      usedBySameOperation = true;
      break;
    }

  return usedBySameOperation;
}

/// secondGetRef candidate is checked against every element of firstGetRefList
/// whether a load/store chain exists between them:
///
///    a = load firstGetRef[]
///    b = f(a)
///    store b secondGetRef[]
///
/// The check must be done both ways:
///
///    a = load secondGetRef[]
///    b = f(a)
///    store b firstGetRef[]
///
bool getRefUsesAreMutuallyDisjoint(
    SmallVectorImpl<KrnlGetRefOp> &firstGetRefList,
    SmallVectorImpl<KrnlGetRefOp> &secondGetRefList) {
  for (auto getRef : secondGetRefList) {
    if (!getRefUsesAreDisjoint(firstGetRefList, getRef)) {
      return false;
    }
  }

  for (auto getRef : firstGetRefList) {
    if (!getRefUsesAreDisjoint(secondGetRefList, getRef)) {
      return false;
    }
  }
  return true;
}

/// Returns the outermost krnl.iterate that contains this operation.
/// Example:
///
/// func() {
///   if {
///     krnl.iterate {  <--- Outermost loop.
///       krnl.iterate {
///         if {
///           ... op ...
///         }
///       }
///     }
///   }
/// }
///
Operation *getOutermostLoop(Operation *op) {
  Operation *outermostLoop = nullptr;

  // Get current block.
  Block *currentBlock = op->getBlock();

  // Current block must exist.
  assert(currentBlock && "Operation not in a block.");

  // Compute parent operation of the current block. Every block has
  // a parent operation.
  Operation *parentBlockOp = currentBlock->getParentOp();
  while (!llvm::dyn_cast_or_null<FuncOp>(parentBlockOp)) {
    if (llvm::dyn_cast_or_null<KrnlIterateOp>(parentBlockOp))
      outermostLoop = parentBlockOp;
    parentBlockOp = parentBlockOp->getBlock()->getParentOp();
  }

  return outermostLoop;
}

/// Returns true if two operations share the same outermost krnl.iterate.
bool checkOuterLoopsMatch(Operation *op1, Operation *op2) {
  // Check if the outer loops of the two operations match.
  // If one of the operations is not part of a loop (i.e. the returned
  // operation of the getOutermostLoop is nullptr) then return false.
  Operation *outerLoop1 = getOutermostLoop(op1);

  if (!outerLoop1)
    return false;

  Operation *outerLoop2 = getOutermostLoop(op2);

  if (!outerLoop2)
    return false;

  // If both outer loops are valid, check if they match.
  return outerLoop1 == outerLoop2;
}

/// Returns true if the extremities of the live ranges of two getrefs
/// share the same outermost krnl.iterate. The live range for one GetRef
/// is given by the firstOp and lastOp values. The live range of the other
/// GetRef is given by its full live range.
bool liveRangesInSameLoopNest(Operation *firstOp, Operation *lastOp,
    std::vector<Operation *> liveRangeOpList) {
  // If any of the firstOp or lastOp are in the top level block of the
  // function, then they cannot share a loop nest with the last or first
  // operation in the live range respectively.
  bool firstOpInTopLevelBlock = opInTopLevelBlock(firstOp);
  bool lastOpInTopLevelBlock = opInTopLevelBlock(firstOp);

  // If both firstOp and lastOp are in the top level block then they cannot
  // share a loop nest with the live range.
  if (firstOpInTopLevelBlock && lastOpInTopLevelBlock)
    return false;

  // Repeat checks for first/last operation in live range.
  Operation *liveRangeFirstOp = liveRangeOpList[0];
  assert(liveRangeOpList.size() > 0 &&
         "Live range empty but must have at least one element.");
  Operation *liveRangeLastOp = liveRangeOpList[liveRangeOpList.size() - 1];

  bool firstLROpInTopLevelBlock = opInTopLevelBlock(liveRangeFirstOp);
  bool lastLROpInTopLevelBlock = opInTopLevelBlock(liveRangeLastOp);

  // If both live range extremities are in the top level block then they cannot
  // share a loop nest with the other live range.
  if (firstLROpInTopLevelBlock && lastLROpInTopLevelBlock)
    return false;

  // If neither of the lastOp or liveRangeFirstOp are in the top block then
  // check if the outermost loops that contain them are the same. If they are
  // the same then they share the same loop nest, return true.
  if (!lastOpInTopLevelBlock && !firstLROpInTopLevelBlock &&
      checkOuterLoopsMatch(lastOp, liveRangeFirstOp))
    return true;

  // Now check the other pair of extremities. If they are in the same loop nest
  // return true.
  if (!firstOpInTopLevelBlock && !lastLROpInTopLevelBlock &&
      checkOuterLoopsMatch(firstOp, liveRangeLastOp))
    return true;

  // If none of the cases above were met then:
  // 1. at least one of the extremities in each pair is at top-block level.
  // or
  // 2. extremities are in sub-blocks but they do not share a loop nest.
  // In either case the intersection check must return false.
  return false;
}

/// Check that the live range of the secondGetRef does not intersect with
/// any of the live ranges of the GetRefs in firstGetRefList.
bool checkLiveRangesIntersect(
    SmallVectorImpl<KrnlGetRefOp> &firstGetRefList, KrnlGetRefOp secondGetRef) {
  // Get first and last ops for the live range of the secondGetRef.
  Operation *firstOp = getLiveRangeFirstOp(secondGetRef);
  Operation *lastOp = getLiveRangeLastOp(secondGetRef);

  // Check that the live range of each individual element in secondGetRefList
  // is independent from the individual live ranges of the elements
  // of the firstGetRefList.
  for (auto firstGetRef : firstGetRefList) {
    // Fetch the full live range for the first set of getref operations.
    std::vector<Operation *> liveRangeOpList = getLiveRange(firstGetRef);

    // Check if either the first or last ops in the second live range are part
    // of the first live range.
    bool firstOpInLiveRange = operationInLiveRange(firstOp, liveRangeOpList);
    bool lastOpInLiveRange = operationInLiveRange(lastOp, liveRangeOpList);
    if (firstOpInLiveRange || lastOpInLiveRange)
      return true;

    // Since firstOp and lastOp are not part of the live range, check whether
    // the live range is fully contained between firstOp and lastOp. If it is
    // return true.
    if (liveRangeIsContained(firstOp, lastOp, liveRangeOpList))
      return true;

    // Up to this point, the checks we have done allow for ranges to be
    // considered disjoint even when their extremities are part of the same
    // loop nest. This means we have to perform an additional check: if the
    // extremities of the two live ranges share the same loop nest determiend
    // by `krnl.iterate` ops. If they do then the live ranges intersect.
    if (liveRangesInSameLoopNest(firstOp, lastOp, liveRangeOpList))
      return true;
  }

  // If all getRef live ranges are independent then no intersection exists.
  return false;
}

//===----------------------------------------------------------------------===//
// Rewrite patterns.
//===----------------------------------------------------------------------===//

// This pattern transforms an existing static memory pool into a more compact
// version of itself. This means allowing more than one krnl.getref to use
// the same slot inside the memory pool. A slot is any contiguous chunk of
// of memory used by a given getref.
//
// Example:
//
// Unoptimized:
//  %1 = alloc() : memref<2000xi8>
//  %2 = "krnl.getref"(%1, 1600)
//  %3 = "krnl.getref"(%1, 1200)
//  %4 = "krnl.getref"(%1, 800)
//  %5 = "krnl.getref"(%1, 400)
//  %6 = "krnl.getref"(%1, 0)
//
// Optimized:
//  %1 = alloc() : memref<2000xi8>
//  %2 = "krnl.getref"(%1, 1600)
//  %3 = "krnl.getref"(%1, 400)
//  %4 = "krnl.getref"(%1, 1600)
//  %5 = "krnl.getref"(%1, 400)
//  %6 = "krnl.getref"(%1, 400)
//
// Note: this rule does not actually alter the total size of the memory
// pool, it just reuses slots where possible. The compaction of the memory
// pool is performed by the KrnlCompactStaticMemoryPools rule.
//
class KrnlOptimizeStaticMemoryPools : public OpRewritePattern<KrnlGetRefOp> {
public:
  using OpRewritePattern<KrnlGetRefOp>::OpRewritePattern;

  BlockToCompactedAlignments *blockToStaticPoolAlignments;
  BlockToDiscardedGetRefs *blockToDiscardedGetRefs;
  KrnlOptimizeStaticMemoryPools(MLIRContext *context,
      BlockToCompactedAlignments *_blockToStaticPoolAlignments,
      BlockToDiscardedGetRefs *_blockToDiscardedGetRefs)
      : OpRewritePattern<KrnlGetRefOp>(context) {
    blockToStaticPoolAlignments = _blockToStaticPoolAlignments;
    blockToDiscardedGetRefs = _blockToDiscardedGetRefs;
  }

  LogicalResult matchAndRewrite(
      KrnlGetRefOp firstGetRef, PatternRewriter &rewriter) const override {
    auto loc = firstGetRef.getLoc();
    auto memRefType = firstGetRef.getResult().getType().dyn_cast<MemRefType>();

    // Only handle krnl.getref ops that return a constant shaped MemRef.
    if (!hasAllConstantDimensions(memRefType))
      return failure();

    // Retrieve the AllocOp that this GetRef uses.
    auto staticMemPool = getAllocOfGetRef(&firstGetRef);

    // Ensure that the alloc obtained above is static memory pool.
    auto memPoolType =
        staticMemPool.getResult().getType().dyn_cast<MemRefType>();
    auto memPoolShape = memPoolType.getShape();

    // Static memory pool type must be byte.
    if (getMemRefEltSizeInBytes(memPoolType) != 1)
      return failure();

    // Rank of the static memory pool must be 1.
    if (memPoolShape.size() != 1)
      return failure();

    // Determine if the static memory pool is bundled i.e. participates in more
    // than one getRef.
    if (getAllocGetRefNum(&staticMemPool) < 2)
      return failure();

    // Get parent block.
    Block *parentBlock = firstGetRef.getOperation()->getBlock();

    // Get alignment of the static memory pool.
    int64_t alignment = getAllocAlignment(staticMemPool);

    // Check if this block has already been compacted for the alignment of the
    // current memory pool. If it has then skip its optimization.
    if (blockToStaticPoolAlignments->count(parentBlock) > 0 &&
        blockToStaticPoolAlignments->at(parentBlock).count(alignment) > 0)
      return failure();

    // TODO: relax this condition.
    // If this is not the top block fail.
    if (!llvm::dyn_cast_or_null<FuncOp>(parentBlock->getParentOp()))
      return failure();

    // List of all GetRefs which share the slot with firstGetRef.
    SmallVector<KrnlGetRefOp, 4> firstGetRefList =
        getAllGetRefWithSameOffset(&firstGetRef);

    // All matches are discovered in one application so applying the rule to
    // an already optimized set of getrefs will not find new reuses.
    if (firstGetRefList.size() > 1)
      return failure();

    // Get a GetRef, other than the current one, that uses the same static
    // memory pool.
    SmallVector<KrnlGetRefOp, 4> getRefCandidates;
    llvm::SmallSet<KrnlGetRefOp, 16> listOfDiscardedGetRefs;
    if (blockToDiscardedGetRefs->count(parentBlock) > 0)
      listOfDiscardedGetRefs = blockToDiscardedGetRefs->at(parentBlock);

    for (auto &op :
        llvm::make_range(parentBlock->begin(), std::prev(parentBlock->end()))) {
      KrnlGetRefOp candidate = llvm::dyn_cast_or_null<KrnlGetRefOp>(&op);

      // If not a valid KrnlGetRefOp, continue searching.
      if (!candidate)
        continue;

      // If candidate is already sharing a slot with firstGetRef, skip it.
      bool sharesSlot = false;
      for (auto getRef : firstGetRefList)
        if (getRef == candidate) {
          sharesSlot = true;
          break;
        }
      if (sharesSlot)
        continue;

      // Check candidate is not in the discarded list:
      bool isDiscardedGetRef = false;
      for (auto discardedGetRef : listOfDiscardedGetRefs)
        if (discardedGetRef == candidate) {
          isDiscardedGetRef = true;
          break;
        }
      if (isDiscardedGetRef)
        continue;

      // The second krnl.getref properties:
      // - must use the same static memory pool as the first
      // krnl.getref;
      // - the result must have the same memory footprint as the first.
      memref::AllocOp allocOfCandidate = getAllocOfGetRef(&candidate);
      if (allocOfCandidate == staticMemPool &&
          getMemRefSizeInBytes(firstGetRef.getResult()) ==
              getMemRefSizeInBytes(candidate.getResult())) {
        getRefCandidates.emplace_back(candidate);
      }
    }

    // If no candidate was found, pattern matching failed.
    if (getRefCandidates.size() < 1)
      return failure();

    // TODO: conditional printing of progress:
    // printf(" Candidates: %d --- Visited getrefs = %d\n",
    //     getRefCandidates.size(), listOfDiscardedGetRefs.size());

    // TODO: conditional printing of progress:
    // printf(" Start processing candidates ...\n");
    SmallVector<KrnlGetRefOp, 4> validSlotReusers;
    for (auto secondGetRef : getRefCandidates) {
      // Check that the current candidate has not already been added as a valid
      // slot reuser.
      bool isSlotReuser = false;
      for (auto slotReuser : validSlotReusers) {
        if (slotReuser == secondGetRef) {
          isSlotReuser = true;
          break;
        }
      }
      if (isSlotReuser) {
        continue;
      }

      // If the second getref has the same offset as the first then the rewrite
      // rule has already been applied to this getref so there is no work to do.
      if (firstGetRef.offset() == secondGetRef.offset()) {
        continue;
      }

      // Both first and second getRef ops may have already been processed by
      // this rewrite rule. There could be several krnl.getref with the same
      // offset as firstGetRef and several krnl.getRef with the same offset as
      // secondGetRef. In general we have to be able to handle this case.
      SmallVector<KrnlGetRefOp, 4> secondGetRefList =
          getAllGetRefWithSameOffsetExcept(&secondGetRef, validSlotReusers);

      // Do not merge the secondGetRef if secondGetRef has any reusers. It
      // means that the analysis has already been performed on secondGetRef
      // and all the possible reuses have already been found for secondGetRef.
      if (secondGetRefList.size() > 1) {
        continue;
      }

      // If the two getRefs are used by the same operation which we know
      // nothing about, then we assume the worst case semantics i.e. that
      // the operation acts as a function which can modify the content of
      // one of the getRefs based on the other. This implies that the two
      // getRefs cannot share the same memory pool slot.
      if (getRefUsesAreNotUsedBySameOp(firstGetRefList, secondGetRef)) {
        continue;
      }

      // Check that the usage of the candidate getrefs is disjoint from the
      // usage of any of the first getrefs. This means that for any store to a
      // getref in secondGetRefList, the value stored does not involve a load
      // from a getref in firstGetRefList (and vice-versa).
      if (!getRefUsesAreMutuallyDisjoint(firstGetRefList, secondGetRefList)) {
        continue;
      }

      // Check live ranges do not intersect.
      // Live range, chain of instructions between the first and last
      // load/store from/to any krnl.getref in a given list.
      if (checkLiveRangesIntersect(firstGetRefList, secondGetRef)) {
        continue;
      }

      // If this is a valid slot reuser then this is the only slot in which
      // it can fit so it cannot participate in any other slot.
      if (blockToDiscardedGetRefs->count(parentBlock) == 0)
        blockToDiscardedGetRefs->insert(
            std::pair<Block *, llvm::SmallSet<KrnlGetRefOp, 16>>(
                parentBlock, llvm::SmallSet<KrnlGetRefOp, 16>()));
      blockToDiscardedGetRefs->at(parentBlock).insert(secondGetRef);

      // Add candidate to list of valid reusers.
      validSlotReusers.emplace_back(secondGetRef);

      // Add the currently discovered krnl.getref valid reuser to the list of
      // firstGetRef reusers. This ensures that the rest of the candidates
      // take into consideration this reuser when analyzing if a new reuse is
      // valid.
      firstGetRefList.emplace_back(secondGetRef);
    }

    // TODO: conditional printing of progress:
    // printf(" Done processing candidates, slot reusers = %d\n",
    //     validSlotReusers.size());

    // Never consider the matched getref as a candidate ever again.
    if (blockToDiscardedGetRefs->count(parentBlock) == 0)
      blockToDiscardedGetRefs->insert(
          std::pair<Block *, llvm::SmallSet<KrnlGetRefOp, 16>>(
              parentBlock, llvm::SmallSet<KrnlGetRefOp, 16>()));
    blockToDiscardedGetRefs->at(parentBlock).insert(firstGetRef);

    // No valid slot reuse getRefs have been identified.
    if (validSlotReusers.size() == 0)
      return failure();

    // A suitable slot can be reused. Convert all secondGetRefList entries
    // to use the same slot in the memory pool as all the firstGetRefList
    // entries.
    for (auto secondGetRef : validSlotReusers) {
      auto newGetRefOp =
          rewriter.create<KrnlGetRefOp>(loc, secondGetRef.getResult().getType(),
              staticMemPool, firstGetRef.offset());
      newGetRefOp.getOperation()->moveBefore(secondGetRef);
      rewriter.replaceOp(secondGetRef, newGetRefOp.getResult());
    }

    return success();
  }
};

// This pattern will compact the memory pool after the optimization pattern is
// run. This means that after this rule is applied there are no slots in the
// mempool that are not used at least once.
//
// Example:
//
// Optimized memory pool:
//  %1 = alloc() : memref<2000xi8>
//  %2 = "krnl.getref"(%1, 1600)
//  %3 = "krnl.getref"(%1, 400)
//  %4 = "krnl.getref"(%1, 1600)
//  %5 = "krnl.getref"(%1, 400)
//  %6 = "krnl.getref"(%1, 400)
//
// Compacted optimized memory pool:
//  %1 = alloc() : memref<800xi8>
//  %2 = "krnl.getref"(%1, 400)
//  %3 = "krnl.getref"(%1, 0)
//  %4 = "krnl.getref"(%1, 400)
//  %5 = "krnl.getref"(%1, 0)
//  %6 = "krnl.getref"(%1, 0)
//
class KrnlCompactStaticMemoryPools : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  BlockToCompactedAlignments *blockToStaticPoolAlignments;
  KrnlCompactStaticMemoryPools(MLIRContext *context,
      BlockToCompactedAlignments *_blockToStaticPoolAlignments)
      : OpRewritePattern<memref::AllocOp>(context) {
    blockToStaticPoolAlignments = _blockToStaticPoolAlignments;
  }

  LogicalResult matchAndRewrite(
      memref::AllocOp allocOp, PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    auto memPoolType = allocOp.getResult().getType().dyn_cast<MemRefType>();
    auto memPoolShape = memPoolType.getShape();

    // This is a memory pool if it is used by at least one getref.
    if (getAllocGetRefNum(&allocOp) < 1)
      return failure();

    // Only handle alloc ops that return a constant shaped MemRef.
    if (!hasAllConstantDimensions(memPoolType))
      return failure();

    // Static memory pool type must be byte.
    if (getMemRefEltSizeInBytes(memPoolType) != 1)
      return failure();

    // Rank of the static memory pool must be 1.
    if (memPoolShape.size() != 1)
      return failure();

    // Get parent block.
    Block *parentBlock = allocOp.getOperation()->getBlock();

    // Get alignment.
    int64_t alignment = getAllocAlignment(allocOp);

    // Check if this block has already been compacted for current alignment.
    // If it has then skip its processing.
    if (blockToStaticPoolAlignments->count(parentBlock) > 0 &&
        blockToStaticPoolAlignments->at(parentBlock).count(alignment) > 0)
      return failure();

    // If this is not the top block, fail.
    if (!llvm::dyn_cast_or_null<FuncOp>(parentBlock->getParentOp()))
      return failure();

    // Compute size of all krnl.getref operations that use this memory pool.
    int64_t usedMemory = getAllocGetRefTotalSize(&allocOp);

    // TODO: enable back once changes to bundle stage are also included.
    // assert(usedMemory <= memPoolShape[0] &&
    //       "Used memory exceeds allocated memory.");

    // Check if changes to the memory pool are required.
    if (memPoolShape[0] == usedMemory)
      return failure();

    // Compute the shape of the new static memory pool.
    SmallVector<int64_t, 1> newStaticMemPoolShape;
    newStaticMemPoolShape.emplace_back(usedMemory);
    auto newStaticMemPoolType =
        MemRefType::get(newStaticMemPoolShape, rewriter.getIntegerType(8));

    // We need to emit a new alloc of smaller size.
    memref::AllocOp newStaticMemPool = rewriter.create<memref::AllocOp>(
        loc, newStaticMemPoolType, allocOp.alignmentAttr());
    newStaticMemPool.getOperation()->moveBefore(allocOp);

    // Changes are required, memory pool needs to be compacted.
    SmallVector<KrnlGetRefOp, 4> distinctGetRefs =
        getAllDistinctGetRefsForAlloc(&allocOp);

    // Size of all distinct getrefs:
    int64_t distinctGRSize = 0;
    for (auto getRefOp : distinctGetRefs) {
      int64_t memrefSize = getMemRefSizeInBytes(getRefOp.getResult());
      distinctGRSize += memrefSize;
      if (alignment > 0) {
        int64_t misalignment = memrefSize % alignment;
        if (misalignment > 0)
          distinctGRSize += alignment - misalignment;
      }
    }
    assert(distinctGRSize == usedMemory &&
           "Size of all distinct getrefs must match the total used memory");

    // Each krnl.getref using the alloc needs to be re-emitted with the new
    // static memory pool and the new offset.
    int64_t currentOffset = 0;
    std::vector<std::pair<KrnlGetRefOp, KrnlGetRefOp>> oldToNewGetRef;
    for (auto getRefOp : distinctGetRefs) {
      // Emit the current offset inside the static memory pool.
      auto newOffset = rewriter.create<arith::ConstantOp>(loc,
          rewriter.getIntegerAttr(rewriter.getIntegerType(64), currentOffset));

      // Size of current getref.
      int64_t currentGetRefSize = getMemRefSizeInBytes(getRefOp.getResult());
      if (alignment > 0) {
        int64_t misalignment = currentGetRefSize % alignment;
        if (misalignment > 0)
          currentGetRefSize += alignment - misalignment;
      }

      // Get all getRefs which share the same memory slot.
      SmallVector<KrnlGetRefOp, 4> sameSlotGetRefs =
          getAllGetRefWithSameOffset(&getRefOp);

      // Replace each one with a getref using the new offset in the compacted
      // memory pool.
      for (auto oldGetRef : sameSlotGetRefs) {
        // Create a new krnl.getref using the new memory pool and new offset.
        auto newGetRefOp = rewriter.create<KrnlGetRefOp>(
            loc, oldGetRef.getResult().getType(), newStaticMemPool, newOffset);
        newGetRefOp.getOperation()->moveBefore(oldGetRef);

        oldToNewGetRef.emplace_back(
            std::pair<KrnlGetRefOp, KrnlGetRefOp>(oldGetRef, newGetRefOp));
      }

      // Update offset.
      currentOffset += currentGetRefSize;
    }

    assert(currentOffset == usedMemory &&
           "Size total used memory must match the last offset.");

    for (auto getRefPair : oldToNewGetRef)
      rewriter.replaceOp(getRefPair.first, getRefPair.second.getResult());

    rewriter.replaceOp(allocOp, newStaticMemPool.getResult());

    // Update compacted flag.
    if (blockToStaticPoolAlignments->count(parentBlock) == 0)
      blockToStaticPoolAlignments->insert(
          std::pair<Block *, llvm::SmallSet<int64_t, 16>>(
              parentBlock, llvm::SmallSet<int64_t, 16>()));
    blockToStaticPoolAlignments->at(parentBlock).insert(alignment);

    return success();
  }
};

/*!
 *  Function pass that optimizes memory pools.
 */
class KrnlOptimizeMemoryPoolsPass
    : public PassWrapper<KrnlOptimizeMemoryPoolsPass, OperationPass<FuncOp>> {
  BlockToCompactedAlignments blockToStaticPoolAlignments;
  BlockToDiscardedGetRefs blockToDiscardedGetRefs;

public:
  StringRef getArgument() const override { return "optimize-memory-pools"; }

  StringRef getDescription() const override {
    return "Optimize the static and dynamic memory pools.";
  }

  void runOnOperation() override {
    auto function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<KrnlOptimizeStaticMemoryPools>(
        &getContext(), &blockToStaticPoolAlignments, &blockToDiscardedGetRefs);
    patterns.insert<KrnlCompactStaticMemoryPools>(
        &getContext(), &blockToStaticPoolAlignments);

    // No need to test, its ok to fail the apply.
    LogicalResult res =
        applyPatternsAndFoldGreedily(function, std::move(patterns));
    assert((succeeded(res) || failed(res)) && "remove unused var warning");
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createKrnlOptimizeMemoryPoolsPass() {
  return std::make_unique<KrnlOptimizeMemoryPoolsPass>();
}
