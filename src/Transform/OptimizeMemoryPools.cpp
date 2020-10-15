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

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
namespace {

/// Get the AllocOp of the current GetRef.
AllocOp getAllocOfGetRef(KrnlGetRefOp *getRef) {
  auto parentBlock = getRef->getOperation()->getBlock();

  AllocOp alloc = nullptr;
  parentBlock->walk([&alloc, getRef](AllocOp op) {
    auto getRefAlloc = getRef->getOperands()[0];
    if (op.getResult() == getRefAlloc)
      alloc = op;
  });

  return alloc;
}

/// Get the number of GetRef ops associated with this AllocOp.
int64_t getAllocGetRefNum(AllocOp *allocOp) {
  auto parentBlock = allocOp->getOperation()->getBlock();

  int64_t numGetRefs = 0;
  parentBlock->walk([&numGetRefs, allocOp](KrnlGetRefOp op) {
    auto result = allocOp->getResult();
    if (op.getOperands()[0] == result)
      numGetRefs++;
  });

  return numGetRefs;
}

/// Get the total size in bytes used by the getref operations associated
/// with a given memory pool.
int64_t getAllocGetRefTotalSize(AllocOp *allocOp) {
  auto parentBlock = allocOp->getOperation()->getBlock();

  int64_t totalSize = 0;
  SmallVector<KrnlGetRefOp, 4> seenGetRefs;
  parentBlock->walk([&totalSize, &seenGetRefs, allocOp](KrnlGetRefOp op) {
    // Check that the krnl.getref operation has not already been counted.
    // We must make sure we count the memory footprint of getref operations
    // sharing a slot only once.
    for (auto getRef : seenGetRefs)
      if (op.offset() == getRef.offset())
        return;

    // Footprint has not been counter yet. Add it to totalSize.
    auto result = allocOp->getResult();
    if (op.getOperands()[0] == result)
      totalSize += getMemRefSizeInBytes(op.getResult());

    // Act krnl.getref operation as seen.
    seenGetRefs.emplace_back(op);
  });

  return totalSize;
}

// Check if this value is an argument of one of the blocks nested
// around it.
bool isBlockArgument(KrnlGetRefOp firstGetRef, Value operand) {
  // Parent operation of the current block.
  Operation *parentBlockOp;
  Block *currentBlock = firstGetRef.getOperation()->getBlock();

  do {
    // Check the arguments of the current block.
    for (auto arg : currentBlock->getArguments())
      if (operand == arg)
        return true;

    parentBlockOp = currentBlock->getParentOp();
    currentBlock = parentBlockOp->getBlock();

  } while (!llvm::dyn_cast_or_null<FuncOp>(parentBlockOp));

  return false;
}

/// Returns a list of operations in the current block that use the getref.
std::vector<Operation *> getGetRefStores(KrnlGetRefOp *getRef) {
  auto parentBlock = getRef->getOperation()->getBlock();
  std::vector<Operation *> stores;

  parentBlock->walk([&stores, getRef](StoreOp op) {
    for (const auto &operand : op.getOperands())
      if (operand == getRef->getResult())
        stores.emplace_back(op);
  });

  parentBlock->walk([&stores, getRef](AffineStoreOp op) {
    for (const auto &operand : op.getOperands())
      if (operand == getRef->getResult())
        stores.emplace_back(op);
  });

  // The list contains at least one use.
  return stores;
}

/// Returns a list of distinct krnl.getref operations in the current
/// block that use the memory pool.
SmallVector<KrnlGetRefOp, 4> getAllDistinctGetRefsForAlloc(AllocOp *allocOp) {
  auto parentBlock = allocOp->getOperation()->getBlock();
  SmallVector<KrnlGetRefOp, 4> getRefs;

  parentBlock->walk([&getRefs, allocOp](KrnlGetRefOp op) {
    // If a getRef with the same memory pool and offset has
    // already been added, skip it.
    for (auto getRef : getRefs)
      if (op.mempool() == getRef.mempool() && op.offset() == op.offset())
        return;

    if (op.getOperands()[0] == allocOp->getResult())
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

bool getRefUsesAreDisjoint(
    SmallVector<KrnlGetRefOp, 4> firstGetRefList, KrnlGetRefOp secondGetRef) {
  // Return variable.
  bool refsUseIsDisjoint = true;

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
      if (dependentOps.count(definingOperation) == 0) {
        // Add value to dependent values list.
        dependentOps.insert(definingOperation);

        if (definingOperation &&
            (llvm::dyn_cast<AffineLoadOp>(definingOperation) ||
             llvm::dyn_cast<LoadOp>(definingOperation))) {
          // Check that the MemRef operand of this load operation is
          // not in the firstGetRefList.
          Value loadOperand = definingOperation->getOperands()[0];
          if (!isBlockArgument(secondGetRef, loadOperand)) {
            Operation *loadOperandDefinition = loadOperand.getDefiningOp();
            KrnlGetRefOp loadGetRefOperand =
                llvm::dyn_cast<KrnlGetRefOp>(loadOperandDefinition);

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

bool getRefUsesAreMutuallyDisjoint(SmallVector<KrnlGetRefOp, 4> firstGetRefList,
    SmallVector<KrnlGetRefOp, 4> secondGetRefList) {
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

//===----------------------------------------------------------------------===//
// Rewrite patterns.
//===----------------------------------------------------------------------===//

class KrnlOptimizeStaticMemoryPools : public OpRewritePattern<KrnlGetRefOp> {
public:
  using OpRewritePattern<KrnlGetRefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlGetRefOp firstGetRef, PatternRewriter &rewriter) const override {
    auto loc = firstGetRef.getLoc();
    auto memRefType = convertToMemRefType(firstGetRef.getResult().getType());
    auto memRefShape = memRefType.getShape();

    // Only handle krnl.getref ops that return a constant shaped MemRef.
    if (!hasAllConstantDimensions(memRefType))
      return failure();

    // Retrieve the AllocOp that this GetRef uses.
    auto staticMemPool = getAllocOfGetRef(&firstGetRef);

    // Ensure that the alloc obtained above is static memory pool.
    auto memPoolType = convertToMemRefType(staticMemPool.getResult().getType());
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

    // Get a GetRef, other than the current one, that uses the same static
    // memory pool.
    SmallVector<KrnlGetRefOp, 4> getRefCandidates;
    for (auto &op :
        llvm::make_range(parentBlock->begin(), std::prev(parentBlock->end()))) {
      KrnlGetRefOp candidate = llvm::dyn_cast_or_null<KrnlGetRefOp>(&op);

      // The second krnl.getref properties:
      // - must be valid;
      // - cannot be the same krnl.getref as the first;
      // - must use the same static memory pool as the first krnl.getref;
      // - the result must have the same memory footprint as the first.
      if (candidate && candidate != firstGetRef &&
          getAllocOfGetRef(&candidate) == staticMemPool &&
          getMemRefSizeInBytes(firstGetRef.getResult()) ==
              getMemRefSizeInBytes(candidate.getResult())) {
        getRefCandidates.emplace_back(candidate);
      }
    }

    // If no candidate was found, pattern matching failed.
    if (getRefCandidates.size() < 1)
      return failure();

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
      if (isSlotReuser)
        continue;

      // If the second getref has the same offset as the first then the rewrite
      // rule has already been applied to this getref so there is no work to do.
      if (firstGetRef.offset() == secondGetRef.offset())
        continue;

      // Both first and second getRef ops may have already been processed by
      // this rewrite rule. There could be several krnl.getref with the same
      // offset as firstGetRef and several krnl.getRef with the same offset as
      // secondGetRef. In general we have to be able to handle this case.
      SmallVector<KrnlGetRefOp, 4> firstGetRefList =
          getAllGetRefWithSameOffset(&firstGetRef);
      SmallVector<KrnlGetRefOp, 4> secondGetRefList =
          getAllGetRefWithSameOffset(&secondGetRef);

      // Add all the currently discovered krnl.getref reusers that have not yet
      // been actually processed but are now known to be valid reusers of the
      // same slot. This is done for the purpose of checking validity of the
      // other remaining candidates which have to consider that there is now
      // an additional getref that uses the same slot.
      for (auto validUnemittedReuser : validSlotReusers)
        firstGetRefList.emplace_back(validUnemittedReuser);

      // Check that the usage of the candidate getrefs is disjoint from the
      // usage of any of the first getrefs. This means that for any store to a
      // getref in secondGetRefList, the value stored does not involve a load
      // from a getref in firstGetRefList (and vice-versa).
      bool refsUseIsDisjoint =
          getRefUsesAreMutuallyDisjoint(firstGetRefList, secondGetRefList);

      if (!refsUseIsDisjoint)
        continue;

      printf("Found a match:\n");
      firstGetRef.dump();
      secondGetRef.dump();

      for (auto secondGetRef : secondGetRefList)
        validSlotReusers.emplace_back(secondGetRef);
    }

    // No valid slot reuse getRefs have been identified.
    if (validSlotReusers.size() == 0)
      return failure();

    // A suitable slot can be reused. Convert all secondGetRefList entries to
    // use the same slot in the memory pool as all the firstGetRefList entries.
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

class KrnlCompactStaticMemoryPools : public OpRewritePattern<AllocOp> {
public:
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AllocOp allocOp, PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    auto memPoolType = convertToMemRefType(allocOp.getResult().getType());
    auto memPoolShape = memPoolType.getShape();

    // Only handle alloc ops that return a constant shaped MemRef.
    if (!hasAllConstantDimensions(memPoolType))
      return failure();

    // Static memory pool type must be byte.
    if (getMemRefEltSizeInBytes(memPoolType) != 1)
      return failure();

    // Rank of the static memory pool must be 1.
    if (memPoolShape.size() != 1)
      return failure();

    // This is a memory pool if it is used by at least one getref.
    if (getAllocGetRefNum(&allocOp) < 1)
      return failure();

    // Compute size of all krnl.getref operations that use this memory pool.
    int64_t usedMemory = getAllocGetRefTotalSize(&allocOp);

    assert(usedMemory <= memPoolShape[0] &&
           "Used memory exceeds allocated memory.");

    // Check if changes to the memory pool are required.
    if (memPoolShape[0] == usedMemory)
      return failure();

    // Compute the shape of the new static memory pool.
    SmallVector<int64_t, 1> newStaticMemPoolShape;
    newStaticMemPoolShape.emplace_back(usedMemory);
    auto newStaticMemPoolType =
        MemRefType::get(newStaticMemPoolShape, rewriter.getIntegerType(8));

    // We need to emit a new alloc of smaller size.
    AllocOp newStaticMemPool =
        rewriter.create<AllocOp>(loc, newStaticMemPoolType);
    newStaticMemPool.getOperation()->moveBefore(allocOp);

    // Changes are required, memory pool needs to be compacted.
    SmallVector<KrnlGetRefOp, 4> distinctGetRefs =
        getAllDistinctGetRefsForAlloc(&allocOp);

    // Each krnl.getref using the alloc needs to be re-emitted with the new
    // static memory pool and the new offset.
    int64_t currentOffset = 0;
    std::map<KrnlGetRefOp, KrnlGetRefOp> oldToNewGetRef;
    for (auto getRefOp : distinctGetRefs) {
      // Emit the current offset inside the static memory pool.
      auto newOffset = rewriter.create<ConstantOp>(loc,
          rewriter.getIntegerAttr(rewriter.getIntegerType(64), currentOffset));

      // Size of current getref.
      int64_t currentGetRefSize = getMemRefSizeInBytes(getRefOp.getResult());

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

        oldToNewGetRef.insert(
            std::pair<KrnlGetRefOp, KrnlGetRefOp>(oldGetRef, newGetRefOp));
      }

      // Update offset.
      currentOffset += currentGetRefSize;
    }

    for (auto getRefPair : oldToNewGetRef)
      rewriter.replaceOp(getRefPair.first, getRefPair.second.getResult());

    rewriter.replaceOp(allocOp, newStaticMemPool.getResult());

    return success();
  }
};

/*!
 *  Function pass that optimizes memory pools.
 */
class KrnlOptimizeMemoryPoolsPass
    : public PassWrapper<KrnlOptimizeMemoryPoolsPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<KrnlOptimizeStaticMemoryPools>(&getContext());
    patterns.insert<KrnlCompactStaticMemoryPools>(&getContext());

    applyPatternsAndFoldGreedily(function, patterns);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createKrnlOptimizeMemoryPoolsPass() {
  return std::make_unique<KrnlOptimizeMemoryPoolsPass>();
}
