/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- BundleMemoryPools.cpp - Bundle Memory Pools for  internal MemRefs -===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// For certain cases the number of individual memory allocations required for
// all internal tensors is large and needs to be mitigated. This pass bundles
// all the internal MemRef memory pools emitted by the EnableMemoryPool pass
// int a single memory pool.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Data structures for managing memory pools.
//===----------------------------------------------------------------------===//

// Data structure for managing memory pools.
// For each block track the set of memory pools for a given alignment.
// Memory pools are created for an alloc as long as the MemRef created by the
// alloc:
// - does not contain any affine maps;
// - the type of the MemRef is not index.
typedef std::map<int64_t, memref::AllocOp> AlignmentToMemPool;
typedef std::map<Block *, AlignmentToMemPool *> BlockToMemPool;

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

/// Retrieve function which contains the current operation.
ATTRIBUTE(unused) FuncOp getContainingFunction(memref::AllocOp op) {
  Operation *parentFuncOp = op->getParentOp();

  // While parent is not a FuncOp and its cast to a FuncOp is null.
  while (!llvm::dyn_cast_or_null<FuncOp>(parentFuncOp))
    parentFuncOp = parentFuncOp->getParentOp();

  return cast<FuncOp>(parentFuncOp);
}

// Check if this value is an argument of one of the blocks nested
// around it.
bool isBlockArgument(memref::AllocOp allocOp, Value operand) {
  // Parent operation of the current block.
  Operation *parentBlockOp;
  Block *currentBlock = allocOp.getOperation()->getBlock();

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

ATTRIBUTE(unused) KrnlGetRefOp getUnbundledGetRef(memref::AllocOp *memPool) {
  auto parentBlock = memPool->getOperation()->getBlock();

  KrnlGetRefOp unbundledGetRef = nullptr;
  parentBlock->walk([&unbundledGetRef, memPool](KrnlGetRefOp op) {
    auto result = memPool->getResult();
    if (op.getOperands()[0] != result)
      unbundledGetRef = op;
  });

  return unbundledGetRef;
}

KrnlGetRefOp getCurrentAllocGetRef(memref::AllocOp *allocOp) {
  auto parentBlock = allocOp->getOperation()->getBlock();

  KrnlGetRefOp currentAllocGetRef = nullptr;
  parentBlock->walk([&currentAllocGetRef, allocOp](KrnlGetRefOp op) {
    auto result = allocOp->getResult();
    if (op.getOperands()[0] == result)
      currentAllocGetRef = op;
  });

  return currentAllocGetRef;
}

//===----------------------------------------------------------------------===//
// Rewrite patterns.
//===----------------------------------------------------------------------===//

/*!
 *  RewritePattern that replaces:
 *    %mempool = alloc() : memref<<dims1>x<type>>
 *    %mem2 = alloc() : memref<<dims2>x<type>>
 *    %1 = krnl.getref %mem2 0 : memref<<dims2>x<type>>
 *  =>
 *    %mempool = alloc() : memref<<dims1 + dims2>x<type>>
 *    %1 = krnl.getref %mem1 <dims1> : memref<<dims2>x<type>>
 *
 *
 *  ASSUMPTION: All krnl.getref operations in the program have been emitted
 *              by the EnableMemoryPool pass i.e. there are no krnl.getref
 *              operations which are not related to the memory pool.
 *              krnl.getref is an operation specific to memory management
 *              for other use cases use MLIR Standard dialect operations.
 *              This assumption simplifies the code and avoids additional
 *              checks to ensure that all the participating krnl.getref
 *              operations are part of memory pooling.
 */

class KrnlBundleStaticMemoryPools : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  BlockToMemPool *blockToStaticPool;
  KrnlBundleStaticMemoryPools(
      MLIRContext *context, BlockToMemPool *_blockToStaticPool)
      : OpRewritePattern<memref::AllocOp>(context) {
    blockToStaticPool = _blockToStaticPool;
  }

  LogicalResult matchAndRewrite(
      memref::AllocOp allocOp, PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    auto memRefType = allocOp.getResult().getType().dyn_cast<MemRefType>();
    auto memRefShape = memRefType.getShape();

    // If alloca result is not used by getref then it cannot be part of
    // the memory pool.
    if (!checkOpResultIsUsedByGetRef(&allocOp))
      return failure();

    // Only handle constant AllocOps.
    if (!hasAllConstantDimensions(memRefType))
      return failure();

    // Alloc memory type must be byte.
    if (getMemRefEltSizeInBytes(memRefType) != 1)
      return failure();

    // Rank of the allocated MemRef must be 1.
    if (memRefShape.size() != 1)
      return failure();

    // Get parent block.
    Block *parentBlock = allocOp.getOperation()->getBlock();

    // Check if parent block has been seen before.
    if (blockToStaticPool->count(parentBlock) == 0) {
      allocOp.getOperation()->moveBefore(&parentBlock->front());
      // Create new entry in the block map.
      AlignmentToMemPool *alignmentToMemPool = new AlignmentToMemPool();
      blockToStaticPool->insert(std::pair<Block *, AlignmentToMemPool *>(
          parentBlock, alignmentToMemPool));
    }

    // Populate alignment integer.
    int64_t alignment = getAllocAlignment(allocOp);

    // If this parent block has been found present in the map, check that
    // a static memory bundle with the current alignment already exists.
    AlignmentToMemPool *alignmentToMemPool = blockToStaticPool->at(parentBlock);
    if (alignmentToMemPool->count(alignment) == 0) {
      // If static memory bundle for this alignment does not exist, then
      // create an entry.
      alignmentToMemPool->insert(
          std::pair<int64_t, memref::AllocOp>(alignment, allocOp));

      // This is the initial memory pool for this block and alignment
      // so trivially bundle it and return success.
      return success();
    }

    // Static memory pool for this alignment exists, fetch it.
    memref::AllocOp staticMemPoolAlloc = alignmentToMemPool->at(alignment);

    // If this is the alloc representing the memory pool and the function
    // already has an init block, pattern matching must fail to avoid
    // processing the static memory pool a second time.
    if (allocOp == staticMemPoolAlloc)
      return failure();

    auto staticMemPoolShape = staticMemPoolAlloc.getResult()
                                  .getType()
                                  .dyn_cast<MemRefType>()
                                  .getShape();
    int64_t currentMemPoolSize = staticMemPoolShape[0];
    if (alignment > 0) {
      int64_t misalignment = currentMemPoolSize % alignment;
      if (misalignment > 0)
        currentMemPoolSize += alignment - misalignment;
    }

    // Get the getref of the current allocOp. There is exactly one such getref.
    KrnlGetRefOp currentAllocGetRef = getCurrentAllocGetRef(&allocOp);
    if (!currentAllocGetRef)
      return failure();

    // Current memory pool size is the offset for the newly bundled
    // internal MemRef. Emit the offset as a constant.
    auto offset = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(
                 rewriter.getIntegerType(64), currentMemPoolSize));

    // Size in bytes of the output of the krnl.getref operation.
    int64_t unbundledTotalSize = memRefShape[0];

    // Compute new size.
    int64_t bundleTotalSize = unbundledTotalSize + currentMemPoolSize;

    // We need to emit a new alloc which contains the additional MemRef.
    SmallVector<int64_t, 1> newMemPoolShape;
    newMemPoolShape.emplace_back(bundleTotalSize);
    auto bundledMemPoolMemRefType =
        MemRefType::get(newMemPoolShape, rewriter.getIntegerType(8));

    auto newStaticMemPoolAlloc = rewriter.create<memref::AllocOp>(
        loc, bundledMemPoolMemRefType, staticMemPoolAlloc.alignmentAttr());

    // The newly bundled MemRef expressed as a KrnlGetRefOp.
    auto bundledMemRef = rewriter.create<KrnlGetRefOp>(loc,
        currentAllocGetRef.getResult().getType(), newStaticMemPoolAlloc,
        offset);
    rewriter.replaceOp(currentAllocGetRef, bundledMemRef.getResult());

    // Replace old memory pool with new one.
    rewriter.replaceOp(staticMemPoolAlloc, newStaticMemPoolAlloc.getResult());

    // Update data structure to contain the newly constructed static memory
    // pool.
    alignmentToMemPool->erase(alignment);
    alignmentToMemPool->insert(
        std::pair<int64_t, memref::AllocOp>(alignment, newStaticMemPoolAlloc));

    return success();
  }
};

/*!
 *  RewritePattern that merges a new dynamic AllocOp with the existing dynamic
 *  memory pool.
 *    %dyn_mempool = alloc(%a) : memref<?xi8>
 *    %new_alloc = alloc(%b) : memref<?xi8>
 *    %new_ref = krnl.getref %new_alloc 0 : memref<?xi8>
 *  =>
 *    %c = addi %a, %b
 *    %dyn_mempool = alloc(%c) : memref<?xi8>
 *    %new_ref = krnl.getref %dyn_mempool %a : memref<?xi8>
 */

class KrnlBundleDynamicMemoryPools : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  BlockToMemPool *blockToDynamicPool;
  KrnlBundleDynamicMemoryPools(
      MLIRContext *context, BlockToMemPool *_blockToDynamicPool)
      : OpRewritePattern<memref::AllocOp>(context) {
    blockToDynamicPool = _blockToDynamicPool;
  }

  LogicalResult matchAndRewrite(
      memref::AllocOp allocOp, PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    auto memRefType = allocOp.getResult().getType().dyn_cast<MemRefType>();
    auto memRefShape = memRefType.getShape();

    // If alloca result is not used by getref then it cannot be part of
    // the memory pool.
    if (!checkOpResultIsUsedByGetRef(&allocOp))
      return failure();

    // Only handle dynamic allocs here.
    if (hasAllConstantDimensions(memRefType))
      return failure();

    // Alloc memory type must be byte.
    if (getMemRefEltSizeInBytes(memRefType) != 1)
      return failure();

    // Rank of the allocated MemRef must be 1.
    if (memRefShape.size() != 1)
      return failure();

    // Visit dependendent operations in the current parent block and assemble
    // a trace of operations which participate in the computation of the size
    // of the AllocOp.
    auto parentBlock = allocOp.getOperation()->getBlock();

    // Compute alignment.
    int64_t alignment = getAllocAlignment(allocOp);

    // If this is not the first time we process an alloc in this block, avoid
    // processing the current dynamic memory pool again.
    if (blockToDynamicPool->count(parentBlock) > 0) {
      AlignmentToMemPool *memPoolList = blockToDynamicPool->at(parentBlock);
      if (memPoolList->count(alignment) > 0 &&
          allocOp == memPoolList->at(alignment))
        return failure();
    }

    // Initialize work queue data structure.
    std::vector<Value> operandList;
    for (const auto &operand : allocOp.getOperands()) {
      operandList.emplace_back(operand);
    }

    // Check if list of operations depends on dynamic local AllocOp.
    bool dependsOnLocalDynamicAlloc = false;

    // Construct the list of Values on which the current AllocOp depends on.
    llvm::SetVector<Operation *> dependentOps;
    while (operandList.size() > 0) {
      Value currentElement = operandList[0];
      Operation *definingOperation = currentElement.getDefiningOp();

      // If this value has not been seen before, process it.
      if (dependentOps.count(definingOperation) == 0) {
        // Add value to dependent values list.
        dependentOps.insert(definingOperation);

        // Add operands to work queue.
        for (const auto &operand : definingOperation->getOperands()) {
          // Check operand is not a block argument. If it is skip it, we
          // consider block arguments to be leafs.
          if (!isBlockArgument(allocOp, operand)) {
            operandList.emplace_back(operand);

            // Check if the current operation is an AllocOp with dynamic
            // sizes or a KrnlGetRefOp.
            // If that's the case then it means that the whole set of
            // instructions cannot be moved.
            Operation *operandOp = operand.getDefiningOp();
            if (operandOp) {
              auto localAlloc = llvm::dyn_cast<memref::AllocOp>(operandOp);
              if (localAlloc) {
                auto memRefType =
                    localAlloc.getResult().getType().dyn_cast<MemRefType>();
                if (!hasAllConstantDimensions(memRefType))
                  dependsOnLocalDynamicAlloc = true;
              }

              // If operand is a getref then this alloc cannot be bundled.
              auto memPool = llvm::dyn_cast<KrnlGetRefOp>(operandOp);
              if (memPool)
                dependsOnLocalDynamicAlloc = true;
            }
          }
        }
      }

      // Erase first element from work queue.
      operandList.erase(operandList.begin());
    }

    if (dependsOnLocalDynamicAlloc)
      return failure();

    // Order the dependent values in the same order they appear in the code.
    // One cannot iterate over and make changes to the order of the operations
    // of a block. A temporary ordered list of dependent instructions is
    // necessary.
    llvm::SmallVector<Operation *, 32> orderedDependentOps;
    for (auto &op :
        llvm::make_range(parentBlock->begin(), std::prev(parentBlock->end())))
      if (dependentOps.count(&op) > 0)
        orderedDependentOps.emplace_back(&op);

    // If this is the first valid alloc we can bundle in this block, then we
    // need to move it to the top of the block as it will consitute an
    // insertion point for all other bundle-able AllocOps in the block.
    AlignmentToMemPool *alignmentToMemPool = nullptr;
    if (blockToDynamicPool->count(parentBlock) == 0) {
      allocOp.getOperation()->moveBefore(&parentBlock->front());

      // Create new entry in the block map.
      alignmentToMemPool = new AlignmentToMemPool();
      blockToDynamicPool->insert(std::pair<Block *, AlignmentToMemPool *>(
          parentBlock, alignmentToMemPool));
    } else {
      alignmentToMemPool = blockToDynamicPool->at(parentBlock);
    }

    bool isFirstBundledAllocWithThisAlignment =
        alignmentToMemPool->count(alignment) == 0;

    // This is the first dynamic alloc with this alignment.
    if (isFirstBundledAllocWithThisAlignment) {
      allocOp.getOperation()->moveBefore(&parentBlock->front());
      alignmentToMemPool->insert(
          std::pair<int64_t, memref::AllocOp>(alignment, allocOp));
    }

    // Move the computation instructions at the start of the block.
    memref::AllocOp oldDynamicMemoryPool = alignmentToMemPool->at(alignment);
    std::reverse(orderedDependentOps.begin(), orderedDependentOps.end());
    for (auto &op : orderedDependentOps)
      op->moveBefore(&parentBlock->front());

    // Bundle MemRef type: <?xi8>
    SmallVector<int64_t, 1> memPoolShape;
    memPoolShape.emplace_back(-1);
    auto bundledMemPoolMemRefType =
        MemRefType::get(memPoolShape, rewriter.getIntegerType(8));

    // Get the getref of the current allocOp. There is exactly one such getref.
    KrnlGetRefOp currentAllocGetRef = getCurrentAllocGetRef(&allocOp);
    if (!currentAllocGetRef)
      return failure();

    // Add the current alloc size to the current MemPool size.
    Value dynamicMemoryPoolSize = oldDynamicMemoryPool.getOperand(0);
    if (isFirstBundledAllocWithThisAlignment) {
      Value zero = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
      zero.getDefiningOp()->moveBefore(oldDynamicMemoryPool);
      dynamicMemoryPoolSize = zero;
    }

    arith::AddIOp bundledAllocOperand = rewriter.create<arith::AddIOp>(
        loc, dynamicMemoryPoolSize, allocOp.getOperand(0));
    bundledAllocOperand.getOperation()->moveBefore(oldDynamicMemoryPool);

    // The newly bundled MemRef expressed as a KrnlGetRefOp.
    // Current memory pool size is the offset for the newly bundled
    // internal MemRef.
    Value integerDynamicMemoryPoolSize = rewriter.create<arith::IndexCastOp>(
        loc, dynamicMemoryPoolSize, rewriter.getIntegerType(64));
    integerDynamicMemoryPoolSize.getDefiningOp()->moveBefore(
        oldDynamicMemoryPool);

    // We need to emit a new alloc which contains the additional MemRef.
    memref::AllocOp bundledAlloc = rewriter.create<memref::AllocOp>(loc,
        bundledMemPoolMemRefType, bundledAllocOperand.getResult(),
        oldDynamicMemoryPool.alignmentAttr());
    bundledAlloc.getOperation()->moveBefore(oldDynamicMemoryPool);

    KrnlGetRefOp bundledMemRef = rewriter.create<KrnlGetRefOp>(loc,
        currentAllocGetRef.getResult().getType(), bundledAlloc,
        integerDynamicMemoryPoolSize, currentAllocGetRef.getDynamicSizes());

    // The get ref can be kept in its original location.
    bundledMemRef.getOperation()->moveBefore(currentAllocGetRef);

    // Replace old memory pool with new one.
    rewriter.replaceOp(oldDynamicMemoryPool, bundledAlloc.getResult());

    // Replace old getref with new getref from new memory pool.
    rewriter.replaceOp(currentAllocGetRef, bundledMemRef.getResult());

    // Update MemPool data structure.
    alignmentToMemPool->erase(alignment);
    alignmentToMemPool->insert(
        std::pair<int64_t, memref::AllocOp>(alignment, bundledAlloc));

    return success();
  }
};

/*
 * Move all constants to the top of their respective block to avoid
 * unwanted merges.
 */
class KrnlMoveConstantsUp : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      arith::ConstantOp constOp, PatternRewriter &rewriter) const override {
    // Get parent block.
    auto parentBlock = constOp.getOperation()->getBlock();

    // Ensure it's the top block.
    if (!llvm::dyn_cast_or_null<FuncOp>(parentBlock->getParentOp()))
      return failure();

    // Move instruction to the top.
    constOp.getOperation()->moveBefore(&parentBlock->front());
    return success();
  }
};

/*!
 *  Function pass that enables memory pooling for MemRefs.
 */

class KrnlBundleMemoryPoolsPass
    : public PassWrapper<KrnlBundleMemoryPoolsPass, OperationPass<FuncOp>> {

  BlockToMemPool blockToStaticPool;
  BlockToMemPool blockToDynamicPool;

public:
  StringRef getArgument() const override { return "bundle-memory-pools"; }

  StringRef getDescription() const override {
    return "Bundle memory pools of internal MemRefs into a single memory pool.";
  }

  void runOnOperation() override {
    auto function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<KrnlBundleStaticMemoryPools>(
        &getContext(), &blockToStaticPool);
    // patterns.insert<KrnlBundleDynamicMemoryPools>(
    //     &getContext(), &blockToDynamicPool);
    patterns.insert<KrnlMoveConstantsUp>(&getContext());

    // No need to test, its ok to fail the apply.
    LogicalResult res =
        applyPatternsAndFoldGreedily(function, std::move(patterns));
    assert((succeeded(res) || failed(res)) && "remove unused var warning");

    BlockToMemPool::iterator it;
    for (it = blockToStaticPool.begin(); it != blockToStaticPool.end(); it++)
      free(it->second);
    for (it = blockToDynamicPool.begin(); it != blockToDynamicPool.end(); it++)
      free(it->second);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createKrnlBundleMemoryPoolsPass() {
  return std::make_unique<KrnlBundleMemoryPoolsPass>();
}
