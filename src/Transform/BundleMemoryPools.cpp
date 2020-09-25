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
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Insertion point for initialization instructions and the blocks used for
// inserting the initialization and main code. These blocks will disappear
// when the first canonicalization is performed because the init block
// unconditionally branches into the second block. These blocks exist only for
// the purpose of this optimization.
// The information is recorded on a per function basis.
//===----------------------------------------------------------------------===//

typedef struct ONNXOperandsInitState {
  Block *initBlock;
  Block *mainBlock;
  BranchOp branchInit;
  AllocOp dynamicMemoryPool;
} ONNXOperandsInitState;

// Helper data structure for the bundling of dynamic AllocOps.
std::map<FuncOp, std::unique_ptr<ONNXOperandsInitState>> initMap;

typedef struct InitDataStructure {
  AllocOp staticMemoryPool;
} InitDataStructure;
std::map<Block *, std::unique_ptr<InitDataStructure>> blockInitMap;

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

/// Retrieve function which contains the current operation.
FuncOp getContainingFunction(AllocOp op) {
  Operation *parentFuncOp = op.getParentOp();

  // While parent is not a FuncOp and its cast to a FuncOp is null.
  while (!llvm::dyn_cast_or_null<FuncOp>(parentFuncOp))
    parentFuncOp = parentFuncOp->getParentOp();

  return cast<FuncOp>(parentFuncOp);
}

bool hasInitBlock(FuncOp function) {
  std::unique_ptr<ONNXOperandsInitState> &initState = initMap.at(function);
  return initState->initBlock != nullptr;
}

bool addInitBlock(PatternRewriter &rewriter, Location loc, AllocOp allocOp) {
  // If this is the first time we encounter an operation in this
  // function, we create an entry inside the initMap and split the
  // function body into an init block and a main block.
  //
  // function func_name() {
  //    ... init block ...
  //    br ^bb1
  //  ^bb1:  // pred: ^bb0
  //    ... main block ...
  //    return
  // }
  //
  // Note: the block ^bb0 being the first block has its label omitted.
  //
  FuncOp function = getContainingFunction(allocOp);
  // If the function does not contain an init block, create one.
  if (!hasInitBlock(function)) {
    std::unique_ptr<ONNXOperandsInitState> &initState = initMap.at(function);
    initState = std::make_unique<ONNXOperandsInitState>();

    // All input arguments are considered as part of the initialization block
    // so add them to the operandsInInitBlock set.
    Block *functionBlock = &function.front();
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(functionBlock);

    initState->initBlock = rewriter.getInsertionBlock();
    auto currentPoint = rewriter.getInsertionPoint();
    initState->mainBlock =
        rewriter.splitBlock(initState->initBlock, currentPoint);

    rewriter.setInsertionPointToEnd(initState->initBlock);

    // Insert a branch operation from initBlock to mainBlock. This
    // ensures the final code contains legal blocks.
    initState->branchInit =
        rewriter.create<BranchOp>(loc, initState->mainBlock);

    rewriter.setInsertionPointToStart(initState->mainBlock);

    // Save a reference to the current dynamic memory pool value.
    initState->dynamicMemoryPool = allocOp;

    return true;
  }

  return false;
}

bool isBlockArgument(Block *block, Value operand) {
  for (auto arg : block->getArguments())
    if (operand == arg)
      return true;
  return false;
}

KrnlGetRefOp getUnbundledGetRef(AllocOp *memPool) {
  auto parentBlock = memPool->getOperation()->getBlock();

  parentBlock->dump();
  printf("\n\n");

  KrnlGetRefOp unbundledGetRef = nullptr;
  parentBlock->walk([&unbundledGetRef, memPool](KrnlGetRefOp op) {
    auto result = memPool->getResult();
    if (op.getOperands()[0] != result)
      unbundledGetRef = op;
  });

  return unbundledGetRef;
}

KrnlGetRefOp getCurrentAllocGetRef(AllocOp *allocOp) {
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

class KrnlBundleStaticMemoryPools : public OpRewritePattern<AllocOp> {
public:
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AllocOp allocOp, PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    auto memRefType = convertToMemRefType(allocOp.getResult().getType());
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

    if (blockInitMap.count(parentBlock) == 0) {
      // Create new entry in the block map.
      blockInitMap.insert(
          std::pair<Block *, std::unique_ptr<InitDataStructure>>(
              parentBlock, std::make_unique<InitDataStructure>()));

      // Initialize the map with the alloc which will become the static memory
      // pool. The alloc is moved at the top of the block.
      std::unique_ptr<InitDataStructure> &initState =
          blockInitMap.at(parentBlock);
      initState = std::make_unique<InitDataStructure>();
      allocOp.getOperation()->moveBefore(&parentBlock->front());
      initState->staticMemoryPool = allocOp;

      // This is the initial memory pool for this block and it is
      // trivially bundled hence it's safe to return success.
      return success();
    }

    // If this parent block has been found present in the map, it means
    // a static memory bundle already exists. Fetch it.
    std::unique_ptr<InitDataStructure> &initState =
        blockInitMap.at(parentBlock);
    AllocOp MemPoolAllocOp = initState->staticMemoryPool;

    // If this is the alloc representing the memory pool and the function
    // already has an init block, pattern matching must fail to avoid
    // processing the dynamic memory pool a second time.
    if (allocOp == initState->staticMemoryPool)
      return failure();

    auto staticMemPoolShape =
        convertToMemRefType(MemPoolAllocOp.getResult().getType()).getShape();
    int64_t currentMemPoolSize = staticMemPoolShape[0];

    // Get the getref of the current allocOp. There is exactly one such getref.
    KrnlGetRefOp currentAllocGetRef = getCurrentAllocGetRef(&allocOp);
    if (!currentAllocGetRef)
      return failure();

    // Current memory pool size is the offset for the newly bundled
    // internal MemRef. Emit the offset as a constant.
    auto offset = rewriter.create<ConstantOp>(
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
    auto newStaticMemPoolAlloc =
        rewriter.create<AllocOp>(loc, bundledMemPoolMemRefType);

    // The newly bundled MemRef expressed as a KrnlGetRefOp.
    auto bundledMemRef = rewriter.create<KrnlGetRefOp>(loc,
        currentAllocGetRef.getResult().getType(), newStaticMemPoolAlloc,
        offset);
    rewriter.replaceOp(currentAllocGetRef, bundledMemRef.getResult());

    // Replace old memory pool with new one.
    rewriter.replaceOp(MemPoolAllocOp, newStaticMemPoolAlloc.getResult());

    // Update data structure to contain the newly constructed static memory
    // pool.
    initState->staticMemoryPool = newStaticMemPoolAlloc;

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

class KrnlBundleDynamicMemoryPools : public OpRewritePattern<AllocOp> {
public:
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AllocOp allocOp, PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    auto memRefType = convertToMemRefType(allocOp.getResult().getType());
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

    // Get function.
    FuncOp function = getContainingFunction(allocOp);
    Block *firstBlock = &function.getBody().front();

    // If this is the alloc representing the memory pool and the function
    // already has an init block, pattern matching must fail to avoid
    // processing the dynamic memory pool a second time.
    if (hasInitBlock(function)) {
      std::unique_ptr<ONNXOperandsInitState> &initState = initMap.at(function);
      if (allocOp == initState->dynamicMemoryPool)
        return failure();
    }

    // Initialize work queue data structure.
    Operation *op = allocOp.getOperation();
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
        // printf("Processing the args of the following op:\n");
        for (const auto &operand : definingOperation->getOperands()) {
          // Check operand is not a block argument. If it is skip it, we
          // consider block arguments to be leafs.
          if (!isBlockArgument(firstBlock, operand)) {
            operandList.emplace_back(operand);

            // Check if the current operation is a dim or a load and the
            // argument list involves a local AllocOp with dynamic sizes.
            // If that's the case then it means that the whole set of
            // instructions cannot be moved.
            // Check if the current operation is a DimOp or a LoadOp.
            if (llvm::dyn_cast<DimOp>(definingOperation) ||
                llvm::dyn_cast<LoadOp>(definingOperation)) {
              Operation *operandOp = operand.getDefiningOp();
              if (operandOp) {
                auto localAlloc = llvm::dyn_cast<AllocOp>(operandOp);
                if (localAlloc) {
                  auto memRefType =
                      convertToMemRefType(localAlloc.getResult().getType());
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

    // If no dynamic alloc is in the trace of the dependent operations,
    // emit the size calculation in the init block, if one exists already,
    // if not, create the init block.
    bool addedInitBlock = addInitBlock(rewriter, loc, allocOp);

    // Move the ordered dependent size calculation to the init block.
    std::unique_ptr<ONNXOperandsInitState> &initState = initMap.at(function);
    for (auto &op : orderedDependentOps)
      op->moveBefore(initState->branchInit);

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
    Value dynamicMemoryPoolSize = initState->dynamicMemoryPool.getOperand(0);
    if (addedInitBlock) {
      Value zero = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
      zero.getDefiningOp()->moveBefore(initState->branchInit);
      dynamicMemoryPoolSize = zero;
    }

    AddIOp bundledAllocOperand = rewriter.create<AddIOp>(
        loc, dynamicMemoryPoolSize, allocOp.getOperand(0));
    bundledAllocOperand.getOperation()->moveBefore(initState->branchInit);

    // The newly bundled MemRef expressed as a KrnlGetRefOp.
    // Current memory pool size is the offset for the newly bundled
    // internal MemRef.
    Value integerDynamicMemoryPoolSize = rewriter.create<IndexCastOp>(
        loc, dynamicMemoryPoolSize, rewriter.getIntegerType(64));
    integerDynamicMemoryPoolSize.getDefiningOp()->moveBefore(
        initState->branchInit);

    // We need to emit a new alloc which contains the additional MemRef.
    AllocOp bundledAlloc = rewriter.create<AllocOp>(
        loc, bundledMemPoolMemRefType, bundledAllocOperand.getResult());
    bundledAlloc.getOperation()->moveBefore(&initState->mainBlock->front());

    KrnlGetRefOp bundledMemRef = rewriter.create<KrnlGetRefOp>(loc,
        currentAllocGetRef.getResult().getType(), bundledAlloc,
        integerDynamicMemoryPoolSize);
    bundledMemRef.getOperation()->moveAfter(bundledAlloc);

    // Replace old memory pool with new one.
    rewriter.replaceOp(initState->dynamicMemoryPool, bundledAlloc.getResult());

    // Replace old getref with new getref from new memory pool.
    rewriter.replaceOp(currentAllocGetRef, bundledMemRef.getResult());

    // Update MemPool size.
    initState->dynamicMemoryPool = bundledAlloc;

    return success();
  }
};

/*!
 *  Function pass that enables memory pooling for MemRefs.
 */
class KrnlBundleMemoryPoolsPass
    : public PassWrapper<KrnlBundleMemoryPoolsPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();
    initMap.insert(std::pair<FuncOp, std::unique_ptr<ONNXOperandsInitState>>(
        function, std::make_unique<ONNXOperandsInitState>()));

    // Initialize state for this function.
    std::unique_ptr<ONNXOperandsInitState> &initState = initMap.at(function);
    initState->initBlock = nullptr;

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<KrnlBundleStaticMemoryPools, KrnlBundleDynamicMemoryPools>(
        &getContext());

    applyPatternsAndFoldGreedily(function, patterns);

    initMap.erase(function);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createKrnlBundleMemoryPoolsPass() {
  return std::make_unique<KrnlBundleMemoryPoolsPass>();
}
