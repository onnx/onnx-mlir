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

// /// Retrieve function which contains the current operation.
// FuncOp getContainingFunction(KrnlGetRefOp op) {
//   Operation *parentFuncOp = op.getParentOp();

//   // While parent is not a FuncOp and its cast to a FuncOp is null.
//   while (!llvm::dyn_cast_or_null<FuncOp>(parentFuncOp))
//     parentFuncOp = parentFuncOp->getParentOp();

//   return cast<FuncOp>(parentFuncOp);
// }

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
    bool opUsesGetRef = false;
    for (const auto &operand : op.getOperands())
      if (operand == getRef->getResult())
        opUsesGetRef = true;

    if (opUsesGetRef)
      stores.emplace_back(op);
  });

  parentBlock->walk([&stores, getRef](AffineStoreOp op) {
    bool opUsesGetRef = false;
    for (const auto &operand : op.getOperands())
      if (operand == getRef->getResult())
        opUsesGetRef = true;

    if (opUsesGetRef)
      stores.emplace_back(op);
  });

  // The list contains at least one use.
  return stores;
}

bool getRefUsesAreDisjoint(
    KrnlGetRefOp firstGetRef, KrnlGetRefOp secondGetRef) {
  // Return variable.
  bool refsUseIsDisjoint = true;

  // printf("\n\n==============START LOOP=============\n");
  // printf("First:\n");
  // firstGetRef.dump();
  // printf("Second:\n");
  // secondGetRef.dump();

  // Compute all the stores into the second getref.
  std::vector<Operation *> allStores = getGetRefStores(&secondGetRef);

  // For each store, analyze the list of dependendent operations that
  // contributes to the computation of the value being stored. The leaf
  // values are going to be represented by: load operations and constants.
  for (const auto &store : allStores) {
    // Initialize work queue data structure.
    std::vector<Value> operandList;
    operandList.emplace_back(store->getOperands()[0]);

    // printf("Current STORE: \n");
    // store->dump();

    // Construct the list of Values on which the current AllocOp depends on.
    llvm::SetVector<Operation *> dependentOps;
    while (operandList.size() > 0) {
      Value currentElement = operandList[0];
      Operation *definingOperation = currentElement.getDefiningOp();

      // If this value has not been seen before, process it.
      if (dependentOps.count(definingOperation) == 0) {
        // Add value to dependent values list.
        dependentOps.insert(definingOperation);
        // TODO: remove debug print.
        // definingOperation->dump();

        if (llvm::dyn_cast<AffineLoadOp>(definingOperation) ||
            llvm::dyn_cast<LoadOp>(definingOperation)) {
          // Check that the MemRef operand of this store operation is
          // not the firstGetRef.
          Value memRefOperand = definingOperation->getOperands()[0];
          // printf("Is block argument = %d\n", isBlockArgument(firstGetRef, memRefOperand));
          if (!isBlockArgument(firstGetRef, memRefOperand)) {
            Operation *operandDefinition = memRefOperand.getDefiningOp();
            KrnlGetRefOp storeGetRefOperand =
                llvm::dyn_cast<KrnlGetRefOp>(operandDefinition);

            // printf("Defining operation for the 2nd operand of the store:\n");
            // operandDefinition->dump();
            if (storeGetRefOperand && firstGetRef == storeGetRefOperand) {
              refsUseIsDisjoint = false;
            }
          }
        } else {
          // Add operands to work queue.
          for (const auto &operand : definingOperation->getOperands())
            if (!isBlockArgument(firstGetRef, operand))
              operandList.emplace_back(operand);
        }
      }

      // Erase first element from work queue.
      operandList.erase(operandList.begin());
    }

    // printf("===========================\n");

    // Exit if use is not disjoint.
    if (!refsUseIsDisjoint)
      break;
  }

  // printf("==============DONE LOOP=============\n");
  return refsUseIsDisjoint;
}

bool getRefUsesAreMutuallyDisjoin(
    KrnlGetRefOp firstGetRef, KrnlGetRefOp secondGetRef) {
  return getRefUsesAreDisjoint(firstGetRef, secondGetRef) &&
    getRefUsesAreDisjoint(secondGetRef, firstGetRef);
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
    auto memPoolType =
        convertToMemRefType(staticMemPool.getResult().getType());
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
    KrnlGetRefOp secondGetRef = nullptr;
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
        secondGetRef = candidate;
      }
    }

    // If no secondGetRef was found, pattern matching failed.
    if (!secondGetRef)
      return failure();

    // A suitable candidate has been found. The next step is to check that
    // the usage of the candidate getref is disjoint from the usage of the
    // first getref. This means that for any store to the secondGetRef, the
    // value stored does not involve a load from the firstGetRef.
    bool refsUseIsDisjoint =
        getRefUsesAreMutuallyDisjoin(firstGetRef, secondGetRef);
    // printf("============== DONE ANALYSIS =============\n");

    if (!refsUseIsDisjoint)
      return failure();

    printf("==============Found a replacement!=============\n");
    firstGetRef.dump();
    secondGetRef.dump();
    printf("==============END ANALYSIS=============\n");

    // A suitable replacement has been found, perform replacement, replace
    // second getref with first getref.
    rewriter.replaceOp(secondGetRef, firstGetRef.getResult());

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

    // dynamicPoolMap.insert(
    //     std::pair<FuncOp, std::unique_ptr<BlockToDynamicPool>>(
    //         function, std::make_unique<BlockToDynamicPool>()));

    // staticPoolMap.insert(std::pair<FuncOp, std::unique_ptr<BlockToStaticPool>>(
    //     function, std::make_unique<BlockToStaticPool>()));

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<KrnlOptimizeStaticMemoryPools>(
        &getContext());

    applyPatternsAndFoldGreedily(function, patterns);

    // dynamicPoolMap.erase(function);
    // staticPoolMap.erase(function);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createKrnlOptimizeMemoryPoolsPass() {
  return std::make_unique<KrnlOptimizeMemoryPoolsPass>();
}
