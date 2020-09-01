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
#include "llvm/ADT/ArrayRef.h"
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
// The support happens on a per function basis.
//===----------------------------------------------------------------------===//

typedef struct ONNXOperandsInitState {
  Block *initBlock;
  Block *mainBlock;
  BranchOp branchInit;
  AllocOp dynamicMemoryPool;
  Value currentMemPoolSize;
  // llvm::SetVector<Value> operandsInInitBlock;
} ONNXOperandsInitState;

typedef std::map<FuncOp, std::unique_ptr<ONNXOperandsInitState>>
  FunctionToInitStates;

// This map is used by the FrontendToKrnlLoweringPass pass to keep track of the
// allocations emitted in the initialization block for each function of a given
// module. A translation unit can consist of several modules, each with several
// functions hence the structure shown below.
// This data structure enables the emission of dyanmic `alloc` instructions
// in the initialization block of a function if all the other operands the
// computation of its parameters depends on are also present in that function's
// initialization block.
// This data structure is live only during the execution of the frontend
// lowering to Krnl dialect pass (FrontendToKrnlLoweringPass).
std::map<ModuleOp, std::unique_ptr<FunctionToInitStates>> initMap;

/// Retrieve function which contains the current operation.
FuncOp getContainingFunction(AllocOp op) {
  Operation *parentFuncOp = op.getParentOp();

  // While parent is not a FuncOp and its cast to a FuncOp is null.
  while (!llvm::dyn_cast_or_null<FuncOp>(parentFuncOp))
    parentFuncOp = parentFuncOp->getParentOp();

  return cast<FuncOp>(parentFuncOp);
}

FuncOp getContainingFunctionForOp(Operation *op) {
  Operation *parentFuncOp = op->getParentOp();

  // While parent is not a FuncOp and its cast to a FuncOp is null.
  while (!llvm::dyn_cast_or_null<FuncOp>(parentFuncOp))
    parentFuncOp = parentFuncOp->getParentOp();

  return cast<FuncOp>(parentFuncOp);
}

// // The alloc cannot be part of the memory pool if the size computation
// // depends on an intermediate AllocOp.
// bool allocCanBePartOfMemPool(AllocOp) {
//   bool canBeIncludedInMemPool = true;

//   // TODO
//   auto parentBlock = allocOp->getOperation()->getBlock();

//   return canBeIncludedInMemPool;
// }

void addInitBlock(PatternRewriter &rewriter, Location loc, std::unique_ptr<FunctionToInitStates> &initStates, AllocOp allocOp) {
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
  // ModuleOp module = cast<ModuleOp>(function.getParentOp());

  // // If this is the first time we encounter the module, add it.
  // if (initMap.count(module) == 0) {
  //   initMap.insert(std::pair<ModuleOp, std::unique_ptr<FunctionToInitStates>>(
  //      module, std::make_unique<FunctionToInitStates>()));
  // }

  // std::unique_ptr<FunctionToInitStates> &initStates = initMap.at(module);
  if (initStates->count(function) == 0) {
    initStates->insert(
        std::pair<FuncOp, std::unique_ptr<ONNXOperandsInitState>>(
            function, std::make_unique<ONNXOperandsInitState>()));
    std::unique_ptr<ONNXOperandsInitState> &initState =
        initStates->at(function);

    // All input arguments are considered as part of the initialization block
    // so add them to the operandsInInitBlock set.
    Block *functionBlock = &function.front();
    // for (auto arg : functionBlock->getArguments())
    //   initState->operandsInInitBlock.insert(arg);

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

    // Save a reference to the current dynamic memory pool value.
    initState->dynamicMemoryPool = allocOp;

    // Move it to the start of the mainBlock.
    // The allocs can now be moved because an alloc always precedes
    // any getrefs accessing it. No getrefs are ever moved inside
    // the init block.
    allocOp.moveBefore(initState->mainBlock->front());

    // Current memory pool size is zero.
    initState->currentMemPoolSize =
        emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
  }
}

bool isBlockArgument(Block *block, Value operand) {
  for (auto arg : block->getArguments())
    if (operand == arg)
      return true;
  return false;
}

// void printDependentOps(AllocOp *allocOp) {
//   auto parentBlock = allocOp->getOperation()->getBlock();

//   // Initialize work queue data structure.
//   Operation *op = allocOp->getOperation();
//   std::vector<Value> operandList;
//     for (const auto &operand : allocOp->getOperands()) {
//     operandList.emplace_back(operand);
//   }

//   // Check if list of operations depends on dynamic local allocs.
//   bool dependsOnLocalAlloc = false;

//   // Construct the list of Values on which the current AllocOp depends on.
//   llvm::SetVector<Operation *> dependentOps;
//   while (operandList.size() > 0) {
//     Value currentElement = operandList[0];
//     Operation *definingOperation = currentElement.getDefiningOp();

//     // If this value has not been seen before, process it.
//     if (dependentOps.count(definingOperation) == 0) {
//       // Add value to dependent values list.
//       dependentOps.insert(definingOperation);

//       // Add operands to work queue.
//       for (const auto &operand : definingOperation->getOperands()) {
//         // Check operand is not a block argument. If it is skip it, we
//         // consider block arguments to be leafs.
//         if (!isBlockArgument(parentBlock, operand))
//           operandList.emplace_back(operand);

//         // Check if the current operation is a dim or a load and the
//         // argument list involves a local AllocOp with dynamic sizes.
//         // If that's the case then it means that the whole set of
//         // instructions cannot be moved.
//         // Check if the current operation is a DimOp or a LoadOp.
//         if (llvm::dyn_cast<DimOp>(definingOperation) ||
//             llvm::dyn_cast<LoadOp>(definingOperation)) {
//           Operation *operandOp = operand.getDefiningOp();
//           if (operandOp) {
//             auto localAlloc = llvm::dyn_cast<AllocOp>(operandOp);
//             if (localAlloc) {
//               auto memRefType =
//                   convertToMemRefType(localAlloc.getResult().getType());
//               if (!hasAllConstantDimensions(memRefType))
//                 dependsOnLocalAlloc = true;
//             }
//           }
//         }
//       }
//     }

//     // Erase first element from work queue.
//     operandList.erase(operandList.begin());
//   }

//   // Print the dependent values in the correct order.
//   for (auto &op : llvm::make_range(parentBlock->begin(), std::prev(parentBlock->end()))) {
//     if (dependentOps.count(&op) > 0) {
//       op.dump();
//     }
//   }

//   printf("==================> %d\n", dependsOnLocalAlloc);
// }

KrnlGetRefOp getUnbundledGetRef(AllocOp *memPool) {
  auto parentBlock = memPool->getOperation()->getBlock();

  KrnlGetRefOp unbundledGetRef = nullptr;
  parentBlock->walk([&unbundledGetRef, memPool](KrnlGetRefOp op) {
    auto result = memPool->getResult();
    if (op.getOperands()[0] != result)
      unbundledGetRef = op;
  });

  return unbundledGetRef;
}

KrnlGetRefOp getCurrentAllocGetRef(AllocOp allocOp) {
  auto parentBlock = allocOp.getOperation()->getBlock();

  KrnlGetRefOp unbundledGetRef = nullptr;
  parentBlock->walk([&unbundledGetRef, memPool](KrnlGetRefOp op) {
    auto result = memPool->getResult();
    if (op.getOperands()[0] != result)
      unbundledGetRef = op;
  });

  return unbundledGetRef;
}

/*!
 *  RewritePattern that replaces:
 *    %mem1 = alloc() : memref<<dims1>x<type>>
 *    %mem2 = alloc() : memref<<dims2>x<type>>
 *    %1 = krnl.getref %mem2 0 : memref<<dims2>x<type>>
 *  =>
 *    %mem1 = alloc() : memref<<dims1 + dims2>x<type>>
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

class KrnlBundleMemoryPools : public OpRewritePattern<AllocOp> {
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

    // TODO: remove once we support the bundling of dynamic memory pools.
    if (!hasAllConstantDimensions(memRefType))
      return failure();

    // Alloc memory type must be byte.
    if (getMemRefEltSizeInBytes(memRefType) != 1)
      return failure();

    // Rank of the allocated MemRef must be 1.
    if (memRefShape.size() != 1)
      return failure();

    // TODO: Change this when dyanmic shapes are supported.
    // TODO: Add support for dynamic shapes.
    int64_t currentMemPoolSize = memRefShape[0];

    // Get a KrnlGetRefOp which does not use the current alloc.
    if (KrnlGetRefOp unbundledGetRef = getUnbundledGetRef(&allocOp)) {
      // Current memory pool size is the offset for the newly bundled
      // internal MemRef. Emit the offset as a constant.
      auto offset = rewriter.create<ConstantOp>(
          loc, rewriter.getIntegerAttr(
                   rewriter.getIntegerType(64), currentMemPoolSize));

      // Size in bytes of the output of the krnl.getref operation.
      int64_t unbundledTotalSize =
          getMemRefSizeInBytes(unbundledGetRef.getResult());

      // Compute new size.
      int64_t bundleTotalSize = unbundledTotalSize + currentMemPoolSize;

      // We need to emit a new alloc which contains the additional MemRef.
      SmallVector<int64_t, 1> newMemPoolShape;
      newMemPoolShape.emplace_back(bundleTotalSize);
      auto bundledMemPoolMemRefType =
          MemRefType::get(newMemPoolShape, rewriter.getIntegerType(8));
      auto bundledAlloc =
          rewriter.create<AllocOp>(loc, bundledMemPoolMemRefType);

      // The newly bundled MemRef expressed as a KrnlGetRefOp.
      auto bundledMemRef = rewriter.create<KrnlGetRefOp>(
          loc, unbundledGetRef.getResult().getType(), bundledAlloc, offset);
      rewriter.replaceOp(unbundledGetRef, bundledMemRef.getResult());

      // Replace old memory pool with new one.
      rewriter.replaceOp(allocOp, bundledAlloc.getResult());

      return success();
    }

    return failure();
  }
};

class KrnlBundleDynamicMemoryPools : public OpRewritePattern<AllocOp> {
public:
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AllocOp allocOp, PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    auto memRefType = convertToMemRefType(allocOp.getResult().getType());
    auto memRefShape = memRefType.getShape();

    printf("Currently looking at: \n");
    allocOp.getOperation()->dump();

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

    // Module where function resides.
    ModuleOp module = cast<ModuleOp>(function.getParentOp());
    std::unique_ptr<FunctionToInitStates> &initStates = initMap.at(module);

    // If this is the alloc representing the memory pool and the function
    // already has an init block, pattern matching must fail to avoid
    // processing the dynamic memory pool a second time.
    if (initStates->count(function) != 0 &&
        allocOp == initState->dynamicMemoryPool)
      return failure();

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

    printf("==================> %d\n", dependsOnLocalDynamicAlloc);
    if (dependsOnLocalDynamicAlloc)
      return failure();

    // Order the dependent values in the same order they appear in the code.
    // One cannot iterate over and make changes to the order of the operations
    // of a block. A temporary ordered list of dependent instructions is
    // necessary.
    llvm::SmallVector<Operation *, 32> orderedDependentOps;
    for (auto &op : llvm::make_range(parentBlock->begin(), std::prev(parentBlock->end()))) {
      if (dependentOps.count(&op) > 0) {
        orderedDependentOps.emplace_back(&op);
        op.dump();
      }
    }

    // If no dynamic alloc is in the trace of the dependent operations,
    // emit the size calculation in the init block, if one exists already,
    // if not, create the init block.
    addInitBlock(rewriter, loc, initStates, allocOp);

    // Move the ordered dependent size calculation to the init block.
    std::unique_ptr<ONNXOperandsInitState> &initState =
        initStates->at(function);
    for (auto &op : orderedDependentOps)
      op->moveBefore(initState->branchInit);

    // Bundle MemRef type: <?xi8>
    SmallVector<int64_t, 1> memPoolShape;
    newMemPoolShape.emplace_back(-1);
    auto bundledMemPoolMemRefType =
        MemRefType::get(newMemPoolShape, rewriter.getIntegerType(8));

    // Get the getref of the current allocOp. There is exactly one such getref.
    KrnlGetRefOp unbundledGetRef = getCurrentAllocGetRef(allocOp);

    if (KrnlGetRefOp unbundledGetRef = getUnbundledGetRef(&allocOp)) {
      // Add the current alloc size to the current MemPool size.
      Value bundledAllocOperand = rewriter.create<AddIOp>(loc, initState->currentMemPoolSize, allocOp.getOperand(0))

      // We need to emit a new alloc which contains the additional MemRef.
      auto bundledAlloc =
          rewriter.create<AllocOp>(loc, bundledMemPoolMemRefType, bundledAllocOperand);

      // The newly bundled MemRef expressed as a KrnlGetRefOp.
      // Current memory pool size is the offset for the newly bundled
      // internal MemRef.
      auto bundledMemRef = rewriter.create<KrnlGetRefOp>(
          loc, unbundledGetRef.getResult().getType(), bundledAlloc, initState->currentMemPoolSize);
      rewriter.replaceOp(unbundledGetRef, bundledMemRef.getResult());

      // Replace old memory pool with new one.
      rewriter.replaceOp(allocOp, bundledAlloc.getResult());

      // Update MemPool size.
      initState->currentMemPoolSize = bundledAllocOperand;

      return success();
    }

    // initState->dynamicMemoryPool = op;
    // initState->currentMemPoolSize = op.getOperand(0);

    // addInitBlock(rewriter, loc, allocOp);

    // In case this is the first AllocOp add init block and initialize
    // the data structure for keeping track of the memory pool.
    // addInitBlock(rewriter, loc, allocOp);
    // printDependentOps(&allocOp);
    return failure();
  }
};

// struct FuncOpSignatureConversion : public OpRewritePattern<FuncOp> {
//    FuncOpSignatureConversion(MLIRContext *ctx, TypeConverter &converter)
//        : OpConversionPattern(converter, ctx) {}

//     /// Hook for derived classes to implement combined matching and rewriting.
//    LogicalResult matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
//        ConversionPatternRewriter &rewriter) const override {
//      FunctionType type = funcOp.getType();

//       // Convert the original function types.
//      TypeConverter::SignatureConversion result(type.getNumInputs());
//      SmallVector<Type, 1> newResults;
//      if (failed(typeConverter->convertSignatureArgs(type.getInputs(), result)) ||
//          failed(typeConverter->convertTypes(type.getResults(), newResults)) ||
//          failed(rewriter.convertRegionTypes(
//              &funcOp.getBody(), *typeConverter, &result)))
//        return failure();

//       // Update the function signature in-place.
//      rewriter.updateRootInPlace(funcOp, [&] {
//        funcOp.setType(FunctionType::get(
//            result.getConvertedTypes(), newResults, funcOp.getContext()));
//      });
//      addInitBlock(rewriter, funcOp.getLoc(), funcOp);
//      return success();
//    }
//  };

/*!
 *  Function pass that enables memory pooling for MemRefs.
 */
class KrnlBundleMemoryPoolsPass
    : public PassWrapper<KrnlBundleMemoryPoolsPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    ModuleOp module = cast<ModuleOp>(function.getParentOp());
    initMap.insert(std::pair<ModuleOp, std::unique_ptr<FunctionToInitStates>>(
       module, std::make_unique<FunctionToInitStates>()));

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<KrnlBundleMemoryPools, KrnlBundleDynamicMemoryPools>(&getContext());

    applyPatternsAndFoldGreedily(function, patterns);

    initMap.erase(module);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createKrnlBundleMemoryPoolsPass() {
  return std::make_unique<KrnlBundleMemoryPoolsPass>();
}
