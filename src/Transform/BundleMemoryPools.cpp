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

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

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
      unbundledGetRef.dump();

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

/*!
 *  Function pass that enables memory pooling for MemRefs.
 */
class KrnlBundleMemoryPoolsPass
    : public PassWrapper<KrnlBundleMemoryPoolsPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<KrnlBundleMemoryPools>(&getContext());

    applyPatternsAndFoldGreedily(function, patterns);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createKrnlBundleMemoryPoolsPass() {
  return std::make_unique<KrnlBundleMemoryPoolsPass>();
}
