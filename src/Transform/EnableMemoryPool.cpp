//===-------- EnableMemoryPool.cpp - Enable Memory Pool for MemRefs -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// For certain cases the number of individual memory allocations required for
// all internal tensors is large and needs to be mitigated. This pass enables a
// managed memory pool for allocating MemRefs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

bool checkOpResultIsReturned(AllocOp *allocOp) {
  FuncOp function = getContainingFunction(allocOp->getOperation());

  bool opIsReturned = false;
  function.walk([&opIsReturned, allocOp](ReturnOp op) {
    auto result = allocOp->getResult();
    for (const auto &operand : op.getOperands())
      if (operand == result)
        opIsReturned = true;
  });

  return opIsReturned;
}

/*!
 *  RewritePattern that replaces:
 *    %0 = alloc() : memref<<dims>x<type>>
 *  with:
 *    %mem = alloc() : memref<<dims>x<type>>
 *    %0 = krnl.getref %mem <offset> : memref<<dims>x<type>>
 *
 *  For now, to enable testing, offset will always be 0.
 */

class KrnlEnableMemoryPool : public OpRewritePattern<AllocOp> {
public:
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AllocOp allocOp, PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    auto memRefType = convertToMemRefType(allocOp.getResult().getType());

    // For now we only support constant tensors.
    // TODO: Enable this pass for MemRef with dyanmic shapes.
    // If alloc operation is not returned then it is a candidate for
    // being included in the memory pool.
    if (checkOpResultIsReturned(&allocOp))
      return failure();

    // Check the result of this alloc is not already used by a krnl.getref.
    if (checkOpResultIsUsedByGetRef(&allocOp))
      return failure();

    AllocOp newAlloc;
    SmallVector<int64_t, 1> memPoolShape;
    if (hasAllConstantDimensions(memRefType)) {
      // Compute total size.
      int64_t totalSize = getMemRefSizeInBytes(allocOp.getResult());

      // Emit new alloc.
      memPoolShape.emplace_back(totalSize);
      auto memPoolMemRefType =
          MemRefType::get(memPoolShape, rewriter.getIntegerType(8));
      newAlloc = rewriter.create<AllocOp>(loc, memPoolMemRefType);
    } else {
      memPoolShape.emplace_back(-1);
      auto memPoolMemRefType =
          MemRefType::get(memPoolShape, rewriter.getIntegerType(8));

      Value dyanmicTotalSize =
          getDynamicMemRefSizeInBytes(memRefType, loc, rewriter, allocOp);
      newAlloc =
          rewriter.create<AllocOp>(loc, memPoolMemRefType, dyanmicTotalSize);
    }

    // Emit new dealloc.
    auto dealloc = rewriter.create<DeallocOp>(loc, newAlloc);
    auto parentBlock = allocOp.getOperation()->getBlock();
    dealloc.getOperation()->moveBefore(&parentBlock->back());

    // Get reference to local MemRef.
    auto zero = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
    auto poolMemRef =
        rewriter.create<KrnlGetRefOp>(loc, memRefType, newAlloc, zero);

    rewriter.replaceOp(allocOp, poolMemRef.getResult());

    return success();
  }
};

class KrnlEliminateOldDealloc : public OpRewritePattern<DeallocOp> {
public:
  using OpRewritePattern<DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      DeallocOp deallocOp, PatternRewriter &rewriter) const override {
    if (auto getRefOp = llvm::dyn_cast<KrnlGetRefOp>(
            deallocOp.getOperand().getDefiningOp())) {
      rewriter.eraseOp(deallocOp);
      return success();
    }

    return failure();
  }
};

// TODO: Replace old dealloc with krnl.unsetref.

/*!
 *  Function pass that enables memory pooling for MemRefs.
 */
class KrnlEnableMemoryPoolPass
    : public PassWrapper<KrnlEnableMemoryPoolPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<KrnlEnableMemoryPool>(&getContext());
    patterns.insert<KrnlEliminateOldDealloc>(&getContext());

    applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createKrnlEnableMemoryPoolPass() {
  return std::make_unique<KrnlEnableMemoryPoolPass>();
}
