/*
 * SPDX-License-Identifier: Apache-2.0
 */

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

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;

namespace {

bool checkOpResultIsReturned(memref::AllocOp *allocOp) {
  FuncOp function = getContainingFunction(allocOp->getOperation());

  bool opIsReturned = false;

  // Check if the result value of `allocOp` is an operand of
  // `ReinterpretCastOp`, and store the result value of `ReinterpretCastOp`.
  // Reshape, Squeeze, and Unsqueeze ops are checked because they are lowered to
  // `ReinterpretCastOp`.
  SmallVector<Value, 32> castOpResults;
  function.walk([allocOp, &castOpResults](Operation *op) {
    if (isa<memref::ReinterpretCastOp>(op) || isa<memref::CastOp>(op) ||
        isa<ONNXReshapeOp>(op) || isa<ONNXSqueezeV11Op>(op) ||
        isa<ONNXUnsqueezeV11Op>(op)) {
      auto result = allocOp->getResult();
      for (const auto &operand : op->getOperands())
        if (operand == result ||
            std::find(castOpResults.begin(), castOpResults.end(), operand) !=
                castOpResults.end())
          castOpResults.emplace_back(op->getResults()[0]);
    }
  });

  function.walk([&opIsReturned, allocOp, castOpResults](ReturnOp op) {
    auto result = allocOp->getResult();
    for (const auto &operand : op.getOperands()) {
      // Determine if current function returns the result value of the
      // alloc op.
      if (operand == result)
        opIsReturned = true;
      // Determin if the result value of reinterpret_cast op whose operand
      // is the result value of alloc op
      for (const auto &castOpResult : castOpResults)
        if (operand == castOpResult)
          opIsReturned = true;
    }
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

class KrnlEnableMemoryPool : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp allocOp, PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    auto memRefType = allocOp.getResult().getType().dyn_cast<MemRefType>();

    // The MemRef type returned by the AllocOp must be normalized.
    if (!memRefType.getLayout().isIdentity())
      return failure();

    // Filter out MemRefs with Index type.
    auto elementType = memRefType.getElementType();
    if (elementType.isIndex())
      return failure();

    // If alloc operation is not returned then it is a candidate for
    // being included in the memory pool.
    if (checkOpResultIsReturned(&allocOp))
      return failure();

    // Check the result of this alloc is not already used by a krnl.getref.
    if (checkOpResultIsUsedByGetRef(&allocOp))
      return failure();

    // Get parent block.
    Block *parentBlock = allocOp.getOperation()->getBlock();

    // Only enable pooling for top level memrefs.
    if (!llvm::dyn_cast_or_null<FuncOp>(parentBlock->getParentOp()))
      return failure();

    // For now only handle constant MemRefs.
    if (!hasAllConstantDimensions(memRefType))
      return failure();

    memref::AllocOp newAlloc;
    SmallVector<int64_t, 1> memPoolShape;
    if (hasAllConstantDimensions(memRefType)) {
      // Compute total size.
      int64_t totalSize = getMemRefSizeInBytes(allocOp.getResult());

      // Emit new alloc.
      memPoolShape.emplace_back(totalSize);
      auto memPoolMemRefType =
          MemRefType::get(memPoolShape, rewriter.getIntegerType(8));
      newAlloc = rewriter.create<memref::AllocOp>(
          loc, memPoolMemRefType, allocOp.alignmentAttr());
    } else {
      memPoolShape.emplace_back(-1);
      auto memPoolMemRefType =
          MemRefType::get(memPoolShape, rewriter.getIntegerType(8));

      Value dyanmicTotalSize =
          getDynamicMemRefSizeInBytes(memRefType, loc, rewriter, allocOp);
      newAlloc = rewriter.create<memref::AllocOp>(
          loc, memPoolMemRefType, dyanmicTotalSize, allocOp.alignmentAttr());
    }

    // Emit new dealloc.
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, newAlloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());

    // Get reference to local MemRef.
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
    KrnlGetRefOp poolMemRef;
    if (hasAllConstantDimensions(memRefType)) {
      poolMemRef =
          rewriter.create<KrnlGetRefOp>(loc, memRefType, newAlloc, zero);
    } else {
      poolMemRef = rewriter.create<KrnlGetRefOp>(
          loc, memRefType, newAlloc, zero, allocOp.getDynamicSizes());
    }

    rewriter.replaceOp(allocOp, poolMemRef.getResult());

    return success();
  }
};

class KrnlEliminateOldDealloc : public OpRewritePattern<memref::DeallocOp> {
public:
  using OpRewritePattern<memref::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      memref::DeallocOp deallocOp, PatternRewriter &rewriter) const override {
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
    : public PassWrapper<KrnlEnableMemoryPoolPass, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const override { return "enable-memory-pool"; }

  StringRef getDescription() const override {
    return "Enable a memory pool for allocating internal MemRefs.";
  }

  void runOnOperation() override {
    auto function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<KrnlEnableMemoryPool>(&getContext());
    patterns.insert<KrnlEliminateOldDealloc>(&getContext());

    // No need to test, its ok to fail the apply.
    LogicalResult res =
        applyPatternsAndFoldGreedily(function, std::move(patterns));
    assert((succeeded(res) || failed(res)) && "remove unused var warning");
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createKrnlEnableMemoryPoolPass() {
  return std::make_unique<KrnlEnableMemoryPoolPass>();
}
