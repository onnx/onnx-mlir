/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- Parallel.cpp - Lowering Parallel Op and Fork Op
//---------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Parallel and Fork Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "lowering-parallelop-to-krnl"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Helper function
// Return true if `a` happens before `b`, i.e., `a` or one of its ancestors
// properly dominates `b` and `b` is not inside `a`.
// Reference: llvm-project/mlir/lib/Dialect/Transform/IR/TransformInterfaces.cpp
//===----------------------------------------------------------------------===//

static bool happensBefore(Operation *a, Operation *b) {
  do {
    if (a->isProperAncestor(b))
      return false;
    if (Operation *bAncestor = a->getBlock()->findAncestorOpInBlock(*b)) {
      return a->isBeforeInBlock(bAncestor);
    }
  } while ((a = a->getParentOp()));
  return false;
}

void moveAllocOpOperands(SmallVector<Operation *, 4> &opsToMove,
    SmallVector<Operation *, 4> &globalOpsToMove, Operation *parentOp) {
  if (opsToMove.size() == 0)
    return;

  SmallVector<Operation *, 4> nextOpsToMove;
  for (Operation *op : opsToMove) {
    // Added the op in ops list to move if it is still not added.
    if (llvm::find(globalOpsToMove, op) == globalOpsToMove.end()) {
      globalOpsToMove.push_back(op);
      LLVM_DEBUG(llvm::dbgs() << "Added in ops list to move : " << *op << "\n");
    }
    // Check if operands need to be moved. Need to move if defining op for the
    // operand exists in block in parentOp.
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      Value oprd = op->getOperand(i);
      if (isa<BlockArgument>(oprd))
        continue;
      Operation *oprdDefOp = oprd.getDefiningOp();
      Region &parentOpRegion = parentOp->getRegions().front();
      Block &parentOpBlock = parentOpRegion.getBlocks().front();
      LLVM_DEBUG(
          llvm::dbgs() << "Operand DefOp " << i << " = " << *oprdDefOp << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "Operand DefOp need to be moved?:  "
                 << (oprdDefOp->getBlock() == &parentOpBlock) << "\n");
      if (oprdDefOp->getBlock() == &parentOpBlock)
        nextOpsToMove.push_back(oprdDefOp);
    }
  }
  // Check if operands of operandDefOp need to be moved recursively
  moveAllocOpOperands(nextOpsToMove, globalOpsToMove, parentOp);
}

LogicalResult moveAllocOpBeforeAndReplaceAllUses(
    ConversionPatternRewriter &rewriter, Operation *op, Operation *yieldOp) {
  SmallVector<Operation *, 4> globalOpsToMove;
  for (unsigned ii = 0; ii < yieldOp->getNumOperands(); ++ii) {
    // Check the return value of the block to check if operations in the
    // block is already lowered to memref-level IR such as KrnlIR. Assume
    // the block is still not lowered if the return value is still Tensor
    // type. Actual return value of ONNXYieldOp is conveted into tensor by
    // unrealized_conversion_cast. So, check the operand of previous
    // operation.
    Value returnVal = yieldOp->getOperands()[ii];
    if (isa<UnrealizedConversionCastOp>(returnVal.getDefiningOp()))
      returnVal = returnVal.getDefiningOp()->getOperands()[0];
    if (isa<TensorType>(returnVal.getType()))
      return failure();

    // Move allocOps for results before op
    Operation *allocOpForReturnVal = returnVal.getDefiningOp();
    SmallVector<Operation *, 4> opsToMove;
    opsToMove.push_back(allocOpForReturnVal);
    moveAllocOpOperands(opsToMove, globalOpsToMove, op);
    rewriter.replaceAllUsesWith(
        op->getResults()[ii], allocOpForReturnVal->getResult(0));
  }

  llvm::sort(globalOpsToMove,
      [](Operation *a, Operation *b) { return !happensBefore(a, b); });
  Operation *justMovedOp = op;
  for (Operation *gop : globalOpsToMove) {
    gop->moveBefore(justMovedOp);
    justMovedOp = gop;
  }
  return success();
}

struct ONNXParallelOpLowering : public OpConversionPattern<ONNXParallelOp> {
  explicit ONNXParallelOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXParallelOp parallelOp,
      ONNXParallelOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = parallelOp.getOperation();
    Location loc = ONNXLoc<ONNXParallelOp>(op);
    IndexExprScope ieScope(&rewriter, loc);
    ValueRange operands = adaptor.getOperands();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    auto onnxParallelOp = dyn_cast<ONNXParallelOp>(op);
    // Get the parallel region.
    Region &parallelBody = onnxParallelOp.getBody();
    // Make sure the region has only one block.
    if (!parallelBody.hasOneBlock())
      return success();
    // Get YieldOp of the body block.
    Block &bodyBlock = parallelBody.front();
    Operation *yieldOp = bodyBlock.getTerminator();
    if (!isa<ONNXYieldOp>(yieldOp))
      return failure();

    // Move alloc ops included in ForkOps
    SmallVector<ONNXForkOp, 4> forkOps;
    for (Operation &bOp : bodyBlock.getOperations()) {
      if (auto forkOp = dyn_cast<ONNXForkOp>(bOp)) {
        forkOps.push_back(forkOp);
        Operation *forkYieldOp = forkOp.getBody().front().getTerminator();
        if (!isa<ONNXYieldOp>(forkYieldOp))
          return failure();

        if (failed(moveAllocOpBeforeAndReplaceAllUses(
                rewriter, &bOp, forkYieldOp)))
          return failure();
      }
    }

    // Move allocOp included in ParallelOp
    if (failed(moveAllocOpBeforeAndReplaceAllUses(rewriter, op, yieldOp)))
      return failure();

    // Create KrnlIterateOp and replace ParallelOp with it.
    rewriter.setInsertionPoint(op);
    std::vector<Value> loop;
    defineLoops(rewriter, loc, loop, 1);
    krnl::KrnlIterateOperandPack pack(rewriter, loop);
    pack.pushConstantBound(0);
    pack.pushConstantBound(forkOps.size());
    KrnlBuilder createKrnl(rewriter, loc);
    createKrnl.parallel(loop);
    KrnlIterateOp iterateOp = createKrnl.iterate(pack);
    Block &iterationBlock = iterateOp.getBodyRegion().back();
    rewriter.setInsertionPointToStart(&iterationBlock);
    ValueRange indices = createKrnl.getInductionVarValue({loop[0]});
    rewriter.eraseOp(yieldOp);
    rewriter.inlineBlockBefore(&bodyBlock, iterationBlock.getTerminator());

    // Create SCFIfOp and replace ForkOp with it.
    int64_t id = 0;
    for (auto forkOp : forkOps) {
      rewriter.setInsertionPoint(forkOp);
      // Insert scf::IfOp
      Value forkId = create.math.constantIndex(id);
      Value eq = create.math.eq(forkId, indices[0]);
      scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, eq, /*else=*/false);
      Block &ifBlock = ifOp.getThenRegion().back();
      // Insert KrnlRegionOp within scf::IfOp
      rewriter.setInsertionPointToStart(&ifBlock);
      KrnlRegionOp regionOp = rewriter.create<KrnlRegionOp>(loc);
      Block &regionBlock = regionOp.getBodyRegion().front();
      rewriter.setInsertionPointToStart(&regionBlock);
      // Insert KrnlNoneOp. This op is used for inserting forkBlock into
      // regionBlock. This op is deleted after doing it.
      KrnlNoneOp noneOp = rewriter.create<KrnlNoneOp>(loc);
      // Delete terminator of forkRegion.
      Block &forkBlock = forkOp.getRegion().back();
      Operation *forkYieldOp = forkBlock.getTerminator();
      rewriter.eraseOp(forkYieldOp);
      // Insert forkBlock into regionBlock
      rewriter.inlineBlockBefore(&forkOp.getRegion().back(), noneOp);
      rewriter.eraseOp(noneOp);
      rewriter.eraseOp(forkOp);
      id++;
    }

    rewriter.eraseOp(op);
    return success();
  }
};

void populateLoweringONNXParallelOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXParallelOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
