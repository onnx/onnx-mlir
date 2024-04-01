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

using namespace mlir;

namespace onnx_mlir {

void moveAllocOpOperands(
    SmallVector<Operation *, 4> &opsToMove, Operation *justMovedOp) {
  llvm::dbgs() << "opsToMove.size() = " << opsToMove.size() << "\n";
  if (opsToMove.size() == 0)
    return;

  SmallVector<Operation *, 4> opsToMove1;
  for (Operation *op : opsToMove) {
    llvm::dbgs() << "opToMove : " << *op << "\n";
    if (op->getNumOperands() == 0) {
      op->moveBefore(justMovedOp);
      justMovedOp = op;
    } else {
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        Value oprd = op->getOperand(i);
        if (isa<BlockArgument>(oprd))
          continue;
        Operation *oprdDefOp = oprd.getDefiningOp();
        llvm::dbgs() << "oprdDefOp " << i << " = " << *oprdDefOp << "\n";
        llvm::dbgs() << "oprdDefOp need to be moved?:  "
                     << (oprdDefOp->getBlock() != justMovedOp->getBlock())
                     << "\n";
        if (oprdDefOp->getBlock() != justMovedOp->getBlock())
          opsToMove1.push_back(oprdDefOp);
        op->moveBefore(justMovedOp);
        justMovedOp = op;
      }
    }
  }
  moveAllocOpOperands(opsToMove1, justMovedOp);
}

LogicalResult moveAllocOpBeforeAndReplaceAllUses(
    ConversionPatternRewriter &rewriter, Operation *op, Operation *yieldOp) {

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
    moveAllocOpOperands(opsToMove, op);
    rewriter.replaceAllUsesWith(
        op->getResults()[ii], allocOpForReturnVal->getResult(0));
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

    // Move alloc ops included ForkOps
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

    // Move allocOp in ParallelOp region
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
      Value forkId = create.math.constantIndex(id);
      Value eq = create.math.eq(forkId, indices[0]);
      scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, eq, /*else=*/false);
      Block &ifBlock = ifOp.getThenRegion().back();
      Block &forkBlock = forkOp.getRegion().back();
      Operation *forkYieldOp = forkBlock.getTerminator();
      rewriter.eraseOp(forkYieldOp);
      rewriter.inlineBlockBefore(
          &forkOp.getRegion().back(), ifBlock.getTerminator());
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
