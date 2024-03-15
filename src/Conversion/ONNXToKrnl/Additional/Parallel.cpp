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
    // llvm::dbgs() << "ONNXParallelOpLowering start\n";
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

    // ForkOps in ParallelOp region
    // Move alloc op for return values and replace uses with them.
    // Still not replace ForkOps yet.
    //    llvm::dbgs() << "ONNXParallelOpLowering: ParallelOp before move: "
    //		 << parallelOp << "\n";
    SmallVector<ONNXForkOp, 4> forkOps;
    SmallVector<std::vector<Value>, 4> forkOuts;
    for (Operation &op : bodyBlock.getOperations()) {
      if (auto forkOp = dyn_cast<ONNXForkOp>(op)) {
        forkOps.push_back(forkOp);
        auto onnxYieldOp = llvm::dyn_cast<ONNXYieldOp>(
            forkOp.getRegion().back().getTerminator());
        if (!onnxYieldOp)
          return failure();
        // Check the return value of the block to check if operations in the
        // block is already lowered to memref-level IR such as KrnlIR. Assume
        // the block is still not lowered if the return value is still Tensor
        // type. Actual return value of ONNXYieldOp is conveted into tensor by
        // unrealized_conversion_cast. So, check the operand of previous
        // operation.
        // for (Value v : onnxYieldOp.getOperands()) {
        std::vector<Value> forkOut;
        for (unsigned ii = 0; ii < onnxYieldOp.getNumOperands(); ++ii) {
          //	  Value v = onnxYieldOp.getOperands()[ii];
          //          Value returnVal = v.getDefiningOp()->getOperands()[0];
          Value returnVal = onnxYieldOp.getOperands()[ii];
          if (isa<UnrealizedConversionCastOp>(returnVal.getDefiningOp()))
            returnVal = returnVal.getDefiningOp()->getOperands()[0];
          //          llvm::dbgs() << "ONNXParallelOpLowering: ForkOp
          //          returnValue: "
          //                       << returnVal << "\n";
          if (isa<TensorType>(returnVal.getType()))
            return failure();

          // Move allocOps for results before ForkOp
          Operation *justMovedOp = nullptr;
          // Move AllocOp's operands first.
          Operation *returnValAllocOp = returnVal.getDefiningOp();
          for (unsigned i = 0; i < returnValAllocOp->getNumOperands(); ++i) {
            Value oprd = returnValAllocOp->getOperand(i);
            if (isa<BlockArgument>(oprd))
              continue;
            Operation *opToMove = oprd.getDefiningOp();
            if (justMovedOp)
              opToMove->moveAfter(justMovedOp);
            else
              opToMove->moveBefore(forkOp);
            justMovedOp = opToMove;
          }
          // Move AllocOp.
          if (justMovedOp)
            returnValAllocOp->moveAfter(justMovedOp);
          else
            returnValAllocOp->moveBefore(forkOp);

          rewriter.replaceAllUsesWith(
              forkOp.getResults()[ii], returnValAllocOp->getResult(0));
          forkOut.push_back(returnValAllocOp->getResult(0));
        }
        forkOuts.push_back(forkOut);
      }
    }
    //    llvm::dbgs() << "ONNXParallelOpLowering: ParallelOp after move: "
    //                 << parallelOp << "\n";
    // Move allocOp in ParallelOp region
    SmallVector<Value, 4> outs;
    for (unsigned ii = 0; ii < yieldOp->getNumOperands(); ++ii) {
      Value returnVal = yieldOp->getOperands()[ii];
      if (isa<UnrealizedConversionCastOp>(returnVal.getDefiningOp()))
        returnVal = returnVal.getDefiningOp()->getOperands()[0];
      //      llvm::dbgs() << "ONNXParallelOpLowering: ParallelOp returnValue: "
      //                   << returnVal << "\n";
      if (isa<TensorType>(returnVal.getType()))
        return failure();

      // Move allocOps for results before ForkOp
      Operation *justMovedOp = nullptr;
      // Move AllocOp's operands first.
      Operation *returnValAllocOp = returnVal.getDefiningOp();
      for (unsigned i = 0; i < returnValAllocOp->getNumOperands(); ++i) {
        Value oprd = returnValAllocOp->getOperand(i);
        if (isa<BlockArgument>(oprd))
          continue;
        Operation *opToMove = oprd.getDefiningOp();
        if (justMovedOp)
          opToMove->moveAfter(justMovedOp);
        else
          opToMove->moveBefore(parallelOp);
        justMovedOp = opToMove;
      }
      // Move AllocOp.
      if (justMovedOp)
        returnValAllocOp->moveAfter(justMovedOp);
      else
        returnValAllocOp->moveBefore(parallelOp);

      rewriter.replaceAllUsesWith(
          parallelOp.getResults()[ii], returnValAllocOp->getResult(0));
      outs.push_back(returnValAllocOp->getResult(0));
    }
    //    llvm::dbgs() << "ONNXParallelOpLowering: ParallelOp after move 2: "
    //                 << parallelOp << "\n";

    // 2. Create KrnlIterateOp and replace Parallel Op with it.
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
    // rewriter.eraseOp(llvm::dyn_cast<ONNXYieldOp>(yieldOp));
    rewriter.eraseOp(yieldOp);
    rewriter.inlineBlockBefore(&bodyBlock, iterationBlock.getTerminator());
    //    rewriter.eraseOp(op); // Need this kinds of line, but not
    // here
    // llvm::dbgs() << "ONNXParallelOpLowering: iterateOp: " << iterateOp <<
    // "\n";

    // 3. Create SCFIfOp and replace ForkOp with it.
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
      // rewriter.eraseOp(forkOp);
      rewriter.replaceOp(forkOp, forkOuts[id]);
      id++;
    }
    // llvm::dbgs() << "ONNXParallelOpLowering: iterateOp: " << iterateOp <<
    // "\n"; llvm::dbgs() << "ONNXParallelOpLowering end\n";

    rewriter.replaceOp(op, outs);
    return success();
  }
};

void populateLoweringONNXParallelOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXParallelOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
