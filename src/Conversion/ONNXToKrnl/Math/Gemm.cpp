//===----------------- Gemm.cpp - Lowering Gemm Op
//-------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gemm Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

template <typename GemmOp>
struct ONNXGemmOpLowering : public ConversionPattern {
  ONNXGemmOpLowering(MLIRContext *ctx)
      : ConversionPattern(GemmOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Get shape.
    ONNXGemmOpAdaptor operandAdaptor(operands);
    ONNXGemmOp gemmOp = llvm::cast<ONNXGemmOp>(op);
    Location loc = op->getLoc();
    ONNXGemmOpShapeHelper shapeHelper(&gemmOp, &rewriter);
    assert(succeeded(shapeHelper.Compute(operandAdaptor)));
    IndexExprContext outerContext(shapeHelper.context);

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Get the constants: zero, alpha,and beta.
    float alphaLit = gemmOp.alpha().convertToFloat();
    float betaLit = gemmOp.beta().convertToFloat();
    Value alpha = emitConstantOp(rewriter, loc, elementType, alphaLit);
    Value beta = emitConstantOp(rewriter, loc, elementType, betaLit);
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);

    // Loop iterations N=0 & M-1 going over each of the res[n, m] values.
    BuildKrnlLoop outputLoops(rewriter, loc, 2);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // Compute the access functions for res[n,m].
    IndexExpr n =
        outerContext.createLoopInductionIndex(outputLoops.getInductionVar(0));
    IndexExpr m =
        outerContext.createLoopInductionIndex(outputLoops.getInductionVar(1));
    SmallVector<IndexExpr, 4> resAccessFct({n, m});

    // Insert res[n,m] = 0.
    outerContext.createStoreOp(zero, alloc, resAccessFct);

    // Create the inner reduction loop.
    BuildKrnlLoop innerLoops(rewriter, loc, 1);
    innerLoops.createDefineOp();
    innerLoops.pushBounds(0, shapeHelper.aDims[1]);
    innerLoops.createIterateOp();

    // Write code after the completion of the inner loop.
    // Compute the c access function using the broadcast rules.
    SmallVector<IndexExpr, 4> cAccessFct;
    if (shapeHelper.hasBias) {
      for (int x = 2 - shapeHelper.cRank; x < 2; ++x) {
        // If dim > 1, use loop index, otherwise broadcast on 0's element.
        IndexExpr dim = outerContext.createSymbolIndexFromParentContext(
            shapeHelper.cDims[x]);
        cAccessFct.emplace_back(IndexExpr::select(dim > 1, resAccessFct[x], 0));
      }
    }

    // Calculate reduction(AB)*alpha.
    Value loadedAB = outerContext.createLoadOp(alloc, resAccessFct);
    Value alphaAB = rewriter.create<MulFOp>(loc, alpha, loadedAB);
    if (shapeHelper.hasBias) {
      // Res = AB*alpha + beta * C.
      Value loadedC = outerContext.createLoadOp(operandAdaptor.C(), cAccessFct);
      auto betaC = rewriter.create<MulFOp>(loc, beta, loadedC);
      auto Y = rewriter.create<AddFOp>(loc, alphaAB, betaC);
      outerContext.createStoreOp(Y, alloc, resAccessFct);
    } else {
      // No bias, just store alphaAB into res.
      outerContext.createStoreOp(alphaAB, alloc, resAccessFct);
    }

    // Now start writing code inside the inner loop: get A & B access functions.
    rewriter.setInsertionPointToStart(innerLoops.getIterateBlock());
    IndexExpr k =
        outerContext.createLoopInductionIndex(innerLoops.getInductionVar(0));
    SmallVector<IndexExpr, 4> aAccessFct, bAccessFct;
    if (gemmOp.transA() != 0)
      aAccessFct = {k, n};
    else
      aAccessFct = {n, k};
    if (gemmOp.transB() != 0)
      bAccessFct = {m, k};
    else
      bAccessFct = {k, m};
    // Add mat mul operation.
    Value loadedA = outerContext.createLoadOp(operandAdaptor.A(), aAccessFct);
    Value loadedB = outerContext.createLoadOp(operandAdaptor.B(), bAccessFct);
    Value loadedY = outerContext.createLoadOp(alloc, resAccessFct);
    Value AB = rewriter.create<MulFOp>(loc, loadedA, loadedB);
    Value accumulated = rewriter.create<AddFOp>(loc, loadedY, AB);
    outerContext.createStoreOp(accumulated, alloc, resAccessFct);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXGemmOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLowering<ONNXGemmOp>>(ctx);
}
