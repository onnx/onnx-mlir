//===----------------- Matmul.cpp - Lowering Matmul Op --------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXMatMulOpLowering : public ConversionPattern {
  ONNXMatMulOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Get shape.
    ONNXMatMulOpAdaptor operandAdaptor(operands);
    ONNXMatMulOp matMulOp = llvm::cast<ONNXMatMulOp>(op);
    Location loc = op->getLoc();
    ONNXMatMulOpShapeHelper shapeHelper(&matMulOp, &rewriter);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    (void)shapecomputed;
    assert(succeeded(shapecomputed));
    IndexExprContext outerContext(shapeHelper.context);

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Get the constants: zero.
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);

    // Non-reduction loop iterations: output-rank.
    int outerloopNum = shapeHelper.dimsForOutput(0).size();
    BuildKrnlLoop outputLoops(rewriter, loc, outerloopNum);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // Access function for the output, and set it to zero.
    SmallVector<IndexExpr, 4> resAccessFct;
    outerContext.createLoopInductionIndicesFromArrayValues(
        outputLoops.getAllInductionVar(), resAccessFct);
    // Insert res[...] = 0.
    outerContext.createStoreOp(zero, alloc, resAccessFct);

    // Create the inner reduction loop; trip count is last dim of A.
    BuildKrnlLoop innerLoops(rewriter, loc, 1);
    innerLoops.createDefineOp();
    int aRank = shapeHelper.aDims.size();
    int bRank = aRank; // Add for better readability.
    innerLoops.pushBounds(0, shapeHelper.aDims[aRank - 1]);
    innerLoops.createIterateOp();

    // Now start writing code inside the inner loop: get A & B access functions.
    rewriter.setInsertionPointToStart(innerLoops.getIterateBlock());
    IndexExpr k =
        outerContext.createLoopInductionIndex(innerLoops.getInductionVar(0));
    SmallVector<IndexExpr, 4> aAccessFct, bAccessFct;
    for (int i = 0; i < aRank; ++i) {
      // Add index if dim is not a padded dimension.
      if (!shapeHelper.aPadDims[i]) {
        // For A, reduction index is last
        if (i == aRank - 1) {
          aAccessFct.emplace_back(k);
        } else {
          aAccessFct.emplace_back(resAccessFct[i]);
        }
      }
      if (!shapeHelper.bPadDims[i]) {
        // For B, reduction index is second to last.
        if (i == bRank - 2) {
          bAccessFct.emplace_back(k);
        } else if (i == outerloopNum) {
          // When the rank of A 1D, then the output lost one dimension.
          // E,g, (5) x (10, 5, 4) -> padded (1, 5) x (10, 5, 4) = (10, 1, 4).
          // But we drop the "1" so its really (10, 4). When processing the
          // last dim of the reduction (i=2 here), we would normally access
          // output[2] but it does not exist, because we lost a dim in the
          // output due to 1D A.
          bAccessFct.emplace_back(resAccessFct[i - 1]);
        } else {
          bAccessFct.emplace_back(resAccessFct[i]);
        }
      }
    }

    // Add mat mul operation.
    Value loadedA = outerContext.createLoadOp(operandAdaptor.A(), aAccessFct);
    Value loadedB = outerContext.createLoadOp(operandAdaptor.B(), bAccessFct);
    Value loadedY = outerContext.createLoadOp(alloc, resAccessFct);
    Value AB = rewriter.create<MulFOp>(loc, loadedA, loadedB);
    Value accumulated = rewriter.create<AddFOp>(loc, loadedY, AB);
    outerContext.createStoreOp(accumulated, alloc, resAccessFct);

    // Done.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXMatMulOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLowering>(ctx);
}
