/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- LRN.cpp - Lowering LRN Op -----------------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX LRN Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXLRNOpLowering : public ConversionPattern {
  ONNXLRNOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXLRNOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXLRNOpAdaptor operandAdaptor(operands);
    ONNXLRNOp lrnOp = llvm::cast<ONNXLRNOp>(op);
    auto loc = op->getLoc();

    ONNXLRNOpShapeHelper shapeHelper(&lrnOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);

    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapecomputed;
    assert(!failed(shapecomputed) && "expected to succeed");

    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputMemRefShape = outputMemRefType.getShape();
    auto elementType = outputMemRefType.getElementType();
    int64_t outputRank = outputMemRefShape.size();

    Value input = operandAdaptor.X();
    float biasLit = lrnOp.bias().convertToFloat();
    float alphaLit = lrnOp.alpha().convertToFloat();
    float betaLit = lrnOp.beta().convertToFloat();
    int sizeLit = lrnOp.size();
    auto f32Type = FloatType::getF32(rewriter.getContext());
    Value biasValue = emitConstantOp(rewriter, loc, f32Type, biasLit);
    Value alphaDivSizeValue =
        emitConstantOp(rewriter, loc, f32Type, alphaLit / (float)sizeLit);
    Value betaValue = emitConstantOp(rewriter, loc, f32Type, betaLit);

    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput());

    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange outputLoopDef = createKrnl.defineLoops(outputRank);
    SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
    createKrnl.iterateIE(outputLoopDef, outputLoopDef, lbs,
        shapeHelper.dimsForOutput(),
        [&](KrnlBuilder &createKrnl, ValueRange outputLoopInd) {
          // Insert computation of square_sum.
          // square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
          // where max(0, c - floor((size - 1) / 2)) <= i
          // and i<= min(C - 1, c + ceil((size - 1) / 2)).

          // Get a child IndexExpr context.
          IndexExprScope childScope(&rewriter, shapeHelper.scope);

          // Compute the lower bound and upper bound for square_sum.
          constexpr int loopIndexForC = 1;
          Value cValue = outputLoopInd[loopIndexForC];
          DimIndexExpr cIE(cValue);
          MemRefBoundsIndexCapture inputBounds(input);
          DimIndexExpr CIE(inputBounds.getDim(loopIndexForC));
          SymbolIndexExpr sizeIE = LiteralIndexExpr(sizeLit);

          SmallVector<IndexExpr, 2> lbMaxList;
          lbMaxList.emplace_back(LiteralIndexExpr(0));
          lbMaxList.emplace_back(
              cIE - (sizeIE - 1).floorDiv(LiteralIndexExpr(2)));

          SmallVector<IndexExpr, 2> ubMinList;
          ubMinList.emplace_back(CIE);
          ubMinList.emplace_back(
              cIE + 1 + (sizeIE - 1).ceilDiv(LiteralIndexExpr(2)));

          // Initialize sum, single scalar, no need for default alignment.
          MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
          MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
              rewriter, loc);

          Value sumAlloc = create.mem.alloc(scalarMemRefType);
          createKrnl.store(
              emitConstantOp(rewriter, loc, elementType, 0), sumAlloc);

          // Create the sum reduction loop
          krnl::BuildKrnlLoop sumLoops(rewriter, loc, 1);
          sumLoops.createDefineOp();
          sumLoops.pushBounds(lbMaxList, ubMinList);
          sumLoops.createIterateOp();
          auto outputLoopBody = rewriter.saveInsertionPoint();
          rewriter.setInsertionPointToStart(sumLoops.getIterateBlock());

          // Compute quare-sum value
          SmallVector<Value, 4> loadIndices;
          for (int i = 0; i < outputRank; i++)
            loadIndices.emplace_back((i == loopIndexForC)
                                         ? sumLoops.getInductionVar(0)
                                         : outputLoopInd[i]);

          Value loadVal = createKrnl.load(input, loadIndices);
          Value squareVal = create.math.mul(loadVal, loadVal);

          Value sumValue = createKrnl.load(sumAlloc, ArrayRef<Value>{});
          sumValue = create.math.add(sumValue, squareVal);
          createKrnl.store(sumValue, sumAlloc, ArrayRef<Value>{});

          // Compute and store the output
          // y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
          rewriter.restoreInsertionPoint(outputLoopBody);
          SmallVector<Value, 4> storeIndices;
          for (int i = 0; i < outputRank; ++i) {
            storeIndices.emplace_back(outputLoopInd[i]);
          }
          Value xValue = createKrnl.load(input, storeIndices);
          sumValue = createKrnl.load(sumAlloc);
          Value tempValue =
              create.math.pow(create.math.add(biasValue,
                                  create.math.mul(alphaDivSizeValue, sumValue)),
                  betaValue);
          Value resultValue = create.math.div(xValue, tempValue);

          createKrnl.store(resultValue, alloc, storeIndices);
        });

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXLRNOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLRNOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
