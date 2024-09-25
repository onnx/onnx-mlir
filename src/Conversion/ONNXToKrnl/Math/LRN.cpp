/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- LRN.cpp - Lowering LRN Op -----------------------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX LRN Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXLRNOpLowering : public OpConversionPattern<ONNXLRNOp> {
  ONNXLRNOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  using LocalMultiDialectBuilder = MultiDialectBuilder<KrnlBuilder,
      IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder>;

  LogicalResult matchAndRewrite(ONNXLRNOp lrnOp, ONNXLRNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = lrnOp.getOperation();
    Location loc = ONNXLoc<ONNXLRNOp>(op);
    ValueRange operands = adaptor.getOperands();
    LocalMultiDialectBuilder create(rewriter, loc);

    // Get shape.
    ONNXLRNOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    auto outputMemRefShape = outputMemRefType.getShape();
    Type elementType = outputMemRefType.getElementType();
    int64_t outputRank = outputMemRefShape.size();

    Value input = adaptor.getX();
    float biasLit = adaptor.getBias().convertToFloat();
    float alphaLit = adaptor.getAlpha().convertToFloat();
    float betaLit = adaptor.getBeta().convertToFloat();
    int sizeLit = adaptor.getSize();
    auto f32Type = FloatType::getF32(rewriter.getContext());
    Value biasValue = create.math.constant(f32Type, biasLit);
    Value alphaDivSizeValue =
        create.math.constant(f32Type, alphaLit / static_cast<float>(sizeLit));
    Value betaValue = create.math.constant(f32Type, betaLit);

    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    ValueRange outputLoopDef = create.krnl.defineLoops(outputRank);
    SmallVector<IndexExpr, 4> lbs(outputRank, LitIE(0));
    create.krnl.iterateIE(outputLoopDef, outputLoopDef, lbs,
        shapeHelper.getOutputDims(),
        [&](const KrnlBuilder &createKrnl, ValueRange outputLoopInd) {
          // Insert computation of square_sum.
          // square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
          // where max(0, c - floor((size - 1) / 2)) <= i
          // and i<= min(C - 1, c + ceil((size - 1) / 2)).

          // Get a child IndexExpr context.
          LocalMultiDialectBuilder create(createKrnl);
          IndexExprScope childScope(&rewriter, shapeHelper.getScope());

          // Compute the lower bound and upper bound for square_sum.
          constexpr int loopIndexForC = 1;
          DimIndexExpr cIE(outputLoopInd[loopIndexForC]);
          DimIndexExpr CIE = create.krnlIE.getShapeAsDim(input, loopIndexForC);
          SymbolIndexExpr sizeIE = LitIE(sizeLit);

          SmallVector<IndexExpr, 2> lbMaxList;
          lbMaxList.emplace_back(LitIE(0));
          lbMaxList.emplace_back(cIE - (sizeIE - 1).floorDiv(LitIE(2)));

          SmallVector<IndexExpr, 2> ubMinList;
          ubMinList.emplace_back(CIE);
          ubMinList.emplace_back(cIE + 1 + (sizeIE - 1).ceilDiv(LitIE(2)));

          // Initialize sum, single scalar, no need for default alignment.
          MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);

          Value sumAlloc = create.mem.alloc(scalarMemRefType);
          create.krnl.store(create.math.constant(elementType, 0), sumAlloc);

          // Create the sum reduction loop.
          // Old style Krnl Loop definition: do not reuse pattern.
          std::vector<Value> loop;
          defineLoops(rewriter, loc, loop, 1);
          krnl::KrnlIterateOperandPack pack(rewriter, loop);
          pack.pushIndexExprsBound(lbMaxList);
          pack.pushIndexExprsBound(ubMinList);
          KrnlIterateOp iterateOp = create.krnl.iterate(pack);
          Block &iterationBlock = iterateOp.getBodyRegion().front();
          SmallVector<Value, 4> sumLoopInd(
              iterationBlock.getArguments().begin(),
              iterationBlock.getArguments().end());
          auto outputLoopBody = rewriter.saveInsertionPoint();
          rewriter.setInsertionPointToStart(&iterationBlock);

          // Compute square-sum value
          SmallVector<Value, 4> loadIndices;
          for (int i = 0; i < outputRank; i++)
            loadIndices.emplace_back(
                (i == loopIndexForC) ? sumLoopInd[0] : outputLoopInd[i]);

          Value loadVal = create.krnl.load(input, loadIndices);
          Value squareVal = create.math.mul(loadVal, loadVal);

          Value sumValue = create.krnl.load(sumAlloc);
          sumValue = create.math.add(sumValue, squareVal);
          create.krnl.store(sumValue, sumAlloc);

          // Compute and store the output
          // y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
          rewriter.restoreInsertionPoint(outputLoopBody);
          SmallVector<Value, 4> storeIndices;
          for (int i = 0; i < outputRank; ++i)
            storeIndices.emplace_back(outputLoopInd[i]);
          Value xValue = create.krnl.load(input, storeIndices);
          sumValue = create.krnl.load(sumAlloc);
          Value tempValue =
              create.math.pow(create.math.add(biasValue,
                                  create.math.mul(alphaDivSizeValue, sumValue)),
                  betaValue);
          Value resultValue = create.math.div(xValue, tempValue);

          create.krnl.store(resultValue, alloc, storeIndices);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXLRNOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLRNOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
