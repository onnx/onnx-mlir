/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- LRN.cpp - Lowering LRN Op -----------------------===//
//
// Copyright 2020-2023 The IBM Research Authors.
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

struct ONNXLRNOpLowering : public ConversionPattern {
  ONNXLRNOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXLRNOp::getOperationName(), 1, ctx) {}

  using LocalMultiDialectBuilder = MultiDialectBuilder<KrnlBuilder,
      IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder>;

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXLRNOpAdaptor operandAdaptor(operands);
    ONNXLRNOp lrnOp = llvm::cast<ONNXLRNOp>(op);
    Location loc = op->getLoc();
    LocalMultiDialectBuilder create(rewriter, loc);

    // Get shape.
    ONNXLRNOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();

    auto outputMemRefShape = outputMemRefType.getShape();
    Type elementType = outputMemRefType.getElementType();
    int64_t outputRank = outputMemRefShape.size();

    Value input = operandAdaptor.getX();
    float biasLit = lrnOp.getBias().convertToFloat();
    float alphaLit = lrnOp.getAlpha().convertToFloat();
    float betaLit = lrnOp.getBeta().convertToFloat();
    int sizeLit = lrnOp.getSize();
    auto f32Type = FloatType::getF32(rewriter.getContext());
    Value biasValue = create.math.constant(f32Type, biasLit);
    Value alphaDivSizeValue =
        create.math.constant(f32Type, alphaLit / (float)sizeLit);
    Value betaValue = create.math.constant(f32Type, betaLit);

    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    ValueRange outputLoopDef = create.krnl.defineLoops(outputRank);
    SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
    create.krnl.iterateIE(outputLoopDef, outputLoopDef, lbs,
        shapeHelper.getOutputDims(),
        [&](KrnlBuilder &createKrnl, ValueRange outputLoopInd) {
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

          Value sumValue = create.krnl.load(sumAlloc, ArrayRef<Value>{});
          sumValue = create.math.add(sumValue, squareVal);
          create.krnl.store(sumValue, sumAlloc, ArrayRef<Value>{});

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
    return success();
  }
};

void populateLoweringONNXLRNOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLRNOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
