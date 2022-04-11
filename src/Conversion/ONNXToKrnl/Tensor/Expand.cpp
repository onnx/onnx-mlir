/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Expand.cpp - Lowering Expand Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Expand Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXExpandOpLowering : public ConversionPattern {
  ONNXExpandOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXExpandOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXExpandOpAdaptor operandAdaptor(operands);
    ONNXExpandOp expandOp = llvm::dyn_cast<ONNXExpandOp>(op);
    Value input = operandAdaptor.input();
    Location loc = op->getLoc();
    ONNXExpandOpShapeHelper shapeHelper(&expandOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Failed to compute shape");

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t outputRank = outputMemRefType.getRank();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Iterate over the output values.
    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange outputLoopDef = createKrnl.defineLoops(outputRank);
    LiteralIndexExpr zeroIE(0);
    SmallVector<IndexExpr, 4> lbs(outputRank, zeroIE);
    createKrnl.iterateIE(outputLoopDef, outputLoopDef, lbs,
        shapeHelper.dimsForOutput(0),
        [&](KrnlBuilder &createKrnl, ValueRange outputLoopInd) {
          IndexExprScope outputScope(createKrnl, shapeHelper.scope);
          SmallVector<IndexExpr, 4> outputLoopIndices, lhsAccessExprs;
          getIndexExprList<DimIndexExpr>(outputLoopInd, outputLoopIndices);
          LogicalResult res = shapeHelper.GetAccessExprs(
              input, 0, outputLoopIndices, lhsAccessExprs);
          assert(succeeded(res));
          Value val = createKrnl.loadIE(input, lhsAccessExprs);
          createKrnl.store(val, alloc, outputLoopInd);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXExpandOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXExpandOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
