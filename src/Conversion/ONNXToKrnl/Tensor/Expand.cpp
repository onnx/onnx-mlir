/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Expand.cpp - Lowering Expand Op----------------------=== //
//
// Copyright 2020-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Expand Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

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
    Value input = operandAdaptor.getInput();
    Location loc = op->getLoc();
    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder>
        create(rewriter, loc);

    ONNXExpandOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    int64_t outputRank = outputMemRefType.getRank();

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Iterate over the output values.
    ValueRange outputLoopDef = create.krnl.defineLoops(outputRank);
    LiteralIndexExpr zeroIE(0);
    SmallVector<IndexExpr, 4> lbs(outputRank, zeroIE);
    create.krnl.iterateIE(outputLoopDef, outputLoopDef, lbs,
        shapeHelper.getOutputDims(),
        [&](KrnlBuilder &createKrnl, ValueRange outputLoopInd) {
          IndexExprScope outputScope(createKrnl, shapeHelper.getScope());
          SmallVector<IndexExpr, 4> outputLoopIndices, lhsAccessExprs;
          getIndexExprList<DimIndexExpr>(outputLoopInd, outputLoopIndices);
          LogicalResult res = shapeHelper.getAccessExprs(
              input, 0, outputLoopIndices, lhsAccessExprs);
          assert(succeeded(res) && "Could not compute access indices");
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
