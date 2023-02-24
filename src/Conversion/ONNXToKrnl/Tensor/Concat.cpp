/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Concat Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConcatOpLowering : public ConversionPattern {
  ONNXConcatOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXConcatOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Location loc = op->getLoc();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    ONNXConcatOpAdaptor operandAdaptor(operands);
    ONNXConcatOp concatOp = llvm::cast<ONNXConcatOp>(op);
    // Get shape.
    ONNXConcatOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    auto axis = concatOp.getAxis();
    assert(axis >= 0 && "negative axis is supposed to have been normalized");
    unsigned int inputNum = operands.size();

    // Convert the output type to MemRefType.
    Type outputTensorType = *op->result_type_begin();
    Type convertedType = typeConverter->convertType(outputTensorType);
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    auto resultShape = outputMemRefType.getShape();
    unsigned int rank = resultShape.size();

    // Alloc and dealloc.
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), alignment);

    // Creates loops, one for each input.
    // Since the each input should have same size for each dimension(except
    // axis), we will try to make the loop upper bound the same for further
    // optimization. Difference may come from constant vs. dynamic, or dynamic
    // dim of different inputs.
    SmallVector<IndexExpr, 4> commonUB(shapeHelper.getOutputDims());
    // IndexExprScope IEScope(&rewriter, loc);
    IndexExpr accumulatedOffset = LiteralIndexExpr(0);
    for (unsigned int i = 0; i < inputNum; ++i) {
      // Since the accumulatedOffsetValue will be used in a nested
      // IndexExprScope, we get the Value of this IndexExpr and pass it as a
      // symbol
      Value accumulatedOffsetValue = accumulatedOffset.getValue();
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Create loop.
      ValueRange loopDef = create.krnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(operands[i], ubs);
      // For each input, only the dimension 'axis' is different
      commonUB[axis] = ubs[axis];
      create.krnl.iterateIE(loopDef, loopDef, lbs, commonUB,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            // Indices for the read and write.
            SmallVector<Value, 4> readIndices, writeIndices;
            for (unsigned int r = 0; r < rank; ++r) {
              if (r != axis || i == 0)
                writeIndices.emplace_back(loopInd[r]);
              else {
                IndexExprScope IEScope(&rewriter, loc);
                IndexExpr writeOffset = DimIndexExpr(loopInd[r]);
                IndexExpr accumulatedOffsetIE =
                    SymbolIndexExpr(accumulatedOffsetValue);
                writeOffset = writeOffset + accumulatedOffsetIE;
                writeIndices.emplace_back(writeOffset.getValue());
              }
            }
            // Insert copy.
            Value loadData = createKrnl.load(operands[i], loopInd);
            createKrnl.store(loadData, alloc, writeIndices);
          });
      accumulatedOffset =
          accumulatedOffset + create.krnlIE.getShapeAsDim(operands[i], axis);
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXConcatOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConcatOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
