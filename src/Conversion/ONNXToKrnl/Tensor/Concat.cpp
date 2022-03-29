/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Concat Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXConcatOpLowering : public ConversionPattern {
  ONNXConcatOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXConcatOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    auto loc = op->getLoc();

    ONNXConcatOpAdaptor operandAdaptor(operands);
    ONNXConcatOp concatOp = llvm::cast<ONNXConcatOp>(op);
    ONNXConcatOpShapeHelper shapeHelper(&concatOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapecomputed;
    assert(succeeded(shapecomputed));

    auto axis = concatOp.axis();
    unsigned int inputNum = operands.size();

    // Alloc and dealloc.
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto resultShape = outputMemRefType.getShape();
    unsigned int rank = resultShape.size();

    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);

    // Creates loops, one for each input.
    for (unsigned int i = 0; i < inputNum; ++i) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Create loop.
      BuildKrnlLoop inputLoops(rewriter, loc, rank);
      inputLoops.createDefineOp();
      for (unsigned int r = 0; r < rank; ++r)
        inputLoops.pushBounds(0, operands[i], r);
      inputLoops.createIterateOp();
      rewriter.setInsertionPointToStart(inputLoops.getIterateBlock());

      // Indices for the read and write.
      SmallVector<Value, 4> readIndices;
      SmallVector<Value, 4> writeIndices;
      for (unsigned int r = 0; r < rank; ++r) {
        readIndices.emplace_back(inputLoops.getInductionVar(r));
        if (r != axis || i == 0)
          writeIndices.emplace_back(inputLoops.getInductionVar(r));
        else {
          IndexExprScope IEScope(&rewriter, loc);
          IndexExpr writeOffset = DimIndexExpr(inputLoops.getInductionVar(r));
          for (unsigned int j = 0; j < i; j++) {
            MemRefBoundsIndexCapture operandJBounds(operands[j]);
            writeOffset = writeOffset + operandJBounds.getDim(r);
          }
          writeIndices.emplace_back(writeOffset.getValue());
        }
      }
      // Insert copy.
      Value loadData = create.krnl.load(operands[i], readIndices);
      create.krnl.store(loadData, alloc, writeIndices);
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXConcatOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConcatOpLowering>(typeConverter, ctx);
}
