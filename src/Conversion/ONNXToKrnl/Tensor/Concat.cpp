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
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

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
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapecomputed;
    assert(succeeded(shapecomputed) && "Could not compute output shape");

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
    KrnlBuilder createKrnl(rewriter, loc);
    for (unsigned int i = 0; i < inputNum; ++i) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Create loop.
      ValueRange loopDef = createKrnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
      MemRefBoundsIndexCapture bounds(operands[i]);
      SmallVector<IndexExpr, 4> ubs;
      bounds.getDimList(ubs);
      createKrnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            // Indices for the read and write.
            SmallVector<Value, 4> readIndices, writeIndices;
            for (unsigned int r = 0; r < rank; ++r) {
              if (r != axis || i == 0)
                writeIndices.emplace_back(loopInd[r]);
              else {
                IndexExprScope IEScope(&rewriter, loc);
                IndexExpr writeOffset = DimIndexExpr(loopInd[r]);
                for (unsigned int j = 0; j < i; j++) {
                  MemRefBoundsIndexCapture operandJBounds(operands[j]);
                  writeOffset = writeOffset + operandJBounds.getDim(r);
                }
                writeIndices.emplace_back(writeOffset.getValue());
              }
            }
            // Insert copy.
            Value loadData = createKrnl.load(operands[i], loopInd);
            createKrnl.store(loadData, alloc, writeIndices);
          });
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
