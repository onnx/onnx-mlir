/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Slice.cpp - Lowering Slice Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Slice Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSliceOpLowering : public ConversionPattern {
  ONNXSliceOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSliceOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSliceOpAdaptor operandAdaptor(operands);
    ONNXSliceOp sliceOp = llvm::cast<ONNXSliceOp>(op);
    Location loc = op->getLoc();

    ONNXSliceOpShapeHelper shapeHelper(&sliceOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t outputRank = outputMemRefType.getShape().size();
    // Insert an allocation and deallocation for the output of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange loopDef = createKrnl.defineLoops(outputRank);
    SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
    createKrnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.dimsForOutput(),
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Compute indices for the load and store op.
          // Load: "i * step + start" for all dim.
          // Store: "i" for all dims.
          SmallVector<IndexExpr, 4> loadIndices, storeIndices;
          for (int ii = 0; ii < outputRank; ++ii) {
            DimIndexExpr inductionIndex(loopInd[ii]);
            IndexExpr start = SymbolIndexExpr(shapeHelper.starts[ii]);
            IndexExpr step = SymbolIndexExpr(shapeHelper.steps[ii]);
            loadIndices.emplace_back((step * inductionIndex) + start);
            storeIndices.emplace_back(inductionIndex);
          }
          // Load data and store in alloc data.
          Value loadVal = createKrnl.loadIE(operandAdaptor.data(), loadIndices);
          createKrnl.storeIE(loadVal, alloc, storeIndices);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSliceOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSliceOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
