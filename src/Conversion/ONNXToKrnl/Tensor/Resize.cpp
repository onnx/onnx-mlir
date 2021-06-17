/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Resize.cpp - Lowering Resize Op
//-------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Resize Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXResizeOpLowering : public ConversionPattern {
  ONNXResizeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXResizeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Location loc = op->getLoc();
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    ONNXResizeOp sizeOp = llvm::dyn_cast<ONNXResizeOp>(op);

    ONNXResizeOpAdaptor operandAdaptor(operands);
    Value data = operandAdaptor.X();
    ArrayRef<int64_t> dataShape = data.getType().cast<MemRefType>().getShape();

    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getShape().size();

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      llvm_unreachable("unknown shape for output");

    // Get the scales
    ONNXResizeOp resizeOp = llvm::cast<ONNXResizeOp>(op);
    DenseElementsAttr scalesAttrs =
        getDenseElementAttributeFromONNXValue(resizeOp.scales());
    if (!scalesAttrs)
      llvm_unreachable("Not implemented yet");
    SmallVector<float, 4> scalesConstant;
    for (auto scaleAttr : scalesAttrs.getValues<FloatAttr>()) {
      scalesConstant.emplace_back(scaleAttr.getValueAsDouble());
    }

    // Create loops
    BuildKrnlLoop outputLoops(rewriter, loc, rank);
    outputLoops.createDefineAndIterateOp(alloc);
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // Loop body
    SmallVector<Value, 4> readIndices;
    SmallVector<Value, 4> writeIndices;
    for (decltype(rank) i = 0; i < rank; ++i) {
      Value outIndex = outputLoops.getInductionVar(i);
      Value outIndexInteger = rewriter.create<IndexCastOp>(loc, outIndex, rewriter.getIntegerType(64));
      Value outIndexFloat = rewriter.create<SIToFPOp>(loc, rewriter.getF32Type(), outIndexInteger);
      Value scaleVal = emitConstantOp(rewriter, loc, rewriter.getF32Type(), scalesConstant[i]);
      Value inIndexFloat = rewriter.create<DivFOp>(loc, outIndexFloat, scaleVal);
      Value inIndexInteger = rewriter.create<FPToSIOp>(loc, rewriter.getIntegerType(64), inIndexFloat);
      Value inIndex = rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), inIndexInteger);
      readIndices.emplace_back(inIndex);
      writeIndices.emplace_back(outIndex);
    }
    Value loadVal = rewriter.create<KrnlLoadOp>(loc, data, readIndices);
    rewriter.create<KrnlStoreOp>(loc, loadVal, alloc, writeIndices);
    loadVal.dump();

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXResizeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXResizeOpLowering>(ctx);
}
