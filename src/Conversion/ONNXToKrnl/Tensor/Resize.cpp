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
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

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
    ONNXResizeOp resizeOp = llvm::cast<ONNXResizeOp>(op);
    ONNXResizeOpAdaptor operandAdaptor(operands);
    Value data = operandAdaptor.X();

    // Check implementation constraints
    if (resizeOp.mode() != "nearest" ||
        resizeOp.coordinate_transformation_mode() != "asymmetric" ||
        resizeOp.nearest_mode() != "floor")
      llvm_unreachable("not implemented yet");

    // Get the scales
    DenseElementsAttr scalesAttrs =
        getDenseElementAttributeFromONNXValue(resizeOp.scales());
    if (!scalesAttrs)
      return emitError(loc, "Not implemented yet");
    SmallVector<float, 4> scalesConstant;
    for (auto scaleAttr : scalesAttrs.getValues<FloatAttr>()) {
      scalesConstant.emplace_back(scaleAttr.getValueAsDouble());
    }

    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getShape().size();

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      IndexExprScope outerloopContex(&rewriter, loc);
      DimsExpr outputDims(rank);
      MemRefBoundsIndexCapture dataBounds(data);
      for (decltype(rank) i = 0; i < rank; i++) {
        if (memRefType.getShape()[i] != -1) {
	  outputDims[i] = LiteralIndexExpr(memRefType.getShape()[i]);
        } else {
          Value inputDim = dataBounds.getDim(i).getValue();
          Value inputDimInteger = rewriter.create<IndexCastOp>(
              loc, inputDim, rewriter.getIntegerType(64));
          Value inputDimFloat = rewriter.create<SIToFPOp>(
              loc, rewriter.getF32Type(), inputDimInteger);
          Value scaleVal = emitConstantOp(
              rewriter, loc, rewriter.getF32Type(), scalesConstant[i]);
          Value outputDimFloat = rewriter.create<MulFOp>(
              loc, inputDimFloat, scaleVal);         
          Value outputDimInteger = rewriter.create<FPToSIOp>(
              loc, rewriter.getIntegerType(64), outputDimFloat);
          Value outDim = rewriter.create<IndexCastOp>(
              loc, rewriter.getIndexType(), outputDimInteger);
          SymbolIndexExpr outDimIE(outDim);
          outputDims[i] = SymbolIndexExpr(outDimIE);
        }
      }
      alloc = insertAllocAndDeallocSimple(rewriter, op, memRefType, loc, outputDims, insertDealloc);
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
      Value outIndexInteger = rewriter.create<IndexCastOp>(
          loc, outIndex, rewriter.getIntegerType(64));
      Value outIndexFloat = rewriter.create<SIToFPOp>(
          loc, rewriter.getF32Type(), outIndexInteger);
      Value scaleVal = emitConstantOp(
          rewriter, loc, rewriter.getF32Type(), scalesConstant[i]);
      Value inIndexFloat =
          rewriter.create<DivFOp>(loc, outIndexFloat, scaleVal);
      Value inIndexInteger = rewriter.create<FPToSIOp>(
          loc, rewriter.getIntegerType(64), inIndexFloat);
      Value inIndex = rewriter.create<IndexCastOp>(
          loc, rewriter.getIndexType(), inIndexInteger);
      readIndices.emplace_back(inIndex);
      writeIndices.emplace_back(outIndex);
    }
    Value loadVal = rewriter.create<KrnlLoadOp>(loc, data, readIndices);
    rewriter.create<KrnlStoreOp>(loc, loadVal, alloc, writeIndices);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXResizeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXResizeOpLowering>(ctx);
}
