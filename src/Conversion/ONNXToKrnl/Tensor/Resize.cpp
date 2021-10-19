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
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

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
    Value scales = operandAdaptor.scales();
    Value sizes = operandAdaptor.sizes();
    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getShape().size();

    // Check implementation constraints
    if (resizeOp.mode() != "nearest" ||
        (resizeOp.coordinate_transformation_mode() != "asymmetric" &&
            resizeOp.coordinate_transformation_mode() != "half_pixel") ||
        (resizeOp.nearest_mode() != "floor" &&
            resizeOp.nearest_mode() != "round_prefer_floor"))
      return emitError(loc, "not implemented yet");

    SmallVector<Value, 4> scaleValues;
    bool fromScale = !isFromNone(resizeOp.scales());
    IndexExprScope outerloopContex(&rewriter, loc);
    DimsExpr outputDims(rank);
    MemRefBoundsIndexCapture dataBounds(data);
    if (fromScale) {
      // Get the scales
      // SymbolIndexExpr was tried but got runtime error
      // Attribute::cast() const [with U = mlir::IntegerAttr]
      // The reason seems to be that IntegerAttr is assumed
      //
      DenseElementsAttr scalesAttrs =
          getDenseElementAttributeFromONNXValue(resizeOp.scales());
      SmallVector<float, 4> scalesConstant;
      if (scalesAttrs) {
        for (auto scaleAttr : scalesAttrs.getValues<FloatAttr>()) {
          Value scaleConstant = emitConstantOp(rewriter, loc,
              rewriter.getF32Type(), scaleAttr.getValueAsDouble());
          scaleValues.emplace_back(scaleConstant);
        }
      } else {
        for (decltype(rank) i = 0; i < rank; i++) {
          Value indexValue =
              emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
          Value scaleVal = rewriter.create<KrnlLoadOp>(loc, scales, indexValue);
          scaleValues.emplace_back(scaleVal);
        }
      }
    } else {
      for (decltype(rank) i = 0; i < rank; i++) {
        Value indexValue =
            emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
        Value resizedVal = rewriter.create<KrnlLoadOp>(loc, sizes, indexValue);
        Value resizedFVal =
            rewriter.create<SIToFPOp>(loc, rewriter.getF32Type(), resizedVal);
        Value inputDim = dataBounds.getDim(i).getValue();
        Value inputDimInteger = rewriter.create<IndexCastOp>(
            loc, inputDim, rewriter.getIntegerType(64));
        Value inputDimFloat = rewriter.create<SIToFPOp>(
            loc, rewriter.getF32Type(), inputDimInteger);
        Value scaleVal =
            rewriter.create<DivFOp>(loc, resizedFVal, inputDimFloat);
        scaleValues.emplace_back(scaleVal);
      }
    }

    // Keep the code using IndexExpr for bug fixing
    // ArrayValueIndexCapture scaleIEs(op, scales,
    // getDenseElementAttributeFromKrnlValue,
    // loadDenseElementArrayValueAtIndex); for (decltype(rank) i = 0; i < rank;
    // i++) {
    //  scaleValues.emplace_back(scaleIEs.getSymbol(i).getValue());
    // }

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else if (fromScale) {
      for (decltype(rank) i = 0; i < rank; i++) {
        if (memRefType.getShape()[i] != -1) {
          outputDims[i] = LiteralIndexExpr(memRefType.getShape()[i]);
        } else {
          Value inputDim = dataBounds.getDim(i).getValue();
          Value inputDimInteger = rewriter.create<IndexCastOp>(
              loc, inputDim, rewriter.getIntegerType(64));
          Value inputDimFloat = rewriter.create<SIToFPOp>(
              loc, rewriter.getF32Type(), inputDimInteger);
          Value outputDimFloat =
              rewriter.create<MulFOp>(loc, inputDimFloat, scaleValues[i]);
          Value outputDimInteger = rewriter.create<FPToSIOp>(
              loc, rewriter.getIntegerType(64), outputDimFloat);
          Value outDim = rewriter.create<IndexCastOp>(
              loc, rewriter.getIndexType(), outputDimInteger);
          SymbolIndexExpr outDimIE(outDim);
          outputDims[i] = SymbolIndexExpr(outDimIE);
        }
      }
      alloc = insertAllocAndDeallocSimple(
          rewriter, op, memRefType, loc, outputDims, insertDealloc);
    } else {
      // Output is determined by sizes()
      for (decltype(rank) i = 0; i < rank; i++) {
        if (memRefType.getShape()[i] != -1) {
          outputDims[i] = LiteralIndexExpr(memRefType.getShape()[i]);
        } else {
          Value indexValue =
              emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
          Value resizedVal =
              rewriter.create<KrnlLoadOp>(loc, sizes, indexValue);
          Value outDim = rewriter.create<IndexCastOp>(
              loc, rewriter.getIndexType(), resizedVal);
          SymbolIndexExpr outDimIE(outDim);
          outputDims[i] = SymbolIndexExpr(outDimIE);
        }
      }
      alloc = insertAllocAndDeallocSimple(
          rewriter, op, memRefType, loc, outputDims, insertDealloc);
    }

    // Create loops
    BuildKrnlLoop outputLoops(rewriter, loc, rank);
    outputLoops.createDefineAndIterateOp(alloc);
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // Loop body
    SmallVector<Value, 4> readIndices;
    SmallVector<Value, 4> writeIndices;
    for (decltype(rank) i = 0; i < rank; ++i) {
      if (resizeOp.coordinate_transformation_mode() == "asymmetric") {
        Value outIndex = outputLoops.getInductionVar(i);
        Value outIndexInteger = rewriter.create<IndexCastOp>(
            loc, outIndex, rewriter.getIntegerType(64));
        Value outIndexFloat = rewriter.create<SIToFPOp>(
            loc, rewriter.getF32Type(), outIndexInteger);
        Value inIndexFloat =
            rewriter.create<DivFOp>(loc, outIndexFloat, scaleValues[i]);
        // FPToSIOp is round-to-zero, same as floor for positive
        // round_prefer_floor will round 2.5 to 2, not 3
        if (resizeOp.nearest_mode() == "round_prefer_floor") {
          Value deltaConstant =
              emitConstantOp(rewriter, loc, rewriter.getF32Type(), 0.499999);
          inIndexFloat =
              rewriter.create<AddFOp>(loc, inIndexFloat, deltaConstant);
        } else if (resizeOp.nearest_mode() == "floor") {
          inIndexFloat = rewriter.create<FloorFOp>(loc, inIndexFloat);
        }
        Value inIndexInteger = rewriter.create<FPToSIOp>(
            loc, rewriter.getIntegerType(64), inIndexFloat);
        Value inIndex = rewriter.create<IndexCastOp>(
            loc, rewriter.getIndexType(), inIndexInteger);
        readIndices.emplace_back(inIndex);
        writeIndices.emplace_back(outIndex);
      } else if (resizeOp.coordinate_transformation_mode() == "half_pixel") {
        // If coordinate_transformation_mode is "half_pixel",
        // x_original = (x_resized + 0.5) / scale - 0.5,
        Value outIndex = outputLoops.getInductionVar(i);
        Value outIndexInteger = rewriter.create<IndexCastOp>(
            loc, outIndex, rewriter.getIntegerType(64));
        Value outIndexFloat = rewriter.create<SIToFPOp>(
            loc, rewriter.getF32Type(), outIndexInteger);
        Value halfPixelConstant =
            emitConstantOp(rewriter, loc, rewriter.getF32Type(), 0.5);
        Value inIndexFloat = rewriter.create<SubFOp>(loc,
            rewriter.create<DivFOp>(loc,
                rewriter.create<AddFOp>(loc, outIndexFloat, halfPixelConstant),
                scaleValues[i]),
            halfPixelConstant);
        if (resizeOp.nearest_mode() == "round_prefer_floor") {
          Value deltaConstant =
              emitConstantOp(rewriter, loc, rewriter.getF32Type(), 0.499999);
          inIndexFloat =
              rewriter.create<AddFOp>(loc, inIndexFloat, deltaConstant);
        } else if (resizeOp.nearest_mode() == "floor") {
          inIndexFloat = rewriter.create<FloorFOp>(loc, inIndexFloat);
        }
        Value inIndexInteger = rewriter.create<FPToSIOp>(
            loc, rewriter.getIntegerType(64), inIndexFloat);
        Value inIndex = rewriter.create<IndexCastOp>(
            loc, rewriter.getIndexType(), inIndexInteger);
        readIndices.emplace_back(inIndex);
        writeIndices.emplace_back(outIndex);
      }
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
