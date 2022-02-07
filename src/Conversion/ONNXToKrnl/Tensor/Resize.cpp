/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Resize.cpp - Lowering Resize Op ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
  ONNXResizeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXResizeOp::getOperationName(), 1, ctx) {}

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
            resizeOp.coordinate_transformation_mode() != "half_pixel"))
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
        Value resizedFVal = rewriter.create<arith::SIToFPOp>(
            loc, rewriter.getF32Type(), resizedVal);
        Value inputDim = dataBounds.getDim(i).getValue();
        Value inputDimInteger = rewriter.create<arith::IndexCastOp>(
            loc, inputDim, rewriter.getIntegerType(64));
        Value inputDimFloat = rewriter.create<arith::SIToFPOp>(
            loc, rewriter.getF32Type(), inputDimInteger);
        Value scaleVal =
            rewriter.create<arith::DivFOp>(loc, resizedFVal, inputDimFloat);
        scaleValues.emplace_back(scaleVal);
      }
    }

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else if (fromScale) {
      for (decltype(rank) i = 0; i < rank; i++) {
        if (memRefType.getShape()[i] != -1) {
          outputDims[i] = LiteralIndexExpr(memRefType.getShape()[i]);
        } else {
          Value inputDim = dataBounds.getDim(i).getValue();
          Value inputDimInteger = rewriter.create<arith::IndexCastOp>(
              loc, inputDim, rewriter.getIntegerType(64));
          Value inputDimFloat = rewriter.create<arith::SIToFPOp>(
              loc, rewriter.getF32Type(), inputDimInteger);
          Value outputDimFloat = rewriter.create<arith::MulFOp>(
              loc, inputDimFloat, scaleValues[i]);
          Value outputDimInteger = rewriter.create<arith::FPToSIOp>(
              loc, rewriter.getIntegerType(64), outputDimFloat);
          Value outDim = rewriter.create<arith::IndexCastOp>(
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
          Value outDim = rewriter.create<arith::IndexCastOp>(
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

    // Constants used in the loop body
    Value zero = emitConstantOp(rewriter, loc, rewriter.getIntegerType(64), 0);
    Value one = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);

    // Loop body
    SmallVector<Value, 4> readIndices;
    SmallVector<Value, 4> writeIndices;
    for (decltype(rank) i = 0; i < rank; ++i) {
      Value inIndexFloat;
      Value outIndex = outputLoops.getInductionVar(i);
      Value outIndexInteger = rewriter.create<arith::IndexCastOp>(
          loc, outIndex, rewriter.getIntegerType(64));
      Value outIndexFloat = rewriter.create<arith::SIToFPOp>(
          loc, rewriter.getF32Type(), outIndexInteger);

      // Handle coordinate transformation
      if (resizeOp.coordinate_transformation_mode() == "asymmetric") {
        inIndexFloat =
            rewriter.create<arith::DivFOp>(loc, outIndexFloat, scaleValues[i]);
      } else if (resizeOp.coordinate_transformation_mode() == "half_pixel") {
        // If coordinate_transformation_mode is "half_pixel",
        // x_original = (x_resized + 0.5) / scale - 0.5,
        Value halfPixelConstant =
            emitConstantOp(rewriter, loc, rewriter.getF32Type(), 0.5);
        inIndexFloat = rewriter.create<arith::SubFOp>(loc,
            rewriter.create<arith::DivFOp>(loc,
                rewriter.create<arith::AddFOp>(
                    loc, outIndexFloat, halfPixelConstant),
                scaleValues[i]),
            halfPixelConstant);
      }

      // Handle nearest_mode
      if (resizeOp.nearest_mode() == "round_prefer_floor") {
        Value deltaConstant =
            emitConstantOp(rewriter, loc, rewriter.getF32Type(), 0.499999);
        inIndexFloat =
            rewriter.create<arith::AddFOp>(loc, inIndexFloat, deltaConstant);
      } else if (resizeOp.nearest_mode() == "round_prefer_ceil") {
        Value deltaConstant =
            emitConstantOp(rewriter, loc, rewriter.getF32Type(), 0.5);
        inIndexFloat =
            rewriter.create<arith::AddFOp>(loc, inIndexFloat, deltaConstant);
      } else if (resizeOp.nearest_mode() == "floor") {
        inIndexFloat = rewriter.create<math::FloorOp>(loc, inIndexFloat);
      } else if (resizeOp.nearest_mode() == "ceil") {
        inIndexFloat = rewriter.create<math::CeilOp>(loc, inIndexFloat);
      } else {
        llvm_unreachable("Unexpected nearest_mode() for ResizeOp");
      }

      // FPToSIOp is round-to-zero, same as floor for positive
      // round_prefer_floor will round 2.5 to 2, not 3
      Value inIndexInteger = rewriter.create<arith::FPToSIOp>(
          loc, rewriter.getIntegerType(64), inIndexFloat);

      // When the index is out of bound, use the boundary index.
      // This is equivalent to np.pad with mode = "edge"

      // Compare with integer type because lower bound may be negative
      Value lessThanZero = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, inIndexInteger, zero);
      Value inIndexLBPadded =
          rewriter.create<SelectOp>(loc, lessThanZero, zero, inIndexInteger);

      // Upper bound comparison can be done with Index type
      inIndexLBPadded = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), inIndexLBPadded);
      Value inputDim = dataBounds.getDim(i).getValue();
      Value lessThanDim = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, inIndexLBPadded, inputDim);
      Value inputDimMinus = rewriter.create<arith::SubIOp>(loc, inputDim, one);
      Value inIndexPadded = rewriter.create<SelectOp>(
          loc, lessThanDim, inIndexLBPadded, inputDimMinus);

      readIndices.emplace_back(inIndexPadded);
      writeIndices.emplace_back(outIndex);
    }
    Value loadVal = rewriter.create<KrnlLoadOp>(loc, data, readIndices);
    rewriter.create<KrnlStoreOp>(loc, loadVal, alloc, writeIndices);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXResizeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXResizeOpLowering>(typeConverter, ctx);
}
