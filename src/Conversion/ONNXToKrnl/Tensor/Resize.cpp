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

    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);
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
          Value scaleConstant = create.math.constant(
              rewriter.getF32Type(), scaleAttr.getValueAsDouble());
          scaleValues.emplace_back(scaleConstant);
        }
      } else {
        for (decltype(rank) i = 0; i < rank; i++) {
          Value indexValue = create.math.constantIndex(i);
          Value scaleVal = create.krnl.load(scales, indexValue);
          scaleValues.emplace_back(scaleVal);
        }
      }
    } else {
      for (decltype(rank) i = 0; i < rank; i++) {
        Value indexValue = create.math.constantIndex(i);
        Value resizedVal = create.krnl.load(sizes, indexValue);
        Value resizedFVal = rewriter.create<arith::SIToFPOp>(
            loc, rewriter.getF32Type(), resizedVal);
        Value inputDim = dataBounds.getDim(i).getValue();
        Value inputDimFloat = create.math.cast(rewriter.getF32Type(), inputDim);
        Value scaleVal = create.math.div(resizedFVal, inputDimFloat);
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
          Value inputDimFloat =
              create.math.cast(rewriter.getF32Type(), inputDim);
          Value outputDimFloat = create.math.mul(inputDimFloat, scaleValues[i]);
          Value outDim = create.math.castToIndex(outputDimFloat);
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
          Value indexValue = create.math.constantIndex(i);
          Value resizedVal = create.krnl.load(sizes, indexValue);
          Value outDim = create.math.castToIndex(resizedVal);
          SymbolIndexExpr outDimIE(outDim);
          outputDims[i] = SymbolIndexExpr(outDimIE);
        }
      }
      alloc = insertAllocAndDeallocSimple(
          rewriter, op, memRefType, loc, outputDims, insertDealloc);
    }

    // Constants used in the loop body
    Value zero = create.math.constant(rewriter.getIntegerType(64), 0);
    Value one = create.math.constantIndex(1);

    // Create loops
    BuildKrnlLoop outputLoops(rewriter, loc, rank);
    outputLoops.createDefineAndIterateOp(alloc);
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // Loop body
    SmallVector<Value, 4> readIndices;
    SmallVector<Value, 4> writeIndices;
    for (decltype(rank) i = 0; i < rank; ++i) {
      Value inIndexFloat;
      Value outIndex = outputLoops.getInductionVar(i);
      Value outIndexInteger = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIntegerType(64), outIndex);
      Value outIndexFloat =
          create.math.cast(rewriter.getF32Type(), outIndexInteger);

      // Handle coordinate transformation
      if (resizeOp.coordinate_transformation_mode() == "asymmetric") {
        inIndexFloat = create.math.div(outIndexFloat, scaleValues[i]);
      } else if (resizeOp.coordinate_transformation_mode() == "half_pixel") {
        // If coordinate_transformation_mode is "half_pixel",
        // x_original = (x_resized + 0.5) / scale - 0.5,
        Value halfPixelConstant =
            create.math.constant(rewriter.getF32Type(), 0.5);
        Value addValue = create.math.add(outIndexFloat, halfPixelConstant);
        Value divValue = create.math.div(addValue, scaleValues[i]);
        inIndexFloat = create.math.sub(divValue, halfPixelConstant);
      }

      // Handle nearest_mode
      if (resizeOp.nearest_mode() == "round_prefer_floor") {
        // round_prefer_floor will round 2.5 to 2, not 3
        Value deltaConstant =
            create.math.constant(rewriter.getF32Type(), 0.499999);
        inIndexFloat = create.math.add(inIndexFloat, deltaConstant);
      } else if (resizeOp.nearest_mode() == "round_prefer_ceil") {
        Value deltaConstant = create.math.constant(rewriter.getF32Type(), 0.5);
        inIndexFloat = create.math.add(inIndexFloat, deltaConstant);
      } else if (resizeOp.nearest_mode() == "floor") {
        // Not supported by create.math
        inIndexFloat = rewriter.create<math::FloorOp>(loc, inIndexFloat);
      } else if (resizeOp.nearest_mode() == "ceil") {
        // Not supported by create.math
        inIndexFloat = rewriter.create<math::CeilOp>(loc, inIndexFloat);
      } else {
        llvm_unreachable("Unexpected nearest_mode() for ResizeOp");
      }

      // FPToSIOp is round-to-zero, same as floor for positive
      Value inIndexInteger =
          create.math.cast(rewriter.getIntegerType(64), inIndexFloat);

      // When the index is out of bound, use the boundary index.
      // This is equivalent to np.pad with mode = "edge"

      // Compare with integer type because lower bound may be negative
      Value lessThanZero = create.math.slt(inIndexInteger, zero);
      Value inIndexLBPadded =
          create.math.select(lessThanZero, zero, inIndexInteger);

      // Upper bound comparison can be done with Index type
      inIndexLBPadded = create.math.castToIndex(inIndexLBPadded);
      Value inputDim = dataBounds.getDim(i).getValue();

      Value lessThanDim = create.math.slt(inIndexLBPadded, inputDim);
      Value inputDimMinus = create.math.sub(inputDim, one);
      Value inIndexPadded =
          create.math.select(lessThanDim, inIndexLBPadded, inputDimMinus);

      readIndices.emplace_back(inIndexPadded);
      writeIndices.emplace_back(outIndex);
    }
    Value loadVal = create.krnl.load(data, readIndices);
    create.krnl.store(loadVal, alloc, writeIndices);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXResizeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXResizeOpLowering>(typeConverter, ctx);
}
