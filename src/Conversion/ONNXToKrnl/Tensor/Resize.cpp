/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Resize.cpp - Lowering Resize Op ---------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Resize Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXResizeOpLowering : public OpConversionPattern<ONNXResizeOp> {
  ONNXResizeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXResizeOp resizeOp,
      ONNXResizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Operation *op = resizeOp.getOperation();
    Location loc = ONNXLoc<ONNXResizeOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value data = adaptor.getX();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);
    int64_t rank = memRefType.getShape().size();

    // Check limitation imposed by implementation
    // Resize Op is either lowered to loop nests or a function call.
    // In either case, only the default value for some of the attributes are
    // allowed.
    // In the library for Resize, src/Runtime/OMResize.inc, it seems that
    // it is easy to support some other values, but not tested.
    if (resizeOp.getAntialias() != 0 ||
        resizeOp.getCubicCoeffA().convertToDouble() != -0.75 ||
        resizeOp.getExcludeOutside() != 0 ||
        resizeOp.getExtrapolationValue().convertToDouble() != 0. ||
        resizeOp.getExcludeOutside() != 0 ||
        resizeOp.getKeepAspectRatioPolicy() != "stretch") {
      return emitError(
          loc, "attribute value not supported by current implementation#1");
    }

    // When getMode() is "nearest", Resize is lowered to loops.
    // getCoordinateTransformationMode() can be "asymmetric" or "half_pixel"
    if (resizeOp.getMode() == "nearest") {
      if (resizeOp.getCoordinateTransformationMode() != "asymmetric" &&
          resizeOp.getCoordinateTransformationMode() != "half_pixel") {
        return emitError(
            loc, "attribute value not supported by current implementation#2");
      }
    } else {
      // Resize is lowered to a library call.
      // The library assumes that getCoordinateTransformationMode()
      // is "half_pixel"
      if (resizeOp.getCoordinateTransformationMode() != "half_pixel") {
        return emitError(
            loc, "attribute value not supported by current implementation#3");
      }
    }

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Shape helper: compute output dims and scales.
    ONNXResizeOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    Value alloc =
        create.mem.alignedAlloc(memRefType, shapeHelper.getOutputDims());

    // Call external function when the mode is NOT "nearest"
    // Create KrnlCallOp and replace the du chain
    // One of inputs, getScales() and size(), has to be None.
    // For now, None input is picked out by KrnlCall builder,
    // and different function will be called accordingly.
    // Another issue is the attributes with default value.
    // Currently, it is assumed that all the optional attributes have
    // the default value and does appear in the Attribute dictionary.
    // ToFix: Handle attributes for general case
    // A list of attribute names for krnl.call is provided to determine which
    // attributes are passed to the call and in which order
    // The unspecified the attributes are assumed to have the default value.
    if (resizeOp.getMode() != "nearest") {
      std::vector<std::string> attributeNames = {"mode", "nearest_mode"};
      if (!isNoneValue(resizeOp.getScales())) {
        rewriter.create<KrnlCallOp>(
            loc, "Resize_Scales", alloc, op, operands, attributeNames);
      } else {
        rewriter.create<KrnlCallOp>(
            loc, "Resize_Size", alloc, op, operands, attributeNames);
      }
      rewriter.replaceOp(op, alloc);
      onnxToKrnlSimdReport(op);
      return success();
    }
    // It is much more efficient to generate codes directly if possible

    SmallVector<Value, 4> scaleValues;
    IndexExpr::getValues(shapeHelper.scales, scaleValues);

    // Constants used in the loop body
    Value zero = create.math.constant(rewriter.getIntegerType(64), 0);
    Value one = create.math.constantIndex(1);

    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(alloc, ubs);
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &ck, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder>
              create(ck);
          SmallVector<Value, 4> readIndices;
          for (int64_t i = 0; i < rank; ++i) {
            Value inIndexFloat;
            Value outIndex = loopInd[i];
            Value outIndexInteger = rewriter.create<arith::IndexCastOp>(
                loc, rewriter.getIntegerType(64), outIndex);
            Value outIndexFloat =
                create.math.cast(rewriter.getF32Type(), outIndexInteger);

            // Handle coordinate transformation
            if (resizeOp.getCoordinateTransformationMode() == "asymmetric") {
              inIndexFloat = create.math.div(outIndexFloat, scaleValues[i]);
            } else if (resizeOp.getCoordinateTransformationMode() ==
                       "half_pixel") {
              // If coordinate_transformation_mode is "half_pixel",
              // x_original = (x_resized + 0.5) / scale - 0.5,
              Value halfPixelConstant =
                  create.math.constant(rewriter.getF32Type(), 0.5);
              Value addValue =
                  create.math.add(outIndexFloat, halfPixelConstant);
              Value divValue = create.math.div(addValue, scaleValues[i]);
              inIndexFloat = create.math.sub(divValue, halfPixelConstant);
            }

            // Handle nearest_mode
            if (resizeOp.getNearestMode() == "round_prefer_floor") {
              // round_prefer_floor will round 2.5 to 2, not 3
              Value deltaConstant =
                  create.math.constant(rewriter.getF32Type(), 0.499999);
              inIndexFloat = create.math.add(inIndexFloat, deltaConstant);
            } else if (resizeOp.getNearestMode() == "round_prefer_ceil") {
              Value deltaConstant =
                  create.math.constant(rewriter.getF32Type(), 0.5);
              inIndexFloat = create.math.add(inIndexFloat, deltaConstant);
            } else if (resizeOp.getNearestMode() == "floor") {
              // Not supported by create.math
              inIndexFloat = rewriter.create<math::FloorOp>(loc, inIndexFloat);
            } else if (resizeOp.getNearestMode() == "ceil") {
              // Not supported by create.math
              inIndexFloat = rewriter.create<math::CeilOp>(loc, inIndexFloat);
            } else {
              llvm_unreachable("Unexpected getNearestMode() for ResizeOp");
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
            Value inputDim = create.krnlIE.getShapeAsDim(data, i).getValue();

            Value lessThanDim = create.math.slt(inIndexLBPadded, inputDim);
            Value inputDimMinus = create.math.sub(inputDim, one);
            Value inIndexPadded =
                create.math.select(lessThanDim, inIndexLBPadded, inputDimMinus);

            readIndices.emplace_back(inIndexPadded);
          }
          Value loadVal = create.krnl.load(data, readIndices);
          create.krnl.store(loadVal, alloc, loopInd);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXResizeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXResizeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
