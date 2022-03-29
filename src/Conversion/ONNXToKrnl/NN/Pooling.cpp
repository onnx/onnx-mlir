/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Pooling.cpp - Lowering Pooling Ops ------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Pooling Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

// Identity values
template <>
Value getIdentityValue<ONNXMaxPoolSingleOutOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  return create.math.negativeInf(type);
}

template <>
Value getIdentityValue<ONNXAveragePoolOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return emitConstantOp(rewriter, loc, type, 0);
}

// Scalar operations
template <>
struct ScalarOp<ONNXAveragePoolOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

template <>
Value emitScalarOpFor<ONNXMaxPoolSingleOutOp>(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    Type elementType, ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value max = create.math.sgt(lhs, rhs);
  return create.math.select(max, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Get dilation values
//
template <typename PoolOp>
std::vector<int64_t> getDilations(PoolOp poolOp) {
  return {};
}

// MaxPool has dilations attribute.
template <>
std::vector<int64_t> getDilations<ONNXMaxPoolSingleOutOp>(
    ONNXMaxPoolSingleOutOp poolOp) {
  std::vector<int64_t> dilations;
  auto dilationsAttribute = poolOp.dilationsAttr();
  bool isDefaultDilations = true;
  for (auto dilation : dilationsAttribute.getValue()) {
    int64_t dilationValue = dilation.cast<IntegerAttr>().getInt();
    if (dilationValue > 1 && isDefaultDilations)
      isDefaultDilations = false;
    dilations.emplace_back(dilationValue);
  }
  if (isDefaultDilations)
    return {};
  else
    return dilations;
}

//===----------------------------------------------------------------------===//
// Get dilation attribute.
//
template <typename PoolOp>
llvm::Optional<ArrayAttr> getDilationAttr(PoolOp poolOp) {
  return llvm::None;
}

// MaxPool has dilations attribute.
template <>
llvm::Optional<ArrayAttr> getDilationAttr<ONNXMaxPoolSingleOutOp>(
    ONNXMaxPoolSingleOutOp poolOp) {
  return poolOp.dilations();
}

//===----------------------------------------------------------------------===//
// Get count_include_pad values
//
template <typename PoolOp>
bool getCountIncludePad(PoolOp poolOp) {
  return false;
}

// AveragePool has count_include_pad attribute.
template <>
bool getCountIncludePad<ONNXAveragePoolOp>(ONNXAveragePoolOp poolOp) {
  return (poolOp.count_include_pad() == 1);
}

//===----------------------------------------------------------------------===//
// Helper function to do post-processing after applying a filter window.
//
template <typename PoolOp>
void postProcessPoolingWindow(ConversionPatternRewriter &rewriter, Location loc,
    PoolOp poolOp, Value alloc, ArrayRef<Value> resultIndices,
    ArrayRef<int64_t> kernelShape, ArrayRef<Value> poolDimValues) {}

// Calculate the average value for AveragePool.
template <>
void postProcessPoolingWindow<ONNXAveragePoolOp>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXAveragePoolOp poolOp,
    Value alloc, ArrayRef<Value> resultIndices, ArrayRef<int64_t> kernelShape,
    ArrayRef<Value> poolDimValues) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);

  // AveragePool's result type is FloatType, so it's safe to use DivFOp, SubFOp.
  bool countIncludePad = getCountIncludePad<ONNXAveragePoolOp>(poolOp);
  Value numerator = create.krnl.load(alloc, resultIndices);
  Value denominator;
  if (countIncludePad) {
    int64_t kernelSize = 1;
    for (unsigned int i = 0; i < kernelShape.size(); ++i)
      kernelSize *= kernelShape[i];
    denominator =
        emitConstantOp(rewriter, loc, numerator.getType(), kernelSize);
  } else {
    denominator = poolDimValues[0];
    for (unsigned int i = 1; i < poolDimValues.size(); ++i)
      denominator = create.math.mul(denominator, poolDimValues[i]);
    denominator = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIntegerType(64), denominator);
    denominator =
        rewriter.create<arith::SIToFPOp>(loc, numerator.getType(), denominator);
  }

  Value average = create.math.div(numerator, denominator);
  create.krnl.store(average, alloc, resultIndices);
}

//===----------------------------------------------------------------------===//
// Helper function to do post-processing after applying a filter window.
//
template <typename PoolOp>
void postProcessPoolingWindow(ConversionPatternRewriter &rewriter, Location loc,
    PoolOp poolOp, Value alloc, ArrayRef<Value> resultIndices,
    ArrayRef<IndexExpr> kernelShape, ArrayRef<Value> poolDimValues) {}

// Calculate the average value for AveragePool.
template <>
void postProcessPoolingWindow<ONNXAveragePoolOp>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXAveragePoolOp poolOp,
    Value alloc, ArrayRef<Value> resultIndices, ArrayRef<IndexExpr> kernelShape,
    ArrayRef<Value> poolDimValues) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);

  // AveragePool's result type is FloatType, so it's safe to use DivFOp, SubFOp.
  bool countIncludePad = getCountIncludePad<ONNXAveragePoolOp>(poolOp);
  Value numerator = create.krnl.load(alloc, resultIndices);
  Value denominator;
  if (countIncludePad) {
    IndexExpr kernelSize = LiteralIndexExpr(1);
    for (unsigned int i = 0; i < kernelShape.size(); ++i)
      kernelSize = kernelSize * kernelShape[i];
    denominator = kernelSize.getValue();
    denominator = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), denominator);
    // TODO: we are implying here that the dest type is a float.
    denominator =
        rewriter.create<arith::SIToFPOp>(loc, numerator.getType(), denominator);
  } else {
    denominator = poolDimValues[0];
    for (unsigned int i = 1; i < poolDimValues.size(); ++i)
      denominator = create.math.mul(denominator, poolDimValues[i]);
    denominator = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIntegerType(64), denominator);
    // TODO: we are implying here that the dest type is a float.
    denominator =
        rewriter.create<arith::SIToFPOp>(loc, numerator.getType(), denominator);
  }

  Value average = create.math.div(numerator, denominator);
  create.krnl.store(average, alloc, resultIndices);
}

//===----------------------------------------------------------------------===//
// Template function that does pooling.
//
template <typename PoolOp, typename PoolOpAdaptor, typename PoolOpShapeHelper>
struct ONNXPoolOpLowering : public ConversionPattern {
  ONNXPoolOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, PoolOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    PoolOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    PoolOp poolOp = llvm::dyn_cast<PoolOp>(op);
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);

    // Get shape.
    PoolOpShapeHelper shapeHelper(&poolOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed));

    // Read ceil_mode attribute
    auto ceilMode = poolOp.ceil_mode();

    // Type information about the input and result of this operation.
    Value inputOperand = operandAdaptor.X();
    auto inputShape = inputOperand.getType().cast<MemRefType>().getShape();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto outputShape = memRefType.getShape();
    auto outputElementType = memRefType.getElementType();

    // Kernel offset in the input shape.
    int kernelShapeSize = shapeHelper.kernelShape.size();
    int kernelOffset = inputShape.size() - kernelShapeSize;
    bool isDilated = false;
    for (int i = 0; i < kernelShapeSize; ++i)
      if (shapeHelper.dilations[i] > 1)
        isDilated = true;

    // Scope for krnl ops
    IndexExprScope ieScope(&rewriter, loc);

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));

    // input = Pool(output)
    //
    // The input/output shapes will look like this:
    //
    // input (NxCxHxW) -> output (NxCxHOxWO)
    //
    // The loop nest will look as follows:
    //
    // kernelShape = [kH, kW]
    // pads = [ptH, ptW, pbH, pbW]
    // strides = [sH, sW]
    // dilations = [dH, dW]
    // round = ceil if ceilMode else floor
    //
    // for n in range(N):
    //   for c in range(C):
    //     for ho in range(HO):
    //       for wo in range(WO):
    //         # Initialize values for the output.
    //         output[n][c][ho][wo] = getIdentityValue(...);
    //
    //         # Thanks to Tian (@tjingrant) for the following derivation about
    //         # firstValid.
    //         # When dilation is non-unit, the first valid pixel to
    //         # apply pooling on will not be the 0-th pixel, but rather
    //         # the smallest integer n to make -pH + n * 3 greater than
    //         # or equal to 0.
    //         # We derive what is this smallest n:
    //         # -pH + n * dH >= 0
    //         #       n * dH >= pH
    //         #            n >= pH/dH
    //         # thus n = ceil(pH/dH)
    //         # thus the first valid pixel location is
    //         # ceil(pH / dilation) * dilation - pH
    //
    //         firstValidH = ceil(float(ptH / dH)) * dH - ptH
    //         startH = max(firstValidH, ho * sH - ptH)
    //         endH = min(H, ho * sH + (kH -1) * dH  + 1 - ptH)
    //
    //         firstValidW= ceil(float(pW / dW)) * dW - ptW
    //         startW = max(firstValidW, wo * sW - ptW)
    //         endW = min(W, wo * sW + (kW - 1) * dW + 1 - ptW)
    //
    //         # Apply the pooling window.
    //         # The pooling window can be smaller than the kernel when slicing
    //         # over the border edges.
    //         for hi in range(startH, endH, dH):
    //           for wi in range(startW, endW, dW):
    //             output[n, c, ho, wo] = emitScalarOpFor(output[n, c, ho, wo],
    //                                                    input[n, c, hi, wi]);
    //
    //         # The above two for-loops are rewritten as follows:
    //         # (since KrnlIterateOp has not supported `step` yet)
    //         for hp in range(endH - startH):
    //           for wp in range(endW - startW):
    //             hi = hp * dH + startH
    //             wi = wp * dW + startW
    //             output[n, c, ho, wo] = emitScalarOpFor(output[n, c, ho, wo],
    //                                                    input[n, c, hi, wi]);
    //
    //         # Do post processing such as taking average pooling:
    //         postProcessPoolingWindow(...)
    //
    // Helper functions:
    //   getIdentityValue(): to return the indentity value
    //     - negative infinity for MaxPool
    //     - 0 for AveragePool
    //   emitScalarOpFor(): to do primitive computation for Pooling, e.g.
    //     - compute max for MaxPool
    //     - compute sum for AveragePool
    //   postProcessPoolingWindow(): to do post processing over the whole
    //   pooling window, e.g.
    //     - do nothing in case of MaxPool
    //     - calculate the average in case of AveragePool, e.g.
    //         if hDim * wDim> 0:
    //           output[n, c, ho, wo] = output[n, c, ho, wo] / (hDim*wDim)
    //

    // Identity value of the operation.
    auto identity = getIdentityValue<PoolOp>(rewriter, loc, outputElementType);

    // 1. Define output loops to compute one output pixel.
    // for n in range(N):
    //   for c in range(C):
    //     for ho in range(HO):
    //       for wo in range(WO):
    BuildKrnlLoop outputLoops(rewriter, loc, outputShape.size());
    outputLoops.createDefineAndIterateOp(alloc);

    auto ipMainRegion = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());
    {
      // 2. Emit the body of the output loop nest, which applies a pooling
      // window to a region in the input, producing one output pixel.
      SmallVector<IndexExpr, 4> outputIndices;
      for (unsigned int i = 0; i < outputShape.size(); ++i)
        outputIndices.emplace_back(
            DimIndexExpr(outputLoops.getInductionVar(i)));

      // 2.1 Emit: output[n][c][ho][wo] = identity
      // Create a local reduction value for output[n][c][ho][wo].
      // Single scalar, no need for default alignment.
      Value reductionVal =
          create.mem.alloca(MemRefType::get({}, memRefType.getElementType()));
      create.krnl.store(identity, reductionVal);

      // 2.2 Emit affine maps which express the lower and upper bounds for the
      // pooling window's dimensions.
      // The pooling window can be smaller than the kernel when slicing it over
      // the border edges. Thus, we will compute the start and end indices for
      // each dimension as follows.
      //   firstValidH = ceil(float(ptH / dH)) * dH - ptH
      //   startH = max(firstValidH, ho * sH - ptH)
      //   endH = min(H, ho * sH + (kH - 1) * dH  + 1 - pbH)
      //   hDim = round(float(endH - startH) / float(dH))

      // Prepare induction variables.
      SmallVector<SmallVector<IndexExpr, 4>, 4> IVExprs;
      {
        MemRefBoundsIndexCapture inputBounds(inputOperand);
        for (int i = 0; i < kernelShapeSize; ++i) {
          int j = i + kernelOffset;
          SmallVector<IndexExpr, 4> ic;
          // d0, output
          ic.emplace_back(outputIndices[j]);
          // s0, input dim
          ic.emplace_back(inputBounds.getDim(j));
          // s1, kernel dim
          ic.emplace_back(SymbolIndexExpr(shapeHelper.kernelShape[i]));
          // s2, pad dim
          ic.emplace_back(SymbolIndexExpr(shapeHelper.pads[i]));
          // s3, stride dim
          ic.emplace_back(LiteralIndexExpr(shapeHelper.strides[i]));
          // s4, dilation dim
          ic.emplace_back(LiteralIndexExpr(shapeHelper.dilations[i]));
          IVExprs.emplace_back(ic);
        }
      }

      // Compute the start and end position of the conv window.
      //   firstValidH = ceil(float(ptH / dH)) * dH - ptH
      //   startH = max(firstValidH, ho * sH - ptH)
      //   endH = min(H, ho * sH + (kH - 1) * dH  + 1 - pbH)
      SmallVector<IndexExpr, 4> windowStartExprs, windowEndExprs;
      for (int i = 0; i < kernelShapeSize; ++i) {
        std::vector<mlir::IndexExpr> exprs =
            getIndexExprsForConvWindow(IVExprs[i], ceilMode, isDilated);
        windowStartExprs.emplace_back(exprs[0]);
        windowEndExprs.emplace_back(exprs[1]);
      }

      // Compute the size of the full conv window.
      //   hDim = round(float(endH - startH) / float(dH))
      //   wDim = round(float(endW - startW) / float(dW))
      SmallVector<Value, 4> fullWindowSize;
      for (int i = 0; i < kernelShapeSize; ++i) {
        Value dim = create.math.sub(
            windowEndExprs[i].getValue(), windowStartExprs[i].getValue());
        if (isDilated) {
          Value one = create.math.constantIndex(1);
          Value numerator = create.math.add(dim, one);
          Value denominator = IVExprs[i][5].getValue(); // dilations[i]
          dim = create.math.div(numerator, denominator);
          if (ceilMode) {
            auto remainder =
                rewriter.create<arith::RemSIOp>(loc, numerator, denominator);
            Value zero = create.math.constantIndex(0);
            Value isZero = create.math.eq(remainder, zero);
            Value dimPlusOne = create.math.add(dim, one);
            dim = create.math.select(isZero, dim, dimPlusOne);
          }
        }
        fullWindowSize.emplace_back(dim);
      }

      // 2.3 Define pooling loops.
      //  for hp in range(hDim):
      //    for wp in range(wDim):
      //      hi = hp * dH + startH
      //      wi = wp * dW + startW
      //      output[n][c][ho][wo] =
      //        emitScalarOpFor(output[n][c][ho][wo], input[n, c, hi, wi]);
      BuildKrnlLoop poolingLoops(rewriter, loc, kernelShapeSize);
      poolingLoops.createDefineOp();
      // Push bounds.
      AffineMap windowSizeMap =
          getWindowAffineMap(rewriter, ceilMode, isDilated);
      for (int i = 0; i < kernelShapeSize; ++i) {
        // Affine map's operands.
        SmallVector<Value, 4> operands;
        for (IndexExpr expr : IVExprs[i])
          operands.emplace_back(expr.getValue());
        poolingLoops.pushBounds(0, windowSizeMap, operands);
      }
      // Create a krnl iterate.
      poolingLoops.createIterateOp();

      auto ipOuterLoopRegion = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(poolingLoops.getIterateBlock());
      {
        // 2.4 Emit the body of the pooling loop nest.
        // Prepare indices to access a pixel in the input.
        SmallVector<IndexExpr, 4> inputIndices;
        { // Construct inputIndices
          for (int i = 0; i < kernelOffset; ++i)
            inputIndices.emplace_back(outputIndices[i]);
          for (int i = kernelOffset; i < (int)inputShape.size(); ++i) {
            int j = i - kernelOffset;
            DimIndexExpr hp(poolingLoops.getInductionVar(j));
            IndexExpr startH = windowStartExprs[j];
            if (isDilated) {
              // hi = hp * dH + startH
              IndexExpr dH = IVExprs[j][5];
              inputIndices.emplace_back(hp * dH + startH);
            } else {
              // hi = hp + startH
              inputIndices.emplace_back(hp + startH);
            }
          }
        }

        // Apply pooling operation.
        //      output[n][c][ho][wo] =
        //        emitScalarOpFor(output[n][c][ho][wo], input[n, c, hi, wi]);
        Value loadInput = create.krnl.loadIE(inputOperand, inputIndices);
        Value loadPartialOutput = create.krnl.load(reductionVal);
        Value output = emitScalarOpFor<PoolOp>(rewriter, loc, op,
            outputElementType, {loadPartialOutput, loadInput});
        create.krnl.store(output, reductionVal);
      }
      rewriter.restoreInsertionPoint(ipOuterLoopRegion);
      Value output = create.krnl.load(reductionVal);
      create.krnl.storeIE(output, alloc, outputIndices);

      // 2.5 Post-processing for the pooling window, e.g. taking average.
      SmallVector<Value, 4> outputIndicesInValue;
      for (IndexExpr expr : outputIndices)
        outputIndicesInValue.emplace_back(expr.getValue());
      postProcessPoolingWindow<PoolOp>(rewriter, loc, poolOp, alloc,
          outputIndicesInValue, shapeHelper.kernelShape, fullWindowSize);
    }

    // Go back to the main region.
    rewriter.restoreInsertionPoint(ipMainRegion);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXPoolingOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXPoolOpLowering<ONNXMaxPoolSingleOutOp,
      ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>>(
      typeConverter, ctx);
  patterns.insert<ONNXPoolOpLowering<ONNXAveragePoolOp,
      ONNXAveragePoolOpAdaptor, ONNXAveragePoolOpShapeHelper>>(
      typeConverter, ctx);
}
