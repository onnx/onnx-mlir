/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Normalization.cpp - Lowering Normalization Ops -----------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX Normalization Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Batch Norm
//===----------------------------------------------------------------------===//

struct ONNXBatchNormalizationInferenceModeOpLowering
    : public OpConversionPattern<ONNXBatchNormalizationInferenceModeOp> {
  ONNXBatchNormalizationInferenceModeOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(
      ONNXBatchNormalizationInferenceModeOp batchnormOp,
      ONNXBatchNormalizationInferenceModeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // batchnorm{epsilon}(x, scale, bias, mean, variance) =
    //      scale * (x - mean) / sqrt(variance + epsilon) + bias
    Operation *op = batchnormOp.getOperation();
    Location loc = ONNXLoc<ONNXBatchNormalizationInferenceModeOp>(op);

    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();

    Value epsilon = create.math.constant(
        memRefType.getElementType(), adaptor.getEpsilon().convertToDouble());
    Value operand = adaptor.getX();
    Value scale = adaptor.getScale();
    Value bias = adaptor.getB();
    Value mean = adaptor.getMean();
    Value variance = adaptor.getVar();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(operand, memRefType);

    // Operand's dimensions can be in the form of NxCxD1xD2x...xDn or N.
    // In case of N, C is assumed to be 1.
    // Shapes of scale, bias, mean and variance must be C.
    // Computation of BatchNormalization is done as if scale, bias, mean, and
    // variance are reshaped to Cx1x1x...x1.

    // rank
    int64_t rank = memRefType.getRank();

    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, rank);

    // Create a KrnlIterateOp along C dimension.
    // This will be the outer-most loop in order to re-use scale, bias,
    // mean and variance.

    SmallVector<Value, 1> loopCIVs;
    if (rank > 1) {
      // TODO use new KrnlDialectBuilder.
      krnl::KrnlIterateOperandPack cPack(rewriter, originalLoops[1]);
      addDimensionToPack(rewriter, loc, cPack, operand, 1);
      KrnlIterateOp cIterateOp = create.krnl.iterate(cPack);
      Block &cIterationBlock = cIterateOp.getBodyRegion().front();
      rewriter.setInsertionPointToStart(&cIterationBlock);
      for (auto arg : cIterationBlock.getArguments())
        loopCIVs.emplace_back(arg);
    } else
      loopCIVs.emplace_back(create.math.constantIndex(0));

    Value scaleVal = create.krnl.load(scale, loopCIVs);
    Value biasVal = create.krnl.load(bias, loopCIVs);
    Value meanVal = create.krnl.load(mean, loopCIVs);
    Value varianceVal = create.krnl.load(variance, loopCIVs);

    // Create a KrnlIterateOp along the other dimensions.
    SmallVector<int64_t, 4> axes;
    axes.emplace_back(0);
    for (int64_t i = 2; i < rank; ++i)
      axes.emplace_back(i);
    std::vector<Value> packLoops;
    for (size_t i = 0; i < axes.size(); ++i)
      packLoops.emplace_back(originalLoops[axes[i]]);

    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack pack(rewriter, packLoops);
    for (size_t i = 0; i < axes.size(); ++i)
      addDimensionToPack(rewriter, loc, pack, operand, axes[i]);

    KrnlIterateOp iterateOp = create.krnl.iterate(pack);
    Block &iterationBlock = iterateOp.getBodyRegion().front();
    rewriter.setInsertionPointToStart(&iterationBlock);

    SmallVector<Value, 4> loopIVs;
    auto args = iterationBlock.getArguments();
    if (args.size() > 1) {
      loopIVs.emplace_back(args[0]);
      loopIVs.emplace_back(loopCIVs[0]); // Insert C back.
      for (unsigned int i = 1; i < args.size(); ++i)
        loopIVs.emplace_back(args[i]);
    } else if (rank == 2) {
      loopIVs.emplace_back(args[0]);
      loopIVs.emplace_back(loopCIVs[0]); // Insert C back.
    } else
      loopIVs.emplace_back(args[0]);

    Value xVal = create.krnl.load(operand, loopIVs);
    // normalize
    Value dividend = create.math.sub(xVal, meanVal);
    Value adjustedVarianceVal = create.math.add(varianceVal, epsilon);
    Value divisor = create.math.sqrt(adjustedVarianceVal);
    Value normVal = create.math.div(dividend, divisor);
    // scale and shift
    Value scaleNormVal = create.math.mul(scaleVal, normVal);
    Value shiftScaleNormVal = create.math.add(scaleNormVal, biasVal);
    create.krnl.store(shiftScaleNormVal, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    onnxToKrnlSimdReport(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Instance Normalization
//===----------------------------------------------------------------------===//

struct ONNXInstanceNormalizationOpLowering
    : public OpConversionPattern<ONNXInstanceNormalizationOp> {
  ONNXInstanceNormalizationOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXInstanceNormalizationOp instanceOp,
      ONNXInstanceNormalizationOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // instance_normalization{epsilon}(x, scale, bias) =
    //      scale * (x - mean) / sqrt(variance + epsilon) + bias
    Operation *op = instanceOp.getOperation();
    Location loc = ONNXLoc<ONNXInstanceNormalizationOp>(op);

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();
    Type elementType = memRefType.getElementType();
    Value epsilon = create.math.constant(
        elementType, adaptor.getEpsilon().convertToDouble());
    Value inputMemRef = adaptor.getInput();
    Value scaleMemRef = adaptor.getScale();
    Value biasMemRef = adaptor.getB();

    // Insert an allocation and deallocation for the result of this operation.
    Value resMemRef = create.mem.alignedAlloc(inputMemRef, memRefType);

    // Operand's dimensions can be in the form of NxCxD1xD2x...xDn
    // Shapes of scale, bias must be C.

    // Get rank, bounds, and constructors.
    int64_t rank = memRefType.getRank();
    IndexExprScope outerScope(create.krnl);
    SmallVector<IndexExpr, 4> inputBounds;
    create.krnlIE.getShapeAsSymbols(inputMemRef, inputBounds);
    MemRefType tmpType = MemRefType::get({}, elementType);
    Value fZero = create.math.constant(elementType, 0);
    Value tmpMemRef = create.mem.alloca(tmpType);

    // Compute the number of values in a single channel: product of spatial
    // dimensions, converted to float.
    IndexExpr num = inputBounds[2];
    for (int d = 3; d < rank; ++d)
      num = num * inputBounds[d];
    // Convert num to float from Pooling postProcessPoolingWindow.
    Value meanDenom = create.math.cast(elementType, num.getValue());

    // Iterate over the batch and channels.
    LiteralIndexExpr iZero(0);
    ValueRange n_c_loopDef = create.krnl.defineLoops(2);
    create.krnl.iterateIE(n_c_loopDef, n_c_loopDef, {iZero, iZero},
        {inputBounds[0], inputBounds[1]},
        [&](KrnlBuilder &ck, ValueRange n_c_loopInd) {
          MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
              ck);
          IndexExprScope channelScope(ck);
          DimIndexExpr n(n_c_loopInd[0]), c(n_c_loopInd[1]);

          // Set bounds for iterating over values in channel.
          ValueRange spatial_loopDef = create.krnl.defineLoops(rank - 2);
          SmallVector<IndexExpr, 4> lbs(rank - 2, iZero);
          SmallVector<IndexExpr, 4> ubs;
          for (int d = 2; d < rank; ++d)
            ubs.emplace_back(SymbolIndexExpr(inputBounds[d]));

          // First compute the mean: store zero in reduction value, then sum up
          // all of the values in the channel, and divide by the number of
          // values.
          create.krnl.store(fZero, tmpMemRef, {});
          // Iterate over kernel and add values.
          ValueRange spatial2_loopDef = create.krnl.defineLoops(rank - 2);
          create.krnl.iterateIE(spatial2_loopDef, spatial2_loopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange spatial_loopInd) {
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
                SmallVector<Value, 6> inputAccessFct = {
                    n.getValue(), c.getValue()};
                for (int d = 0; d < rank - 2; ++d)
                  inputAccessFct.emplace_back(spatial_loopInd[d]);
                // tmp += input[n,c, spatial dims]
                Value oldSum = create.krnl.load(tmpMemRef, {});
                Value val = create.krnl.load(inputMemRef, inputAccessFct);
                Value newSum = create.math.add(oldSum, val);
                create.krnl.store(newSum, tmpMemRef);
              });
          Value sum = create.krnl.load(tmpMemRef);
          Value mean = create.math.div(sum, meanDenom);
          // Second, compute the standard dev: sum of (val - mean)2 / (num-1).
          create.krnl.store(fZero, tmpMemRef, {});
          // Iterate over kernel and add values.
          create.krnl.iterateIE(spatial_loopDef, spatial_loopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange spatial_loopInd) {
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
                SmallVector<Value, 6> inputAccessFct = {
                    n.getValue(), c.getValue()};
                for (int d = 0; d < rank - 2; ++d)
                  inputAccessFct.emplace_back(spatial_loopInd[d]);
                // tmp += input[n,c, spatial dims]
                Value oldSum = create.krnl.load(tmpMemRef, {});
                Value val = create.krnl.load(inputMemRef, inputAccessFct);
                val = create.math.sub(val, mean);
                val = create.math.mul(val, val);
                Value newSum = create.math.add(oldSum, val);
                create.krnl.store(newSum, tmpMemRef);
              });
          sum = create.krnl.load(tmpMemRef);
          // Variance is numerically off when divided by (num -1), but
          // passes the tests when divided by num, so keep that.
          Value variance = create.math.div(sum, meanDenom);

          // Calculate ahead the scale[c] / sqrt(var + epsilon)
          Value denom = create.math.add(variance, epsilon);
          denom = create.math.sqrt(denom);
          Value nom = create.krnl.load(scaleMemRef, {c.getValue()});
          Value factor = create.math.div(nom, denom);
          Value term = create.krnl.load(biasMemRef, {c.getValue()});

          // Iterate over all channel values and compute y = factor * (x - mean)
          // + term.
          ValueRange spatial3_loopDef = create.krnl.defineLoops(rank - 2);
          create.krnl.iterateIE(spatial3_loopDef, spatial3_loopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange spatial_loopInd) {
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
                SmallVector<Value, 6> accessFct = {n.getValue(), c.getValue()};
                for (int d = 0; d < rank - 2; ++d)
                  accessFct.emplace_back(spatial_loopInd[d]);
                // tmp += input[n,c, spatial dims]
                Value x = create.krnl.load(inputMemRef, accessFct);
                Value val = create.math.sub(x, mean);
                val = create.math.mul(factor, val);
                val = create.math.add(val, term);
                create.krnl.store(val, resMemRef, accessFct);
              });
        }); // For all batches, channels.

    rewriter.replaceOp(op, resMemRef);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Layer Normalization
//===----------------------------------------------------------------------===//

struct ONNXLayerNormalizationOpLowering
    : public OpConversionPattern<ONNXLayerNormalizationOp> {
  ONNXLayerNormalizationOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD)
      : OpConversionPattern(typeConverter, ctx), enableSIMD(enableSIMD) {}

  bool enableSIMD;
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MemRefBuilder, MathBuilder, VectorBuilder, OnnxBuilder>;

  LogicalResult matchAndRewrite(ONNXLayerNormalizationOp lnOp,
      ONNXLayerNormalizationOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Get generic info.
    Operation *op = lnOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    Location loc = ONNXLoc<ONNXLayerNormalizationOp>(op);
    // Create builder and shape helper
    MDBuilder create(rewriter, loc);
    ONNXLayerNormalizationOpShapeHelper shapeHelper(
        op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    MemRefType YMemRefType, meanMemRefType, ISDMemRefType;
    // bool computeMean = false, computeISD = false;

    // Get info.
    Value X = adaptor.getX();
    int64_t axis = lnOp.getAxis();
    MemRefType XMemRefType = X.getType().cast<MemRefType>();
    Type elementType = XMemRefType.getElementType();
    int64_t XRank = XMemRefType.getRank();
    DimsExpr XDims;
    create.krnlIE.getShapeAsSymbols(X, XDims);
    if (axis < 0)
      axis += XRank;
    int64_t innermostLoopCollapse = XRank - axis;

    // Detect if we can use SIMD
    int64_t VL = 0; // VL of 0 means no SIMD.
    int64_t estimatedSimdLoopTripCount;
    if (enableSIMD) {
      VectorMachineSupport *vms =
          VectorMachineSupport::getGlobalVectorMachineSupport();
      VL = create.vec.computeSuitableUnrollFactor(vms, XMemRefType, XDims,
          innermostLoopCollapse, 4, /*canPad*/ false,
          estimatedSimdLoopTripCount);
      LLVM_DEBUG({
        llvm::dbgs() << "  SIMD: " << innermostLoopCollapse << " loops, VL "
                     << VL << "\n";
        if (VL == 0)
          llvm::dbgs() << "  SIMD: no good VL\n";
      });
    }

    FloatAttr epsilonAttr = lnOp.getEpsilonAttr();
    DenseElementsAttr epsilonDenseAttr =
        onnx_mlir::createDenseElementsAttrFromFloatAttr(
            rewriter, elementType, epsilonAttr);
    Value epsilon = create.onnx.constant(epsilonDenseAttr);

    return generateONNXCode(rewriter, create, lnOp, epsilon, axis);
  }

  // Generate the original ONNX operations. This is the unoptimized path.
  // TODO: conversions of types are not handled.
  LogicalResult generateONNXCode(ConversionPatternRewriter &rewriter,
      MDBuilder &create, ONNXLayerNormalizationOp lnOp, Value epsilon,
      int64_t axis) const {
    Value X = lnOp.getX(); // Original value, not translated.
    TensorType XType = X.getType().cast<TensorType>();
    Type elementType = XType.getElementType();
    int64_t XRank = XType.getRank();
    // Create reduction axes array.
    llvm::SmallVector<int64_t, 4> axesIntArray, reductionShape;
    for (int64_t r = 0; r < axis; ++r)
      reductionShape.emplace_back(XType.getShape()[r]);
    for (int64_t r = axis; r < XRank; ++r) {
      reductionShape.emplace_back(1);
      axesIntArray.emplace_back(r);
    }
    Value axes = create.onnx.constant(
        create.getBuilder().getI64TensorAttr(axesIntArray));
    TensorType reductionType =
        RankedTensorType::get(reductionShape, elementType);
    // Reduction of input
    Value meanOfX = create.onnx.reduceMean(reductionType, X, axes);
    Value pow2OfMeanOfX = create.onnx.mul(meanOfX, meanOfX);
    Value XPow2 = create.onnx.mul(X, X);
    Value meanOfXPow2 = create.onnx.reduceMean(reductionType, XPow2, axes);
    Value var = create.onnx.sub(meanOfXPow2, pow2OfMeanOfX);
    Value varWithEpsilon = create.onnx.add(var, epsilon);
    Value stdDev = create.onnx.sqrt(varWithEpsilon);
    Value invStdDev = create.onnx.reciprocal(stdDev);
    Value d = create.onnx.sub(X, meanOfX);
    Value normalized = create.onnx.mul(d, invStdDev);
    Value Y = create.onnx.mul(normalized, lnOp.getScale());
    if (!isNoneValue(lnOp.getB()))
      Y = create.onnx.add(Y, lnOp.getB());
    llvm::SmallVector<Value, 3> outputs;
    outputs.emplace_back(Y);
    Value noneValue;
    if (isNoneValue(lnOp.getMean()))
      outputs.emplace_back(noneValue);
    else
      outputs.emplace_back(meanOfX);
    if (isNoneValue(lnOp.getInvStdDev()))
      outputs.emplace_back(noneValue);
    else
      outputs.emplace_back(invStdDev);
    rewriter.replaceOp(lnOp, outputs);
    return success();
  }
};

void populateLoweringONNXNormalizationOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD) {
  patterns.insert<ONNXBatchNormalizationInferenceModeOpLowering>(
      typeConverter, ctx);
  patterns.insert<ONNXInstanceNormalizationOpLowering>(typeConverter, ctx);
  patterns.insert<ONNXLayerNormalizationOpLowering>(
      typeConverter, ctx, enableSIMD);
}

} // namespace onnx_mlir
