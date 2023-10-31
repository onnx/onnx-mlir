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

#include <functional>

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

using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
    MemRefBuilder, MathBuilder, VectorBuilder, OnnxBuilder,
    AffineBuilderKrnlMem, SCFBuilder>;

// Util: gather the 3 results (possibly none values) and replace the original op
// by these results.
static inline void replaceLayerNormalizationOp(
    ConversionPatternRewriter &rewriter, ONNXLayerNormalizationOp lnOp, Value Y,
    Value meanOfX, Value invStdDev) {
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
}

// Generate the original ONNX operations. This is the unoptimized path.
// TODO: conversions of types are not handled.
LogicalResult generateONNXLayerNormalizationOpONNXCode(
    ConversionPatternRewriter &rewriter, Location loc,
    ONNXLayerNormalizationOp lnOp) {
  MDBuilder create(rewriter, loc);
  Value X = lnOp.getX(); // Original value, not translated.
  TensorType XType = X.getType().cast<TensorType>();
  Type elementType = XType.getElementType();
  int64_t XRank = XType.getRank();
  int64_t axis = getAxisInRange(lnOp.getAxis(), XRank);
  // Get epsilon
  FloatAttr epsilonAttr = lnOp.getEpsilonAttr();
  DenseElementsAttr epsilonDenseAttr =
      onnx_mlir::createDenseElementsAttrFromFloatAttr(
          rewriter, elementType, epsilonAttr);
  Value epsilon = create.onnx.constant(epsilonDenseAttr);

  // Create reduction axes array.
  llvm::SmallVector<int64_t, 4> axesIntArray, reductionShape;
  for (int64_t r = 0; r < axis; ++r)
    reductionShape.emplace_back(XType.getShape()[r]);
  for (int64_t r = axis; r < XRank; ++r) {
    reductionShape.emplace_back(1);
    axesIntArray.emplace_back(r);
  }
  Value axes =
      create.onnx.constant(create.getBuilder().getI64TensorAttr(axesIntArray));
  TensorType reductionType = RankedTensorType::get(reductionShape, elementType);
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
  replaceLayerNormalizationOp(rewriter, lnOp, Y, meanOfX, invStdDev);
  return success();
}

struct ONNXLayerNormalizationOpLowering
    : public OpConversionPattern<ONNXLayerNormalizationOp> {
  ONNXLayerNormalizationOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD)
      : OpConversionPattern(typeConverter, ctx), enableSIMD(enableSIMD) {}

  bool enableSIMD;

  bool isBroadcastCompatible(ONNXLayerNormalizationOpShapeHelper &shapeHelper,
      Value operand, int64_t operandIndex, int64_t axis,
      int64_t innermostLoopCollapse, int64_t XRank) const {
    int64_t rank = operand.getType().cast<MemRefType>().getRank();
    // If we have a scalar, broadcast is always ok.
    if (rank <= 1) {
      return true;
    }
    // When ranks are different, then unsuitable broadcast will happen.
    if (rank != innermostLoopCollapse) {
      LLVM_DEBUG(llvm::dbgs() << "  operand: bad broadcast, fail SIMD\n");
      return false;
    }
    // Check that the dimensions are ok.
    for (int64_t i = axis; i < XRank; ++i) {
      // Should use Dim Analysis, constant dims for now.
      if (!shapeHelper.inputsDims[operandIndex][i].isLiteralAndIdenticalTo(
              shapeHelper.getOutputDims(0)[i])) {
        LLVM_DEBUG(
            llvm::dbgs() << "  operand: dynamic or different dim, fail SIMD\n");
        return false;
      }
    }
    // All good.
    return true;
  }

  bool isSimdizable(MDBuilder &create, ONNXLayerNormalizationOp lnOp,
      ONNXLayerNormalizationOpAdaptor adaptor,
      ONNXLayerNormalizationOpShapeHelper &shapeHelper, int64_t &VL) const {
    VL = 0;
    Operation *op = lnOp.getOperation();
    if (!enableSIMD) {
      onnxToKrnlSimdReport(
          op, /*successful*/ false, 0, 0, "no simd because disabled");
      return false;
    }

    // Get info.
    Value X = adaptor.getX();
    MemRefType XMemRefType = X.getType().cast<MemRefType>();
    DimsExpr XDims = shapeHelper.inputsDims[0];
    int64_t XRank = XMemRefType.getRank();
    int64_t axis = getAxisInRange(lnOp.getAxis(), XRank);
    int64_t innermostLoopCollapse = XRank - axis;
    // Detect if we can use SIMD based on inout/X output/Y shape.
    VectorMachineSupport *vms =
        VectorMachineSupport::getGlobalVectorMachineSupport();

    // Implementation relies into splitting the input X into a 2D vector, with
    // outer dim is batches, and inner dims is where the mean/stddev is
    // performed. This approach could be extended with some work to handle cases
    // where there is no batches at all (so everything is part of mean/std dev
    // computation); this is not supported at this time. Most cases seen are
    // just the last dim for mean/std dev.
    if (axis == 0) {
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, 0,
          "no simd because cannot handle case with axis=0");
      return false;
    }

    // Do not want to disable SIMD for lack of sum across support at this stage.
    // Type elementType = XMemRefType.getElementType();
    // if (vms->getVectorLength(GenericOps::SumAcrossGop, elementType) <= 0) {
    //   LLVM_DEBUG(llvm::dbgs() << "  SIMD: unsupported sum across, fail\n");
    //   return false;
    // }

    int64_t simdLoopStaticTripCount;
    VL = create.vec.computeSuitableUnrollFactor(vms, XMemRefType, XDims,
        innermostLoopCollapse, 4, /*canPad*/ false, simdLoopStaticTripCount);
    LLVM_DEBUG(llvm::dbgs()
                   << "  SIMD: LayerNormalization " << simdLoopStaticTripCount
                   << " loops, VL " << VL << "\n";);
    if (VL == 0) {
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, simdLoopStaticTripCount,
          "no simd because could not find beneficial VL");
      return false;
    }

    // Now if we have SIMD, check scale and Bias have compatible broadcast with
    // X/Y.
    if (!isBroadcastCompatible(shapeHelper, adaptor.getScale(),
            /*scale index*/ 1, axis, innermostLoopCollapse, XRank)) {
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, simdLoopStaticTripCount,
          "no simd because scale is not broadcast compatible");
      return false;
    }
    if (!isNoneValue(adaptor.getB()) &&
        !isBroadcastCompatible(shapeHelper, adaptor.getB(), /*bias index*/ 2,
            axis, innermostLoopCollapse, XRank)) {
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, simdLoopStaticTripCount,
          "no simd because bias is not broadcast compatible");
      return false;
    }
    onnxToKrnlSimdReport(
        op, /*successful*/ true, VL, simdLoopStaticTripCount, "successful");

    return true;
  }

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

    int64_t VL;
    if (isSimdizable(create, lnOp, adaptor, shapeHelper, VL)) {
      return generateSIMDCode(rewriter, loc, lnOp, adaptor, shapeHelper, 4, VL);
    }
    return generateONNXLayerNormalizationOpONNXCode(rewriter, loc, lnOp);
  }

  using F1 = std::function<void(int64_t offsetInt, Value offsetVal)>;

  void inlineFor(MDBuilder &create, int64_t B, F1 genCode) const {
    for (int64_t offsetInt = 0; offsetInt < B; ++offsetInt) {
      Value offsetVal = create.math.constantIndex(offsetInt);
      genCode(offsetInt, offsetVal);
    }
  }

  void convertAlignAllocAndFlatten(MDBuilder &create, Value inputVal,
      DimsExpr &inputDims, int64_t axis, /*output*/ Value &memRef,
      /*output*/ Value &flatMemRef) const {
    // Convert input.
    Type convertedType = typeConverter->convertType(inputVal.getType());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();
    // Allocate.
    memRef = create.mem.alignedAlloc(memRefType, inputDims);
    // Flatten (do not keep flatten dims at this time).
    DimsExpr flatDims;
    flatMemRef = create.mem.reshapeToFlat2D(memRef, inputDims, flatDims, axis);
  }

  // Limitation: scale and bias are either a scalar (to be splatted) or has the
  // same size as the flattened inner dim of X. That info is known at compile
  // time.
  void generateIterWithSIMD(ConversionPatternRewriter &rewriter,
      MDBuilder &create, ONNXLayerNormalizationOp lnOp,
      /* flat inputs */ Value XMemRef, Value scaleMemRef, Value BMemRef,
      /* flat outputs */ Value YMemRef, Value meanMemRef, Value invStdDevMemRef,
      /* temps [B][vec] */ Value redMemRef, Value redMemRef2,
      /* index expr param */ IndexExpr redDim,
      /* value params */ Value i, Value epsilon,
      /* int params */ int64_t B, int64_t VL, bool hasScalarScale,
      bool hasScalarB) const {
    // Vector type.
    Type elementType = YMemRef.getType().cast<ShapedType>().getElementType();
    VectorType vecType = VectorType::get({VL}, elementType);
    // Init the two reductions.
    Value init = create.math.constant(elementType, 0.0);
    Value initVec = create.vec.splat(vecType, init);
    Value zero = create.math.constantIndex(0);
    inlineFor(create, B, [&](int64_t d, Value o) {
      create.vec.store(initVec, redMemRef, {o, zero});
      create.vec.store(initVec, redMemRef2, {o, zero});
    });
    // Perform reduction of entire vectors.
    IndexExpr izero = LiteralIndexExpr(0);
    create.affineKMem.forIE(izero, redDim, VL,
        [&](onnx_mlir::AffineBuilderKrnlMem &ck, mlir::Value j) {
          MDBuilder create(ck);
          // load X, compute X**2, sum into reductions.
          inlineFor(create, B, [&](int64_t d, Value o) {
            Value ii = create.math.add(i, o);
            // Load X, compute X2.
            Value currX = create.vec.load(vecType, XMemRef, {ii, j});
            Value currXSquare = create.math.mul(currX, currX);
            // Load reductions.
            Value currRed = create.vec.load(vecType, redMemRef, {o, zero});
            Value currRed2 = create.vec.load(vecType, redMemRef2, {o, zero});
            // perform reductions.
            Value newRed = create.math.add(currRed, currX);
            Value newRed2 = create.math.add(currRed2, currXSquare);
            // Store reductions.
            create.vec.store(newRed, redMemRef, {o, zero});
            create.vec.store(newRed2, redMemRef2, {o, zero});
          });
        });

    // Sum across, compute mean, var, standard deviation and its inverse.
    llvm::SmallVector<Value, 4> mean(B), invStdDev(B);
    Value redDimFloat = create.math.cast(elementType, redDim.getValue());
    Value oneFloat = create.math.constant(elementType, 1.0);
    inlineFor(create, B, [&](int64_t d, Value o) {
      // Load reductions.
      Value finalRed = create.vec.load(vecType, redMemRef, {o, zero});
      Value finalRed2 = create.vec.load(vecType, redMemRef2, {o, zero});
      // Horizontal reductions.
      Value currSum =
          create.vec.reduction(VectorBuilder::CombiningKind::ADD, finalRed);
      Value currSum2 =
          create.vec.reduction(VectorBuilder::CombiningKind::ADD, finalRed2);
      // Compute means.
      mean[d] = create.math.div(currSum, redDimFloat);
      Value mean2 = create.math.div(currSum2, redDimFloat);
      // Compute standard deviation (with epsilon) and its inverse.
      Value meanSquare = create.math.mul(mean[d], mean[d]);
      Value var = create.math.sub(mean2, meanSquare);
      Value varEps = create.math.add(var, epsilon);
      Value stdDev = create.math.sqrt(varEps);
      invStdDev[d] = create.math.div(oneFloat, stdDev);
    });
    // Normalize of entire vectors.
    create.affineKMem.forIE(izero, redDim, VL,
        [&](onnx_mlir::AffineBuilderKrnlMem &ck, mlir::Value j) {
          MDBuilder create(ck);
          // load X, compute X**2, sum into reductions.
          inlineFor(create, B, [&](int64_t d, Value o) {
            Value ii = create.math.add(i, o);
            // Load X, compute X2.
            Value currX = create.vec.load(vecType, XMemRef, {ii, j});
            Value meanSplat = create.vec.splat(vecType, mean[d]);
            Value XMinusMean = create.math.sub(currX, meanSplat);
            Value invStdDevSplat = create.vec.splat(vecType, invStdDev[d]);
            Value normalizedX = create.math.mul(XMinusMean, invStdDevSplat);
            // Process with multiplying by scale (scalar or 1D vector).
            Value scale;
            if (hasScalarScale) {
              Value scalar = create.krnl.load(scaleMemRef, {zero});
              scale = create.vec.splat(vecType, scalar);
            } else {
              scale = create.vec.load(vecType, scaleMemRef, {j});
            }
            Value Y = create.math.mul(normalizedX, scale);
            // Process with adding bias (scalar or 1D vector), if provided.
            if (BMemRef && !isNoneValue(BMemRef)) {
              Value bias;
              if (hasScalarB) {
                Value scalar = create.krnl.load(BMemRef, {zero});
                bias = create.vec.splat(vecType, scalar);
              } else {
                bias = create.vec.load(vecType, BMemRef, {j});
              }
              Y = create.math.add(Y, bias);
            }
            create.vec.store(Y, YMemRef, {ii, j});
          });
        });
    // save mean and std dev if requested.
    if (meanMemRef) {
      inlineFor(create, B, [&](int64_t d, Value o) {
        Value ii = create.math.add(i, o);
        create.krnl.store(mean[d], meanMemRef, {ii, zero});
      });
    }
    if (invStdDevMemRef) {
      inlineFor(create, B, [&](int64_t d, Value o) {
        Value ii = create.math.add(i, o);
        create.krnl.store(invStdDev[d], invStdDevMemRef, {ii, zero});
      });
    }
  }

  LogicalResult generateSIMDCode(ConversionPatternRewriter &rewriter,
      Location loc, ONNXLayerNormalizationOp lnOp,
      ONNXLayerNormalizationOpAdaptor &adaptor,
      ONNXLayerNormalizationOpShapeHelper &shapeHelper, int64_t B,
      int64_t VL) const {
    MDBuilder create(rewriter, loc);
    Value XMemRef = adaptor.getX();
    MemRefType XMemRefType = XMemRef.getType().cast<MemRefType>();
    Type elementType = XMemRefType.getElementType();
    int64_t XRank = XMemRefType.getRank();
    int64_t axis = getAxisInRange(lnOp.getAxis(), XRank);
    // Get epsilon as a scalar.
    Value epsilon =
        create.math.constant(elementType, lnOp.getEpsilon().convertToDouble());

    // Flatten X input.
    Value XFlatMemRef, scaleFlatMemRef, BFlatMemRef;
    DimsExpr XFlatDims, flatDims;
    XFlatMemRef = create.mem.reshapeToFlat2D(
        XMemRef, shapeHelper.inputsDims[0], XFlatDims, axis);

    // Fully latten scale input.
    int64_t scaleRank =
        adaptor.getScale().getType().cast<MemRefType>().getRank();
    DimsExpr scaleDims;
    for (int64_t i = XRank - scaleRank; i < XRank; ++i)
      scaleDims.emplace_back(shapeHelper.inputsDims[1][i]);
    scaleFlatMemRef = create.mem.reshapeToFlatInnermost(
        adaptor.getScale(), scaleDims, flatDims, scaleRank);

    bool hasScalarScale = flatDims[0].isLiteralAndIdenticalTo(1);
    // Fully latten bias input, if present.
    bool hasScalarB = false;
    if (!isNoneValue(lnOp.getB())) {
      int64_t BRank = adaptor.getB().getType().cast<MemRefType>().getRank();
      DimsExpr BDims;
      for (int64_t i = XRank - BRank; i < XRank; ++i)
        BDims.emplace_back(shapeHelper.inputsDims[2][i]);
      BFlatMemRef = create.mem.reshapeToFlatInnermost(
          adaptor.getB(), BDims, flatDims, BRank);
      hasScalarB = flatDims[0].isLiteralAndIdenticalTo(1);
    }
    // Convert outputs, alloc data, and flatten them too.
    Value YMemRef, meanMemRef, invStdDevMemRef;
    Value YFlatMemRef, meanFlatMemRef, invStdDevFlatMemRef;
    convertAlignAllocAndFlatten(create, lnOp.getY(),
        shapeHelper.getOutputDims(0), axis, YMemRef, YFlatMemRef);
    if (!isNoneValue(lnOp.getMean()))
      convertAlignAllocAndFlatten(create, lnOp.getMean(),
          shapeHelper.getOutputDims(1), axis, meanMemRef, meanFlatMemRef);
    if (!isNoneValue(lnOp.getInvStdDev()))
      convertAlignAllocAndFlatten(create, lnOp.getInvStdDev(),
          shapeHelper.getOutputDims(2), axis, invStdDevMemRef,
          invStdDevFlatMemRef);
    // Alloc mem for reductions (should be private if parallel)
    MemRefType tmpRedType = MemRefType::get({B, VL}, elementType);
    Value tmpRedMemRef = create.mem.alignedAlloca(tmpRedType);
    Value tmpRedMemRef2 = create.mem.alignedAlloca(tmpRedType);
    // Iterate over 1st dim by block
    ValueRange loopDefs = create.krnl.defineLoops(1);
    IndexExpr zero = LiteralIndexExpr(0);
    ValueRange blockedLoopDefs = create.krnl.block(loopDefs[0], B);
    create.krnl.iterateIE({loopDefs[0]}, {blockedLoopDefs[0]}, {zero},
        {XFlatDims[0]}, [&](KrnlBuilder &ck, ValueRange blockedLoopIndices) {
          MDBuilder create(ck);
          IndexExprScope innerScope(ck);
          IndexExpr blockedCurrIndex = DimIndexExpr(blockedLoopIndices[0]);
          IndexExpr blockedUB =
              SymbolIndexExpr(XFlatDims[0].getValue()); // hi alex, take value?
          IndexExpr isFull = create.krnlIE.isTileFull(
              blockedCurrIndex, LiteralIndexExpr(B), blockedUB);
          Value zero = create.math.constantIndex(0);
          Value isNotFullVal = create.math.slt(isFull.getValue(), zero);
          create.scf.ifThenElse(
              isNotFullVal,
              [&](SCFBuilder &scf) {
                MDBuilder create(scf);
                // create.krnl.printf("partial tile\n");
                Value startOfLastBlockVal = blockedCurrIndex.getValue();
                Value blockedUBVal = blockedUB.getValue();
                create.scf.forLoop(startOfLastBlockVal, blockedUBVal, 1,
                    [&](SCFBuilder &scf, Value blockLocalInd) {
                      MDBuilder create(scf);
                      generateIterWithSIMD(rewriter, create, lnOp, XFlatMemRef,
                          scaleFlatMemRef, BFlatMemRef, YFlatMemRef,
                          meanFlatMemRef, invStdDevFlatMemRef, tmpRedMemRef,
                          tmpRedMemRef2, XFlatDims[1], blockLocalInd, epsilon,
                          1, VL, hasScalarScale, hasScalarB);
                    }); /* for inside blocked loop */
              },
              [&](SCFBuilder &scf) {
                MDBuilder create(scf);
                // create.krnl.printf("full tile\n");
                generateIterWithSIMD(rewriter, create, lnOp, XFlatMemRef,
                    scaleFlatMemRef, BFlatMemRef, YFlatMemRef, meanFlatMemRef,
                    invStdDevFlatMemRef, tmpRedMemRef, tmpRedMemRef2,
                    XFlatDims[1], blockedLoopIndices[0], epsilon, B, VL,
                    hasScalarScale, hasScalarB);
              });
        }); /* blocked loop */
    // We don't seem to need to reshape fhe flattened memref, just pass the
    // original ones.
    replaceLayerNormalizationOp(
        rewriter, lnOp, YMemRef, meanMemRef, invStdDevMemRef);
    return success();
  }
};

// clang-format off
/*
 *  Explanation on the SIMD code.

 * Algorithm.

  There are 2 ways to compute var[x]. The first is the method listed in the op
  definition:

  E[ (x -E[x]) ^2]  where E[x] corresponds to the mean of x.

  The second method is:

  E[x]^2 - E[x^2].

  As shown in wikipedia (https://en.wikipedia.org/wiki/Variance)

   E[ (x -E[x]) ^2]) = E[ x^2 -2xE[x] + E[x]^2 ] =
   E[x^2] -2E[x]E[x] + E[x]^2 = E[x^2] - E[x]^2

  This second method is faster as we can compute both E[x] and E[x^2] at the
  same time.

 * Python code that explains how the code was derived.

def layer_norm_simd2_v3(x, a, scale, b):
    (b1, b2) = (2, 4)
    t_rank = len(x.shape)
    assert t_rank==2, "work for rank 2 only"
    (a1, s2)  = x.shape
    mean = np.zeros((a1, 1))
    inv_std_dev = np.zeros((a1, 1))
    y = np.zeros((a1, s2))
    for i1 in range(0, a1, b1):
        # iterate over blocks of b1 values
        
        # MEAN(x), MEAN(x2)
        # iterate over a_block, s_block: parallel add
        r = np.zeros((b1, b2))
        r_square = np.zeros((b1, b2))
        for i2 in range(0, s2, b2): # Unroll B1, SIMD by B2, 
            xx = x[i1:i1+b1, i2:i2+b2]
            xxx = np.multiply(xx, xx)
            r = np.add(r, xx)
            r_square = np.add(r_square, xxx)
            
        # simd reduction; res B1 x 1
        # 2 B1 div
        mean_b = np.sum(r, axis=1, keepdims=True) # SIMD reduction to (B1x1) values.
        mean_b = np.divide(mean_b, s2) # (B2x1) values... so scalar is ok. 
        mean_square_b = np.sum(r_square, axis=1, keepdims=True) # Same.
        mean_square_b = np.divide(mean_square_b, s2) 

        # var = mean_square - mean**2; all compute here are on (B1x1): B1 mul, B1 add
        mean2_b = np.multiply(mean_b, mean_b) # B1 values, ok to do scalar
        var_b = np.subtract(mean_square_b, mean2_b)
        
        # ADD eps, sqrt, inverse
        # computations on B1x1, scalar ok: B1 add, B1 sqrt, B1 div
        var_eps_b = np.add(var_b, 1e-05)
        std_dev_b = np.sqrt(var_eps_b)
        inv_std_dev_b = np.divide(1.0, std_dev_b)

        # tot ops up to here (on B1x1): div: 3*B1, sqrt: B1, mul B1, add/sub 2 B1, sqrt B1: tot 8 B1

        # SIMD on entire S2 size
        for i2 in range(0, s2, b2): # Unroll B1, SIMD by B2, 
            x_b = x[i1:i1+b1, i2:i2+b2]
            d_b = np.subtract(x_b, mean_b) # broadcast of mean_b of 1 -> s2
            normalized_b = np.multiply(d_b, inv_std_dev_b)  # broadcast of mean_b of 1 -> s2
            normalized_scaled_b = np.multiply(normalized_b, scale) # assume here scale is scalar
            y_b = np.add(normalized_scaled_b, b) # assume here b is scalar

            # save values
            #mean[i1:i1+b1, i2:i2+b2] = mean_b
            #inv_std_dev[i1:i1+b1, i2:i2+b2] = inv_std_dev_b
            y[i1:i1+b1, i2:i2+b2] = y_b
        mean[i1:i1+b1, :] = mean_b
        inv_std_dev[i1:i1+b1, :] = inv_std_dev_b

    return (y, mean, inv_std_dev)
*/
// clang-format on

void populateLoweringONNXNormalizationOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD) {
  patterns.insert<ONNXBatchNormalizationInferenceModeOpLowering>(
      typeConverter, ctx);
  patterns.insert<ONNXInstanceNormalizationOpLowering>(typeConverter, ctx);
  patterns.insert<ONNXLayerNormalizationOpLowering>(
      typeConverter, ctx, enableSIMD);
}

} // namespace onnx_mlir
