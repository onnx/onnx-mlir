/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Normalization.cpp - Lowering Normalization Ops -----------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX Normalization Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXBatchNormalizationInferenceModeOpLowering
    : public ConversionPattern {
  ONNXBatchNormalizationInferenceModeOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXBatchNormalizationInferenceModeOp::getOperationName(), 1,
            ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // batchnorm{epsilon}(x, scale, bias, mean, variance) =
    //      scale * (x - mean) / sqrt(variance + epsilon) + bias
    ONNXBatchNormalizationInferenceModeOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto epsilonAttr = FloatAttr::get(memRefType.getElementType(),
        llvm::dyn_cast<ONNXBatchNormalizationInferenceModeOp>(op)
            .epsilon()
            .convertToFloat());
    auto epsilon = rewriter.create<arith::ConstantOp>(loc, epsilonAttr);

    auto operand = operandAdaptor.X();
    auto scale = operandAdaptor.scale();
    auto bias = operandAdaptor.B();
    auto mean = operandAdaptor.mean();
    auto variance = operandAdaptor.var();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, {operand});

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
      KrnlIterateOperandPack cPack(rewriter, originalLoops[1]);
      addDimensionToPack(rewriter, loc, cPack, operand, 1);
      auto cIterateOp = rewriter.create<KrnlIterateOp>(loc, cPack);
      Block &cIterationBlock = cIterateOp.bodyRegion().front();
      rewriter.setInsertionPointToStart(&cIterationBlock);
      for (auto arg : cIterationBlock.getArguments())
        loopCIVs.emplace_back(arg);
    } else {
      loopCIVs.emplace_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    auto scaleVal = rewriter.create<KrnlLoadOp>(loc, scale, loopCIVs);
    auto biasVal = rewriter.create<KrnlLoadOp>(loc, bias, loopCIVs);
    auto meanVal = rewriter.create<KrnlLoadOp>(loc, mean, loopCIVs);
    auto varianceVal = rewriter.create<KrnlLoadOp>(loc, variance, loopCIVs);

    // Create a KrnlIterateOp along the other dimensions.
    SmallVector<int64_t, 4> axes;
    axes.emplace_back(0);
    for (int64_t i = 2; i < rank; ++i)
      axes.emplace_back(i);
    std::vector<Value> packLoops;
    for (unsigned int i = 0; i < axes.size(); ++i) {
      packLoops.emplace_back(originalLoops[axes[i]]);
    }
    KrnlIterateOperandPack pack(rewriter, packLoops);
    for (unsigned int i = 0; i < axes.size(); ++i) {
      addDimensionToPack(rewriter, loc, pack, operand, axes[i]);
    }
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);

    Block &iterationBlock = iterateOp.bodyRegion().front();
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
    } else {
      loopIVs.emplace_back(args[0]);
    }

    auto xVal = rewriter.create<KrnlLoadOp>(loc, operand, loopIVs);
    // normalize
    auto dividend = rewriter.create<arith::SubFOp>(loc, xVal, meanVal);
    auto adjustedVarianceVal =
        rewriter.create<arith::AddFOp>(loc, varianceVal, epsilon);
    auto divisor = rewriter.create<math::SqrtOp>(loc, adjustedVarianceVal);
    auto normVal = rewriter.create<arith::DivFOp>(loc, dividend, divisor);
    // scale and shift
    auto scaleNormVal = rewriter.create<arith::MulFOp>(loc, scaleVal, normVal);
    auto shiftScaleNormVal =
        rewriter.create<arith::AddFOp>(loc, scaleNormVal, biasVal);
    rewriter.create<KrnlStoreOp>(loc, shiftScaleNormVal, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

struct ONNXInstanceNormalizationOpLowering : public ConversionPattern {

  ONNXInstanceNormalizationOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXInstanceNormalizationOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // instance_normalization{epsilon}(x, scale, bias) =
    //      scale * (x - mean) / sqrt(variance + epsilon) + bias
    ONNXInstanceNormalizationOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto elementType = memRefType.getElementType();
    auto epsilonAttr = FloatAttr::get(
        elementType, llvm::dyn_cast<ONNXInstanceNormalizationOp>(op)
                         .epsilon()
                         .convertToFloat());
    auto epsilon = rewriter.create<arith::ConstantOp>(loc, epsilonAttr);

    auto inputMemRef = operandAdaptor.input();
    auto scaleMemRef = operandAdaptor.scale();
    auto biasMemRef = operandAdaptor.B();

    // Insert an allocation and deallocation for the result of this operation.
    Value resMemRef;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      resMemRef =
          insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      resMemRef = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, {inputMemRef});

    // Operand's dimensions can be in the form of NxCxD1xD2x...xDn
    // Shapes of scale, bias must be C.

    // Get rank, bounds, and constructors.
    int64_t rank = memRefType.getRank();
    IndexExprScope outerScope(&rewriter, loc);
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
    MemRefBoundsIndexCapture inputBounds(inputMemRef);
    MemRefType tmpType = MemRefType::get({}, elementType);
    Value fZero = create.math.constant(elementType, 0);

    // Compute the number of values in a single channel: product of spatial
    // dimensions, converted to float.
    IndexExpr num = inputBounds.getSymbol(2);
    for (int d = 3; d < rank; ++d)
      num = num * inputBounds.getSymbol(d);
    // Convert num to float from Pooling postProcessPoolingWindow.
    Value meanDenom = create.math.cast(elementType, num.getValue());

    // Iterate over the batch and channels.
    LiteralIndexExpr iZero(0);
    ValueRange n_c_loopDef = create.krnl.defineLoops(2);
    create.krnl.iterateIE(n_c_loopDef, n_c_loopDef, {iZero, iZero},
        {inputBounds.getSymbol(0), inputBounds.getSymbol(1)},
        [&](KrnlBuilder &createKrnl, ValueRange n_c_loopInd) {
          MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
              createKrnl);
          IndexExprScope channelScope(createKrnl);
          DimIndexExpr n(n_c_loopInd[0]), c(n_c_loopInd[1]);

          // Set bounds for iterating over values in channel.
          ValueRange spatial_loopDef = create.krnl.defineLoops(rank - 2);
          SmallVector<IndexExpr, 4> lbs(rank - 2, iZero);
          SmallVector<IndexExpr, 4> ubs;
          for (int d = 2; d < rank; ++d)
            ubs.emplace_back(inputBounds.getSymbol(d));

          // First compute the mean: store zero in reduction value, then sum up
          // all of the values in the channel, and divide by the number of
          // values.
          Value tmpMemRef = create.mem.alloca(tmpType);
          create.krnl.store(fZero, tmpMemRef, {});
          // Iterate over kernel and add values.
          ValueRange spatial2_loopDef = create.krnl.defineLoops(rank - 2);
          create.krnl.iterateIE(spatial2_loopDef, spatial2_loopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange spatial_loopInd) {
                MathBuilder createMath(createKrnl);
                SmallVector<Value, 6> inputAccessFct = {
                    n.getValue(), c.getValue()};
                for (int d = 0; d < rank - 2; ++d)
                  inputAccessFct.emplace_back(spatial_loopInd[d]);
                // tmp += input[n,c, spatial dims]
                Value oldSum = createKrnl.load(tmpMemRef, {});
                Value val = createKrnl.load(inputMemRef, inputAccessFct);
                Value newSum = createMath.add(oldSum, val);
                createKrnl.store(newSum, tmpMemRef, {});
              });
          Value sum = create.krnl.load(tmpMemRef, {});
          Value mean = create.math.div(sum, meanDenom);
          // Second, compute the standard dev: sum of (val - mean)2 / (num-1).
          create.krnl.store(fZero, tmpMemRef, {});
          // Iterate over kernel and add values.
          create.krnl.iterateIE(spatial_loopDef, spatial_loopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange spatial_loopInd) {
                MathBuilder createMath(createKrnl);
                SmallVector<Value, 6> inputAccessFct = {
                    n.getValue(), c.getValue()};
                for (int d = 0; d < rank - 2; ++d)
                  inputAccessFct.emplace_back(spatial_loopInd[d]);
                // tmp += input[n,c, spatial dims]
                Value oldSum = createKrnl.load(tmpMemRef, {});
                Value val = createKrnl.load(inputMemRef, inputAccessFct);
                val = createMath.sub(val, mean);
                val = createMath.mul(val, val);
                Value newSum = createMath.add(oldSum, val);
                createKrnl.store(newSum, tmpMemRef, {});
              });
          sum = create.krnl.load(tmpMemRef, {});
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
                MathBuilder createMath(createKrnl);
                SmallVector<Value, 6> accessFct = {n.getValue(), c.getValue()};
                for (int d = 0; d < rank - 2; ++d)
                  accessFct.emplace_back(spatial_loopInd[d]);
                // tmp += input[n,c, spatial dims]
                Value x = createKrnl.load(inputMemRef, accessFct);
                Value val = createMath.sub(x, mean);
                val = createMath.mul(factor, val);
                val = createMath.add(val, term);
                createKrnl.store(val, resMemRef, accessFct);
              });
        }); // For all batches, channels.

    rewriter.replaceOp(op, resMemRef);
    return success();
  }
};

void populateLoweringONNXNormalizationOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXBatchNormalizationInferenceModeOpLowering>(ctx);
  patterns.insert<ONNXInstanceNormalizationOpLowering>(ctx);
}
