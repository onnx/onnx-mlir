/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Pooling.cpp - Lowering Pooling Ops ------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Pooling Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

// Identity values
template <>
Value getIdentityValue<ONNXMaxPoolSingleOutOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return emitNegativeInfinityConstantOp(rewriter, loc, type);
}

template <>
Value getIdentityValue<ONNXAveragePoolOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return emitConstantOp(rewriter, loc, type, 0);
}

// Scalar operations
template <>
struct ScalarOp<ONNXAveragePoolOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

template <>
Value emitScalarOpFor<ONNXMaxPoolSingleOutOp>(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    Type elementType, ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  auto max = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, max, lhs, rhs);
  return result;
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
  // AveragePool's result type is FloatType, so it's safe to use DivFOp, SubFOp.
  bool countIncludePad = getCountIncludePad<ONNXAveragePoolOp>(poolOp);
  Value numerator = rewriter.create<KrnlLoadOp>(loc, alloc, resultIndices);
  Value denominator;
  if (countIncludePad) {
    int64_t kernelSize = 1;
    for (int i = 0; i < kernelShape.size(); ++i)
      kernelSize *= kernelShape[i];
    denominator =
        emitConstantOp(rewriter, loc, numerator.getType(), kernelSize);
  } else {
    denominator = poolDimValues[0];
    for (int i = 1; i < poolDimValues.size(); ++i)
      denominator = rewriter.create<MulIOp>(loc, denominator, poolDimValues[i]);
    denominator = rewriter.create<IndexCastOp>(
        loc, denominator, rewriter.getIntegerType(64));
    denominator =
        rewriter.create<SIToFPOp>(loc, denominator, numerator.getType());
  }

  Value average = rewriter.create<DivFOp>(loc, numerator, denominator);

  rewriter.create<KrnlStoreOp>(loc, average, alloc, resultIndices);
}

//===----------------------------------------------------------------------===//
// Helper function to insert alloc and dealloc ops for memref of dynamic shape.
//
Value insertAllocAndDeallocForPooling(ConversionPatternRewriter &rewriter,
    Location loc, bool insertDealloc, MemRefType memRefType, Value inputOperand,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> pads,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations, bool ceilMode) {
  memref::AllocOp alloc;

  // Shape and rank information related to result and kernel.
  auto resultShape = memRefType.getShape();
  auto resultRank = resultShape.size();
  auto kernelRank = kernelShape.size();
  auto kernelOffset = resultRank - kernelRank;

  // Compute dimensions of the result of this operation.
  SmallVector<Value, 2> allocOperands;
  for (int i = 0; i < kernelOffset; ++i) {
    if (resultShape[i] < 0) {
      auto dim = rewriter.create<memref::DimOp>(loc, inputOperand, i);
      allocOperands.emplace_back(dim);
    }
  }

  // Obtain an affine map to compute the output dimension.
  AffineMap dimMap = getConvDimMap(rewriter, ceilMode);
  for (int i = kernelOffset; i < resultShape.size(); ++i) {
    if (resultShape[i] < 0) {
      int spatialIndex = i - kernelOffset;
      // Prepare arguments for the affine map.
      SmallVector<Value, 4> dimArgs;
      dimArgs.emplace_back(
          rewriter.create<memref::DimOp>(loc, inputOperand, i));
      dimArgs.emplace_back(emitConstantOp(
          rewriter, loc, rewriter.getIndexType(), kernelShape[spatialIndex]));
      dimArgs.emplace_back(
          emitConstantOp(rewriter, loc, rewriter.getIndexType(),
              (pads[spatialIndex] + pads[spatialIndex + kernelRank])));
      dimArgs.emplace_back(emitConstantOp(
          rewriter, loc, rewriter.getIndexType(), strides[spatialIndex]));
      dimArgs.emplace_back(
          emitConstantOp(rewriter, loc, rewriter.getIndexType(),
              dilations.empty() ? 1 : dilations[spatialIndex]));

      // Apply the affine map.
      Value dimVal = rewriter.create<AffineApplyOp>(loc, dimMap, dimArgs);
      allocOperands.emplace_back(dimVal);
    }
  }
  alloc = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
  if (insertDealloc) {
    auto *parentBlock = alloc.getOperation()->getBlock();
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }
  return alloc;
}

//===----------------------------------------------------------------------===//
// Template function that does pooling.
//
template <typename PoolOp>
struct ONNXPoolOpLowering : public ConversionPattern {
  ONNXPoolOpLowering(MLIRContext *ctx)
      : ConversionPattern(PoolOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXMaxPoolSingleOutOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    PoolOp poolOp = llvm::dyn_cast<PoolOp>(op);

    // Read kernel_shape attribute
    SmallVector<int64_t, 4> kernelShape;
    auto kernelShapeAttribute = poolOp.kernel_shapeAttr();
    for (Attribute dim : kernelShapeAttribute.getValue())
      kernelShape.emplace_back(dim.cast<IntegerAttr>().getInt());

    // Read strides attribute
    SmallVector<int64_t, 4> strides;
    auto stridesAttribute = poolOp.stridesAttr();
    for (Attribute stride : stridesAttribute.getValue())
      strides.emplace_back(stride.cast<IntegerAttr>().getInt());

    // Read ceil_mode attribute
    auto ceilMode = poolOp.ceil_mode();

    // Read pads attribute
    SmallVector<int64_t, 4> pads;
    auto padsAttribute = poolOp.padsAttr();
    for (Attribute pad : padsAttribute.getValue())
      pads.emplace_back(pad.cast<IntegerAttr>().getInt());

    // Read dilations attribute if the op has.
    std::vector<int64_t> dilations = getDilations<PoolOp>(poolOp);
    bool isDilated = !dilations.empty();

    // Type information about the input and result of this operation.
    auto inputOperand = operandAdaptor.X();
    auto inputShape = inputOperand.getType().cast<MemRefType>().getShape();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto outputShape = memRefType.getShape();
    auto outputElementType = memRefType.getElementType();

    // Kernel offset in the input shape.
    int kernelOffset = inputShape.size() - kernelShape.size();

    // Scope for krnl EDSC ops
    using namespace mlir::edsc;
    ScopedContext scope(rewriter, loc);
    // Scope for IndexExpr.
    IndexExprScope ieScope(&rewriter, loc);

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      alloc = insertAllocAndDeallocForPooling(rewriter, loc, insertDealloc,
          memRefType, inputOperand, kernelShape, pads, strides, dilations,
          ceilMode);
    }

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
      for (int i = 0; i < outputShape.size(); ++i)
        outputIndices.emplace_back(
            DimIndexExpr(outputLoops.getInductionVar(i)));

      // 2.1 Emit: output[n][c][ho][wo] = identity
      // Create a local reduction value for output[n][c][ho][wo].
      Value reductionVal = rewriter.create<memref::AllocaOp>(
          loc, MemRefType::get({}, memRefType.getElementType()));
      rewriter.create<KrnlStoreOp>(
          loc, identity, reductionVal, ArrayRef<Value>{});

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
        for (int i = 0; i < kernelShape.size(); ++i) {
          int j = i + kernelOffset;
          SmallVector<IndexExpr, 4> ic;
          // d0, output
          ic.emplace_back(outputIndices[j]);
          // s0, input dim
          ic.emplace_back(inputBounds.getDim(j));
          // s1, kernel dim
          ic.emplace_back(LiteralIndexExpr(kernelShape[i]));
          // s2, pad dim
          ic.emplace_back(LiteralIndexExpr(pads[i]));
          // s3, stride dim
          ic.emplace_back(LiteralIndexExpr(strides[i]));
          // s4, dilation dim
          ic.emplace_back(LiteralIndexExpr((isDilated) ? dilations[i] : 1));
          IVExprs.emplace_back(ic);
        }
      }

      // Compute the start and end position of the conv window.
      //   firstValidH = ceil(float(ptH / dH)) * dH - ptH
      //   startH = max(firstValidH, ho * sH - ptH)
      //   endH = min(H, ho * sH + (kH - 1) * dH  + 1 - pbH)
      SmallVector<IndexExpr, 4> windowStartExprs, windowEndExprs;
      for (int i = 0; i < kernelShape.size(); ++i) {
        std::vector<mlir::IndexExpr> exprs =
            getIndexExprsForConvWindow(IVExprs[i], ceilMode, isDilated);
        windowStartExprs.emplace_back(exprs[0]);
        windowEndExprs.emplace_back(exprs[1]);
      }

      // Compute the size of the full conv window.
      //   hDim = round(float(endH - startH) / float(dH))
      //   wDim = round(float(endW - startW) / float(dW))
      SmallVector<Value, 4> fullWindowSize;
      for (int i = 0; i < kernelShape.size(); ++i) {
        Value dim = rewriter.create<SubIOp>(
            loc, windowEndExprs[i].getValue(), windowStartExprs[i].getValue());
        if (isDilated) {
          Value one = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);
          Value numerator = rewriter.create<AddIOp>(loc, dim, one);
          Value denominator = IVExprs[i][5].getValue(); // dilations[i]
          dim = rewriter.create<SignedDivIOp>(loc, numerator, denominator);
          if (ceilMode) {
            auto remainder =
                rewriter.create<SignedRemIOp>(loc, numerator, denominator);
            Value zero =
                emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
            auto isZero = rewriter.create<CmpIOp>(
                loc, CmpIPredicate::eq, remainder, zero);
            auto dimPlusOne = rewriter.create<AddIOp>(loc, dim, one);
            dim = rewriter.create<SelectOp>(loc, isZero, dim, dimPlusOne);
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
      BuildKrnlLoop poolingLoops(rewriter, loc, kernelShape.size());
      poolingLoops.createDefineOp();
      // Push bounds.
      AffineMap windowSizeMap =
          getWindowAffineMap(rewriter, ceilMode, isDilated);
      for (int i = 0; i < kernelShape.size(); ++i) {
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
          for (int i = kernelOffset; i < inputShape.size(); ++i) {
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
        Value loadInput = krnl_load(inputOperand, inputIndices);
        Value loadPartialOutput =
            rewriter.create<KrnlLoadOp>(loc, reductionVal, ArrayRef<Value>{});
        Value output = emitScalarOpFor<PoolOp>(rewriter, loc, op,
            outputElementType, {loadPartialOutput, loadInput});
        rewriter.create<KrnlStoreOp>(
            loc, output, reductionVal, ArrayRef<Value>{});
      }
      rewriter.restoreInsertionPoint(ipOuterLoopRegion);
      Value output =
          rewriter.create<KrnlLoadOp>(loc, reductionVal, ArrayRef<Value>{});
      krnl_store(output, alloc, outputIndices);

      // 2.5 Post-processing for the pooling window, e.g. taking average.
      SmallVector<Value, 4> outputIndicesInValue;
      for (IndexExpr expr : outputIndices)
        outputIndicesInValue.emplace_back(expr.getValue());
      postProcessPoolingWindow<PoolOp>(rewriter, loc, poolOp, alloc,
          outputIndicesInValue, kernelShape, fullWindowSize);
    }

    // Go back to the main region.
    rewriter.restoreInsertionPoint(ipMainRegion);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXPoolingOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPoolOpLowering<ONNXMaxPoolSingleOutOp>>(ctx);
  patterns.insert<ONNXPoolOpLowering<ONNXAveragePoolOp>>(ctx);
}
