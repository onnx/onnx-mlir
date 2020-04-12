//===---------------- Pooling.cpp - Lowering Pooling Ops ------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Pooling Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
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
    if (dilationValue > 1 and isDefaultDilations)
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
void doPostProcessingForPooling(ConversionPatternRewriter &rewriter,
    Location loc, PoolOp poolOp, Value alloc, ArrayRef<Value> resultIndices,
    ArrayRef<int64_t> kernelShape, ArrayRef<Value> fwDim) {}

// Calculate the average value for AveragePool.
template <>
void doPostProcessingForPooling<ONNXAveragePoolOp>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXAveragePoolOp poolOp,
    Value alloc, ArrayRef<Value> resultIndices, ArrayRef<int64_t> kernelShape,
    ArrayRef<Value> fwDim) {
  // AveragePool's result type is FloatType, so it's safe to use DivFOp, SubFOp.
  bool countIncludePad = getCountIncludePad<ONNXAveragePoolOp>(poolOp);
  Value numerator = rewriter.create<LoadOp>(loc, alloc, resultIndices);
  Value denominator;
  if (countIncludePad) {
    int64_t kernelSize = 1;
    for (int i = 0; i < kernelShape.size(); ++i)
      kernelSize *= kernelShape[i];
    denominator =
        emitConstantOp(rewriter, loc, numerator.getType(), kernelSize);
  } else {
    denominator = fwDim[0];
    for (int i = 1; i < fwDim.size(); ++i)
      denominator = rewriter.create<MulIOp>(loc, denominator, fwDim[i]);
    denominator = rewriter.create<IndexCastOp>(
        loc, denominator, rewriter.getIntegerType(64));
    denominator =
        rewriter.create<SIToFPOp>(loc, denominator, numerator.getType());
  }

  Value average = rewriter.create<DivFOp>(loc, numerator, denominator);

  rewriter.create<StoreOp>(loc, average, alloc, resultIndices);
}

//===----------------------------------------------------------------------===//
// Helper function to insert alloc and dealloc ops for memref of dynamic shape.
//
Value insertAllocAndDeallocForPooling(ConversionPatternRewriter &rewriter,
    Location loc, bool insertDealloc, MemRefType memRefType, Value inputOperand,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> pads,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations, bool ceilMode) {
  AllocOp alloc;

  // Shape and rank information related to result and kernel.
  auto resultShape = memRefType.getShape();
  auto resultRank = resultShape.size();
  auto kernelRank = kernelShape.size();
  auto kernelOffset = resultRank - kernelRank;

  // Compute dimensions of the result of this operation.
  SmallVector<Value, 2> allocOperands;
  for (int i = 0; i < kernelOffset; ++i) {
    if (resultShape[i] < 0) {
      auto dim = rewriter.create<DimOp>(loc, inputOperand, i);
      allocOperands.emplace_back(dim);
    }
  }

  Value zero, one;
  if (ceilMode) {
    zero = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
  }
  one = rewriter.create<ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));

  int64_t dilation = 1;
  for (int i = kernelOffset; i < resultShape.size(); ++i) {
    if (resultShape[i] < 0) {
      // dim =
      //   let numerator = (input + pad - (kernel - 1) * dilation - 1)
      //   in let denominator = stride
      //      in
      //        if (ceilMode)
      //          ceil(numerator / denominator) + 1
      //        else
      //          floor(numerator / denominator) + 1
      int spatialIndex = i - kernelOffset;

      // numerator = (input + pad - (kernel - 1) * dilation - 1)
      dilation = dilations.empty() ? dilation : dilations[spatialIndex];
      int64_t padKernelDilation =
          (pads[spatialIndex] + pads[spatialIndex + kernelRank]) -
          (kernelShape[spatialIndex] - 1) * dilation - 1;
      auto padKernelDilationVal = emitConstantOp(
          rewriter, loc, rewriter.getIntegerType(64), padKernelDilation);
      auto inputDim = rewriter.create<DimOp>(loc, inputOperand, i);
      auto inputDimVal = rewriter.create<IndexCastOp>(
          loc, inputDim, rewriter.getIntegerType(64));
      auto numeratorVal =
          rewriter.create<AddIOp>(loc, inputDimVal, padKernelDilationVal);
      // denominator
      auto denominatorVal = emitConstantOp(
          rewriter, loc, rewriter.getIntegerType(64), strides[spatialIndex]);

      // numerator / denominator
      Value dimVal =
          rewriter.create<SignedDivIOp>(loc, numeratorVal, denominatorVal);

      if (ceilMode) {
        auto remainder =
            rewriter.create<SignedRemIOp>(loc, numeratorVal, denominatorVal);
        auto isZero =
            rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, remainder, zero);
        auto dimPlusOne = rewriter.create<AddIOp>(loc, dimVal, one);
        dimVal = rewriter.create<SelectOp>(loc, isZero, dimVal, dimPlusOne);
      }

      dimVal = rewriter.create<AddIOp>(loc, dimVal, one);
      allocOperands.emplace_back(
          rewriter.create<IndexCastOp>(loc, dimVal, rewriter.getIndexType()));
    }
  }
  alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
  if (insertDealloc) {
    auto *parentBlock = alloc.getOperation()->getBlock();
    auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
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
    ONNXMaxPoolSingleOutOpOperandAdaptor operandAdaptor(operands);
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
    auto ceilMode = poolOp.ceil_mode().getSExtValue();

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
    auto resultShape = memRefType.getShape();
    auto resultElementType = memRefType.getElementType();

    // Insert an allocation and deallocation for the result of this operation.
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
    // dilations = [d1, d2]
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
    //         hDim= round(float(endH - startH) / float(dH))
    //         wDim= round(float(endW - startW) / float(dW))
    //
    //         for hi in range(startH, endH, dH):
    //           for wi in range(startW, endW, dW):
    //             output[n][c][ho][wo] = emitScalarOpFor(output[n][c][ho][wo],
    //                                                    input[n, c, hi, wi]);
    //
    //         # The above two for-loops are rewritten as follows:
    //         # (since KrnlIterateOp has not supported `step` yet)
    //         for hf in range(hDim):
    //           for wf in range(wDim):
    //             hi = hf * dH + startH
    //             wi = wf * dW + startW
    //             output[n][c][ho][wo] = emitScalarOpFor(output[n][c][ho][wo],
    //                                                    input[n, c, hi, wi]);
    //
    //         # Do post processing such as taking average pooling:
    //         doPostProcessingForPooling(...)
    //
    // Helper functions:
    //   getIdentityValue(): to return the indentity value
    //     - negative infinity for MaxPool
    //     - 0 for AveragePool
    //   emitScalarOpFor(): to do primitive computation for Pooling, e.g.
    //     - compute max for MaxPool
    //     - compute sum for AveragePool
    //   doPostProcessingForPooling(): to do post processing over the whole
    //   filter window, e.g.
    //     - do nothing in case of MaxPool
    //     - calculate the average in case of AveragePool, e.g.
    //         if hDim * wDim> 0:
    //           output[n, c, ho, wo] = output[n, c, ho, wo] / (hDim*wDim)
    //

    // Identity value of the operation.
    auto identity = getIdentityValue<PoolOp>(rewriter, loc, resultElementType);

    // 1. Define outer loops and emit empty optimization block.
    // for n in range(N):
    //   for c in range(C):
    //     for ho in range(HO):
    //       for wo in range(WO):
    BuildKrnlLoop outerLoops(rewriter, loc, resultShape.size());
    outerLoops.createDefineOptimizeAndIterateOp(alloc);

    auto ipMainRegion = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(outerLoops.getIterateBlock());
    {
      // 2. Emit the body of the outer loop nest, which does one filter
      // application.
      SmallVector<Value, 4> resultIndices;
      for (int i = 0; i < resultShape.size(); ++i)
        resultIndices.emplace_back(outerLoops.getInductionVar(i));

      // 2.1 Emit: output[n][c][ho][wo] = identity
      rewriter.create<StoreOp>(loc, identity, alloc, resultIndices);

      // 2.2 Compute the filter window.
      int spatialStartIndex = resultShape.size() - kernelShape.size();
      SmallVector<Value, 4> spatialIndices;
      for (int i = spatialStartIndex; i < resultShape.size(); ++i)
        spatialIndices.emplace_back(outerLoops.getInductionVar(i));

      SmallVector<Value, 4> inputDims;
      for (int i = spatialStartIndex; i < resultShape.size(); ++i) {
        Value dim;
        if (inputShape[i] < 0) {
          dim = rewriter.create<DimOp>(loc, inputOperand, i);
        } else {
          dim = emitConstantOp(
              rewriter, loc, rewriter.getIndexType(), inputShape[i]);
        }
        inputDims.emplace_back(dim);
      }
      // Compute AffineMap which expresses the upper bounds for the filter
      // window's dimensions.
      //   firstValidH = ceil(float(ptH / dH)) * dH - ptH
      //   startH = max(firstValidH, ho * sH - ptH)
      //   endH = min(H, ho * sH + (kH - 1) * dH  + 1 - pbH)
      //   hDim = round(float(endH - startH) / float(dH))

      AffineExpr outputIndex = rewriter.getAffineDimExpr(0);
      AffineExpr inputDim = rewriter.getAffineSymbolExpr(0);
      AffineExpr kernelDim = rewriter.getAffineSymbolExpr(1);
      AffineExpr padTopDim = rewriter.getAffineSymbolExpr(2);
      AffineExpr strideDim = rewriter.getAffineSymbolExpr(3);
      AffineExpr dilationDim = rewriter.getAffineSymbolExpr(4);
      AffineExpr start1 =
          (padTopDim).ceilDiv(dilationDim) * dilationDim - padTopDim;
      AffineExpr start2 = outputIndex * strideDim - padTopDim;
      AffineExpr end1 = inputDim;
      AffineExpr end2 = outputIndex * strideDim +
                        (kernelDim - 1) * dilationDim + 1 - padTopDim;

      SmallVector<AffineExpr, 4> dimExpr;
      // Upperbound for an affine.for is `min AffineMap`, where `min` is
      // automatically inserted when an affine.for is constructed from
      // an AffineMap, thus we rewrite `endH - startH` as follows:
      //   endH - start H
      //     = min(end1, end2) - max(start1, start2)
      //     = min(end1 - start1, end1 - start2, end2 - start1, end2 - start2)
      AffineExpr dimExpr1 = end1 - start1;
      AffineExpr dimExpr2 = end1 - start2;
      AffineExpr dimExpr3 = end2 - start1;
      AffineExpr dimExpr4 = end2 - start2;
      for (AffineExpr de : {dimExpr1, dimExpr2, dimExpr3, dimExpr4}) {
        if (isDilated) {
          de = de + 1;
          de = (ceilMode) ? de.ceilDiv(dilationDim) : de.floorDiv(dilationDim);
        }
        dimExpr.emplace_back(de);
      }
      AffineMap dimMap = AffineMap::get(1, 5, dimExpr);

      // Dimensions and symbols for the affine map.
      SmallVector<SmallVector<Value, 4>, 4> dimAndSyms;
      for (int i = 0; i < spatialIndices.size(); ++i) {
        SmallVector<Value, 4> dimAndSym;
        // d0
        dimAndSym.emplace_back(spatialIndices[i]);
        // s0
        dimAndSym.emplace_back(inputDims[i]);
        // s1
        dimAndSym.emplace_back(emitConstantOp(
            rewriter, loc, rewriter.getIndexType(), kernelShape[i]));
        // s2
        dimAndSym.emplace_back(
            emitConstantOp(rewriter, loc, rewriter.getIndexType(), pads[i]));
        // s3
        dimAndSym.emplace_back(
            emitConstantOp(rewriter, loc, rewriter.getIndexType(), strides[i]));
        // s4
        dimAndSym.emplace_back(emitConstantOp(rewriter, loc,
            rewriter.getIndexType(), (isDilated) ? dilations[i] : 1));
        dimAndSyms.emplace_back(dimAndSym);
      }

      // Compute values for the filter window's dimensions.
      SmallVector<Value, 4> fwStart, fwDim;
      auto startMap = AffineMap::get(1, 5, {start1, start2});
      auto endMap = AffineMap::get(1, 5, {end1, end2});
      for (int i = 0; i < spatialIndices.size(); ++i) {
        Value startIndex = rewriter.create<AffineMaxOp>(
            loc, startMap, ValueRange(dimAndSyms[i]));
        fwStart.emplace_back(startIndex);

        Value endIndex = rewriter.create<AffineMinOp>(
            loc, endMap, ValueRange(dimAndSyms[i]));

        Value dim = rewriter.create<SubIOp>(loc, endIndex, startIndex);
        if (isDilated && dilations[i] != 1) {
          Value one = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);
          Value numerator = rewriter.create<AddIOp>(loc, dim, one);
          Value denominator = dimAndSyms[i][5]; // dilations[i]
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
        fwDim.emplace_back(dim);
      }

      // 2.3 Define inner loops.
      //  for hf in range(hDim):
      //    for wf in range(wDim):
      //      hi = hf * dH + startH
      //      wi = wf * dW + startW
      //      output[n][c][ho][wo] =
      //        emitScalarOpFor(output[n][c][ho][wo], input[n, c, hi, wi]);
      int nInnerLoops = spatialIndices.size();
      BuildKrnlLoop innerLoops(rewriter, loc, nInnerLoops);
      innerLoops.createDefineAndOptimizeOp();
      for (int i = 0; i < nInnerLoops; ++i)
        innerLoops.pushAffineMapBounds(
            0, dimMap, llvm::makeArrayRef(dimAndSyms[i]));

      // 2.4 Emit inner loop nest.
      innerLoops.createIterateOp();

      auto ipOuterLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(innerLoops.getIterateBlock());
      {
        std::vector<Value> dataIndices;
        // Compute data indices.
        for (int i = 0; i < spatialStartIndex; ++i)
          dataIndices.emplace_back(resultIndices[i]);
        for (int i = 0; i < nInnerLoops; ++i) {
          if (isDilated && dilations[i] > 1) {
            // hi = hf * dH + startH
            Value index = rewriter.create<MulIOp>(
                loc, innerLoops.getInductionVar(i), dimAndSyms[i][5]);
            index = rewriter.create<AddIOp>(loc, index, fwStart[i]);
            dataIndices.emplace_back(index);
          } else {
            // hi = hf + startH
            dataIndices.emplace_back(rewriter.create<AddIOp>(
                loc, innerLoops.getInductionVar(i), fwStart[i]));
          }
        }

        Value loadData =
            rewriter.create<LoadOp>(loc, inputOperand, dataIndices);
        Value loadPartialResult =
            rewriter.create<LoadOp>(loc, alloc, resultIndices);
        Value result = emitScalarOpFor<PoolOp>(rewriter, loc, op,
            resultElementType, {loadPartialResult, loadData});
        rewriter.create<StoreOp>(loc, result, alloc, resultIndices);
      }

      // 2.5 Post-processing in the outer loop nest, e.g. taking average.
      rewriter.restoreInsertionPoint(ipOuterLoops);
      doPostProcessingForPooling<PoolOp>(
          rewriter, loc, poolOp, alloc, resultIndices, kernelShape, fwDim);
    }

    // Go back to the main region.
    rewriter.restoreInsertionPoint(ipMainRegion);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXPoolingOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPoolOpLowering<ONNXMaxPoolSingleOutOp>>(ctx);
  patterns.insert<ONNXPoolOpLowering<ONNXAveragePoolOp>>(ctx);
}
