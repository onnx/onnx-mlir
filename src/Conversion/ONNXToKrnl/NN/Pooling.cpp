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
  return dilations;
  // if (isDefaultDilations)
  //  return {};
  // else
  //  return dilations;
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
    std::vector<Value> fwDim, std::vector<Value> fwStart,
    std::vector<Value> fwEnd) {}

// Calculate the average value for AveragePool.
template <>
void doPostProcessingForPooling<ONNXAveragePoolOp>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXAveragePoolOp poolOp,
    Value alloc, ArrayRef<Value> resultIndices, std::vector<Value> fwDim,
    std::vector<Value> fwStart, std::vector<Value> fwEnd) {
  // AveragePool's result type is FloatType, so it's safe to use DivFOp, SubFOp.
  Value numerator = rewriter.create<LoadOp>(loc, alloc, resultIndices);
  Value denominator = fwDim[0];
  for (int i = 1; i < fwDim.size(); ++i)
    denominator = rewriter.create<MulIOp>(loc, denominator, fwDim[i]);

  Value zero = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
  Value isGreaterThanZero =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, denominator, zero);

  Value average = rewriter.create<SelectOp>(loc, isGreaterThanZero,
      rewriter.create<DivFOp>(loc, numerator, denominator), numerator);

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

// Helper function to prepare the information about the filter window.
static void getFilterWindowInfo(ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<Value> spatialIndices, ArrayRef<Value> spatialDims,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> pads,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations, bool ceilMode,
    std::vector<Value> &fwDim, std::vector<Value> &fwStart,
    std::vector<Value> &fwEnd) {
  // Thanks to Tian (@tjingrant) for the following derivation.
  // When dilation is non-unit, the first valid pixel to
  // apply pooling on will not be the 0-th pixel, but rather
  // the smallest integer n to make -pH + n * dH greater than
  // or equal to 0.
  //
  // We derive what is this smallest n:
  // -pH + n * dH >= 0
  //       dH * n >= pH
  //            n >= pH/dH
  // thus n = ceil(pH/dH)
  // thus the first valid pixel location is
  // ceil(pH / dilation) * dilation - pH
  //
  // first_valid_h = ceil(float(pH / dH)) * dH - pH
  // start_h = max(first_valid_h, ho * sH - pH)
  // end_h = min(H, ho * sH + kH * dH - pH)
  // h_count = round(float(end_h - start_h) / float(dH))

  for (int i = 0; i < spatialIndices.size(); ++i) {
    // Compute startIndex and endIndex for the current dimension.
    Value firstValid = emitConstantOp(rewriter, loc, rewriter.getIndexType(),
        (dilations.empty())
            ? -pads[i]
            : (int)ceil((float)pads[i] - dilations[i]) * dilations[i]);
    Value minusPad =
        emitConstantOp(rewriter, loc, rewriter.getIndexType(), -pads[i]);
    Value stride =
        emitConstantOp(rewriter, loc, rewriter.getIndexType(), strides[i]);

    Value startIndex = rewriter.create<AddIOp>(
        loc, rewriter.create<MulIOp>(loc, spatialIndices[i], stride), minusPad);
    Value endIndex = startIndex;
    if (!dilations.empty())
      endIndex = rewriter.create<AddIOp>(loc, startIndex,
          emitConstantOp(rewriter, loc, rewriter.getIndexType(),
              kernelShape[i] * dilations[i]));

    Value maxCondition = rewriter.create<CmpIOp>(
        loc, CmpIPredicate::sgt, firstValid, startIndex);
    startIndex =
        rewriter.create<SelectOp>(loc, maxCondition, firstValid, startIndex);
    fwStart.emplace_back(startIndex);

    Value minCondition = rewriter.create<CmpIOp>(
        loc, CmpIPredicate::slt, spatialDims[i], endIndex);
    endIndex =
        rewriter.create<SelectOp>(loc, minCondition, spatialDims[i], endIndex);
    fwEnd.emplace_back(endIndex);

    // Compute dimension value.
    Value dim = rewriter.create<SubIOp>(loc, endIndex, startIndex);
    if (!dilations.empty()) {
      Value numerator = dim;
      Value denominator =
          emitConstantOp(rewriter, loc, rewriter.getIndexType(), dilations[i]);
      dim = rewriter.create<SignedDivIOp>(loc, numerator, denominator);

      if (ceilMode) {
        auto remainder =
            rewriter.create<SignedRemIOp>(loc, numerator, denominator);
        auto zero = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
        auto one = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);
        auto isZero =
            rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, remainder, zero);
        auto dimPlusOne = rewriter.create<AddIOp>(loc, dim, one);
        dim = rewriter.create<SelectOp>(loc, isZero, dim, dimPlusOne);
      }
    }
    fwDim.emplace_back(dim);
  }
}

//===----------------------------------------------------------------------===//
// Helper function to prepare indices for accessing the input data tensor.
//
void getDataIndicesForPooling(ConversionPatternRewriter &rewriter, Location loc,
    std::vector<Value> &dataIndices, BuildKrnlLoop &outerLoops,
    BuildKrnlLoop &innerLoops, ArrayRef<int64_t> pads,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations, bool ceilMode) {
  int nOuterLoops = outerLoops.getOriginalLoops().size();
  int nInnerLoops = innerLoops.getOriginalLoops().size();

  // Insert batch indices: n, c
  for (int i = 0; i < nOuterLoops - nInnerLoops; ++i)
    dataIndices.emplace_back(outerLoops.getInductionVar(i));

  // Insert spatial indices: sX * rX + kX * dX
  for (int i = nOuterLoops - nInnerLoops; i < nOuterLoops; ++i) {
    // Get index along the inner loop's induction variables.
    // It is used to obtain kernel/pad/stride/dilation index.
    int j = i - (nOuterLoops - nInnerLoops);

    Value spatialIndex = outerLoops.getInductionVar(i);
    // If strides are present (not default) then emit the correct access
    // index.
    // sX *= rX
    if (strides[j] > 1) {
      auto strideIndex =
          emitConstantOp(rewriter, loc, rewriter.getIndexType(), strides[j]);
      spatialIndex = rewriter.create<MulIOp>(
          loc, strideIndex, outerLoops.getInductionVar(i));
    }

    // If dilations are present( not default) then emit the correct access
    // index.
    auto kernelIndex = innerLoops.getInductionVar(j);
    if (!dilations.empty() && dilations[j] > 1) {
      // sX += dX * kW
      auto dilationIndex =
          emitConstantOp(rewriter, loc, rewriter.getIndexType(), dilations[j]);
      auto dilationKernelIndex =
          rewriter.create<MulIOp>(loc, dilationIndex, kernelIndex);
      spatialIndex =
          rewriter.create<AddIOp>(loc, spatialIndex, dilationKernelIndex);
    } else {
      // sX += kX
      spatialIndex = rewriter.create<AddIOp>(loc, spatialIndex, kernelIndex);
    }
    dataIndices.emplace_back(spatialIndex);
  }
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

    // Read count_include_pad attribute if the op has.
    bool countIncludePad = getCountIncludePad<PoolOp>(poolOp);
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

    // R = Pool(D)
    //
    // The input/output shapes will look like this:
    //
    // D (NxCxHxW) -> R (NxCxRHxRW)
    //
    // The loop nest will look as follows:
    //
    // strides = [s1, s2]
    // pads = [p11, p21, p12, p22]
    // dilations = [d1, d2]
    //
    // for n = 0 .. N:
    //   for c = 0 .. C:
    //     for r1 = 0 .. RH:
    //       for r2 = 0 .. RW:
    //         R[n][c][r1][r2] = getIdentityValue(...);
    //         for k1 = 0 .. KH:
    //           for k2 = 0 .. KW:
    //             t = D[n][c][s1 * r1 + k1 * d1][s2 * r2 + k2 * d2];
    //             R[n][c][r1][r2] = mapToLowerScalarOp(R[n][c][r1][r2], t);
    //         doPostProcessingForPooling(...)
    //
    // Naming:
    //   n, c, r1, r2: outer loop nest indices
    //   k1, k2: inner loop nest indices
    //   getIdentityValue(): to return the indentity value
    //     - negative infinity for MaxPool
    //     - 0 for AveragePool
    //   mapToLowerScalarOp(): to do primitive computation for Pooling, e.g.
    //     - compute max for MaxPool
    //     - compute sum for AveragePool
    //   doPostProcessingForPooling(): to do post processing over the whole
    //   filter window, e.g.
    //     - do nothing in case of MaxPool
    //     - calculate the average in case of AveragePool
    //
    // for n in range(N):
    //   for c in range(C):
    //     for ho in range(H_out):
    //       for wo in range(W_out):
    //         # When dilation is non-unit, the first valid pixel to
    //         # apply pooling on will not be the 0-th pixel, but rather
    //         # the smallest integer n to make -pH + n * 3 greater than
    //         # or equal to 0.
    //
    //         # We derive what is this smallest n:
    //         # -pH + n * 3 >= 0
    //         #          3n >= pH
    //         #           n >= pH/3
    //         # thus n = ceil(pH/3)
    //         # thus the first valid pixel location is
    //         # ceil(pH / dilation) * dilation - pH
    //
    //         first_valid_h = ceil(float(pH / dH)) * dH - pH
    //         start_h = max(first_valid_h, ho * sH - pH)
    //         end_h = min(H, ho * sH + kH * dH - pH)
    //
    //         first_valid_w = ceil(float(pW / dW)) * dW - pW
    //         start_w = max(first_valid_h, wo * sW - pW)
    //         end_w = min(W, wo * sW + kW * dW - pW)
    //
    //         h_count = round(float(end_h - start_h) / float(dH))
    //         w_count = round(float(end_w - start_w) / float(dW))
    //
    //         for hi in range(start_h, end_h, dH):
    //           for wi in range(start_w, end_w, dW):
    //             pooled[n, c, ho, wo] += imgs[n, c, hi, wi]
    //
    //         # The above for loops are implemented as follows
    //         # since KrnlIterateOp has not supported `step` yet.
    //         for khi in range(h_count):
    //           for kwi in range(w_count):
    //             hi = khi * dH + start_h
    //             wi = kwi * dW + start_w
    //             pooled[n, c, ho, wo] += imgs[n, c, hi, wi]
    //
    //         # Compute the divisor for average pooling:
    //
    //         if h_count * w_count > 0:
    //           pooled[n, c, ho, wo] = pooled[n, c, ho, wo] / (h_count *
    //           w_count)

    // Identity value of the operation.
    auto identity = getIdentityValue<PoolOp>(rewriter, loc, resultElementType);

    // 1. Define outer loops and emit empty optimization block.
    // for n = 0 .. N:
    //   for c = 0 .. C:
    //     for r1 = 0 .. RH:
    //       for r2 = 0 .. RW:
    auto nOuterLoops = resultShape.size();
    BuildKrnlLoop outerLoops(rewriter, loc, nOuterLoops);
    outerLoops.createDefineOptimizeAndIterateOp(alloc);

    auto ipMainRegion = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(outerLoops.getIterateBlock());
    {
      // 2. Emit the body of the outer loop nest, which does one filter
      // application.
      SmallVector<Value, 4> resultIndices;
      for (int i = 0; i < nOuterLoops; ++i)
        resultIndices.emplace_back(outerLoops.getInductionVar(i));

      // 2.1 Emit: R[n][c][r1][r2] = identity;
      rewriter.create<StoreOp>(loc, identity, alloc, resultIndices);

      // 2.2 Compute the filter window.
      SmallVector<Value, 4> spatialIndices;
      for (int i = 2; i < nOuterLoops; ++i)
        spatialIndices.emplace_back(outerLoops.getInductionVar(i));
      SmallVector<Value, 4> spatialDims;
      for (int i = 2; i < nOuterLoops; ++i) {
        Value dim;
        if (resultShape[i] < 0) {
          dim = rewriter.create<DimOp>(loc, alloc, i);
        } else {
          dim = emitConstantOp(
              rewriter, loc, rewriter.getIndexType(), resultShape[i]);
        }
        spatialDims.emplace_back(dim);
      }
      std::vector<Value> fwDim, fwStart, fwEnd;
      getFilterWindowInfo(rewriter, loc, spatialIndices, spatialDims,
          kernelShape, pads, strides, dilations, ceilMode, fwDim, fwStart,
          fwEnd);

      // Compute AffineMap which expresses the upper bounds for the filter
      // window's dimensions.
      AffineExpr hoExpr = rewriter.getAffineDimExpr(0);
      AffineExpr H = rewriter.getAffineSymbolExpr(0);
      AffineExpr K = rewriter.getAffineSymbolExpr(1);
      AffineExpr P = rewriter.getAffineSymbolExpr(2);
      AffineExpr S = rewriter.getAffineSymbolExpr(3);
      AffineExpr D = rewriter.getAffineSymbolExpr(4);

      AffineExpr start1 = P.ceilDiv(D) - P;
      AffineExpr start2 = hoExpr * S - P;
      AffineExpr end1 = H;
      AffineExpr end2 = hoExpr * S + K * D - P;

      SmallVector<AffineExpr, 4> dimExpr;
      dimExpr.emplace_back((ceilMode) ? (end1 - start1).ceilDiv(D)
                                      : (end1 - start1).floorDiv(D));
      dimExpr.emplace_back((ceilMode) ? (end1 - start2).ceilDiv(D)
                                      : (end1 - start2).floorDiv(D));
      dimExpr.emplace_back((ceilMode) ? (end2 - start1).ceilDiv(D)
                                      : (end2 - start1).floorDiv(D));
      dimExpr.emplace_back((ceilMode) ? (end2 - start2).ceilDiv(D)
                                      : (end2 - start2).floorDiv(D));
      AffineMap dimMap = AffineMap::get(1, 5, dimExpr);

      // Dimensions and symbols for the affine map.
      SmallVector<SmallVector<Value, 4>, 4> dimAndSyms;
      for (int i = 0; i < spatialIndices.size(); ++i) {
        SmallVector<Value, 4> dimAndSym;
        // d0
        dimAndSym.emplace_back(spatialIndices[i]);
        // s0
        dimAndSym.emplace_back(spatialDims[i]);
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
        dimAndSym.emplace_back(emitConstantOp(
            rewriter, loc, rewriter.getIndexType(), dilations[i]));
        dimAndSyms.emplace_back(dimAndSym);
      }

      // 2.3 Define inner loops.
      //   for k1 = 0 .. FH:
      //     for k2 = 0 .. FW:
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
        for (int i = 0; i < 2; ++i)
          dataIndices.emplace_back(resultIndices[i]);
        for (int i = 0; i < nInnerLoops; ++i) {
          if (!dilations.empty()) {
            // hi = khi * dH + start_h
            Value dilation = emitConstantOp(
                rewriter, loc, rewriter.getIndexType(), dilations[i]);
            Value index = rewriter.create<MulIOp>(
                loc, innerLoops.getInductionVar(i), dilation);
            index = rewriter.create<AddIOp>(loc, index, fwStart[i]);
            dataIndices.emplace_back(index);
          } else {
            // hi = khi + start_h
            dataIndices.emplace_back(rewriter.create<AddIOp>(
                loc, innerLoops.getInductionVar(i), fwStart[i]));
          }
        }

        Value loadData =
            rewriter.create<LoadOp>(loc, inputOperand, dataIndices);
        Value loadPartialResult =
            rewriter.create<LoadOp>(loc, alloc, resultIndices);
        Value result = emitScalarOpFor<ONNXMaxPoolSingleOutOp>(rewriter, loc,
            op, resultElementType, {loadPartialResult, loadData});
        rewriter.create<StoreOp>(loc, result, alloc, resultIndices);
      }

      // 2.5 Post-processing in the outer loop nest, e.g. taking average.
      rewriter.restoreInsertionPoint(ipOuterLoops);
      doPostProcessingForPooling<PoolOp>(
          rewriter, loc, poolOp, alloc, resultIndices, fwDim, fwStart, fwEnd);
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
