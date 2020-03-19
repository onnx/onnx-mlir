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

template <>
struct ScalarOp<ONNXAveragePoolOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

template <>
Value mapToLowerScalarOp<ONNXMaxPoolSingleOutOp>(Operation *op,
    ArrayRef<Type> result_types, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  Value lhs = operands[0];
  Value rhs = operands[1];
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

// MaxPool
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

// AveragePool does not have dilations.
template <>
std::vector<int64_t> getDilations<ONNXAveragePoolOp>(ONNXAveragePoolOp poolOp) {
  return {};
}

//===----------------------------------------------------------------------===//
// Get count_include_pad values
//
template <typename PoolOp>
bool getCountIncludePad(PoolOp poolOp) {
  return false;
}

// MaxPool does not have dilations.
template <>
bool getCountIncludePad<ONNXMaxPoolSingleOutOp>(ONNXMaxPoolSingleOutOp poolOp) {
  return false;
}

// AveragePool
template <>
bool getCountIncludePad<ONNXAveragePoolOp>(ONNXAveragePoolOp poolOp) {
  return (poolOp.count_include_pad() == 1);
}

//===----------------------------------------------------------------------===//
// Helper function to do post-processing after applying a filter window.
//
template <typename PoolOp>
void doPostProcessingForPooling(ConversionPatternRewriter &rewriter,
    Location loc, PoolOp poolOp, Value outOfBoundCount,
    ArrayRef<int64_t> kernelShape, ArrayRef<Value> resultIndices, Value alloc) {
}

// MaxPool does not have post-processing.
template <>
void doPostProcessingForPooling<ONNXMaxPoolSingleOutOp>(
    ConversionPatternRewriter &rewriter, Location loc,
    ONNXMaxPoolSingleOutOp poolOp, Value outOfBoundCount,
    ArrayRef<int64_t> kernelShape, ArrayRef<Value> resultIndices, Value alloc) {
}

// AveragePool
// AveragePool's result type is FloatType, so it's safe to use DivFOp, SubFOp.
template <>
void doPostProcessingForPooling<ONNXAveragePoolOp>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXAveragePoolOp poolOp,
    Value outOfBoundCount, ArrayRef<int64_t> kernelShape,
    ArrayRef<Value> resultIndices, Value alloc) {
  bool countIncludePad = (poolOp.count_include_pad() != 0);

  // Compute denomitor to take average.
  int64_t kernelSize = 1;
  for (int i = 0; i < kernelShape.size(); ++i) {
    kernelSize *= kernelShape[i];
  }
  Value denomitor = emitConstantOp(rewriter, loc,
      alloc.getType().cast<MemRefType>().getElementType(), kernelSize);
  if (!countIncludePad) {
    auto outOfBoundCountVal = rewriter.create<LoadOp>(loc, outOfBoundCount);
    denomitor = rewriter.create<SubFOp>(loc, denomitor, outOfBoundCountVal);
  }

  Value loadResult = rewriter.create<LoadOp>(loc, alloc, resultIndices);
  loadResult = rewriter.create<DivFOp>(loc, loadResult, denomitor);
  rewriter.create<StoreOp>(loc, loadResult, alloc, resultIndices);
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

  int spatialRank = resultRank - kernelOffset;
  int64_t dilation = 1;
  for (int i = kernelOffset; i < resultShape.size(); ++i) {
    if (resultShape[i] < 0) {
      // dim =
      //   let numerator = (input + pad - (kernel - 1) * dilation - 1)
      //   in let denomitor = stride
      //      in
      //        if (ceilMode)
      //          ceil(numerator / denominator) + 1
      //        else
      //          floor(numerator / denominator) + 1
      int spatialIndex = i - kernelOffset;

      // numerator = (input + pad - (kernel - 1) * dilation - 1)
      dilation = dilations.empty() ? dilation : dilations[spatialIndex];
      int64_t padKernelDilation =
          (pads[spatialIndex] + pads[spatialIndex + spatialRank]) -
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
// Helper function to prepare indices for accessing the input data tensor.
//
void getDataIndicesForPooling(ConversionPatternRewriter &rewriter, Location loc,
    std::vector<Value> &dataIndices, BuildKrnlLoop &outerLoops,
    BuildKrnlLoop &innerLoops, Value inputOperand, ArrayRef<int64_t> pads,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations, bool ceilMode) {
  auto inputShape = inputOperand.getType().cast<MemRefType>().getShape();

  // Insert batch indices: n, c
  int batchRank = 2;
  for (int i = 0; i < batchRank; ++i)
    dataIndices.emplace_back(outerLoops.getInductionVar(i));

  int nOuterLoops = outerLoops.getOriginalLoops().size();
  // Insert spatial indices: sX * rX + kX * dX
  for (int i = batchRank; i < nOuterLoops; ++i) {
    // Get index along the inner loop's induction variables.
    // It is used to obtain kernel/pad/stride/dilation index.
    int j = i - batchRank;

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
template<typename PoolOp>
struct ONNXPoolOpLowering : public ConversionPattern {
  ONNXPoolOpLowering(MLIRContext *ctx)
      : ConversionPattern(PoolOp::getOperationName(), 1, ctx) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
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
    auto &inputOperand = operands[0];
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
    //         R[n][c][r1][r2] = negative_infinity;
    //         for k1 = 0 .. KH:
    //           for k2 = 0 .. KW:
    //             t = D[n][c][s1 * r1 + k1][s2 * r2 + k2];
    //             R[n][c][r1][r2] = max(R[n][c][r1][r2], t);
    //
    // Naming:
    //   n, c, r1, r2: outer loop nest indices
    //
    // TODO: handle padding.
    //

    auto identity =
      getIdentityValue<PoolOp>(rewriter, loc, resultElementType);

    // 1. Define outer loops and emit empty optimization block.
    // for n = 0 .. N:
    //   for c = 0 .. C:
    //     for r1 = 0 .. RH:
    //       for r2 = 0 .. RW:
    auto nOuterLoops = resultShape.size();
    BuildKrnlLoop outerLoops(rewriter, loc, nOuterLoops);
    outerLoops.createDefineOptimizeAndIterateOp(alloc);
    rewriter.setInsertionPointToStart(outerLoops.getIterateBlock());
    {
      // 2. Emit the body of the outer loop nest, which does one filter
      // application.
      SmallVector<Value, 4> resultIndices;
      for (int i = 0; i < nOuterLoops; ++i)
        resultIndices.emplace_back(outerLoops.getInductionVar(i));

      // 2.1 Emit: R[n][c][r1][r2] = identity;
      rewriter.create<StoreOp>(loc, identity, alloc, resultIndices);

      // Record the number of pixels that are out-of-bound.
      Value outOfBoundCount = insertAllocAndDealloc(
          MemRefType::get({}, resultElementType, {}, 0), loc, rewriter, false);
      Value zero = emitConstantOp(rewriter, loc, resultElementType, 0);
      rewriter.create<StoreOp>(loc, zero, outOfBoundCount);

      // 2.2 Define inner loops.
      //   for k1 = 0 .. KH:
      //     for k2 = 0 .. KW:
      int nInnerLoops = kernelShape.size();
      BuildKrnlLoop innerLoops(rewriter, loc, nInnerLoops);
      innerLoops.createDefineAndOptimizeOp();
      for (int i = 0; i < nInnerLoops; ++i)
        innerLoops.pushBounds(0, kernelShape[i]);

      // 2.3 Emit inner loop nest.
      innerLoops.createIterateOp();

      // 2.4 Post-processing, e.g. taking average.
      doPostProcessingForPooling<PoolOp>(rewriter, loc, poolOp,
          outOfBoundCount, kernelShape, resultIndices, alloc);

      rewriter.setInsertionPointToStart(innerLoops.getIterateBlock());
      {
        std::vector<Value> dataIndices;
        // Compute data index for the result index.
        getDataIndicesForPooling(rewriter, loc, dataIndices, outerLoops,
            innerLoops, inputOperand, pads, strides, dilations, ceilMode);

        // Check whether the data index is out-of-bound or not? This happens
        // when ceil mode or dilation is enabled.
        // // Example of out-of-bound.
        // - Given a 5x5 input X
        //  X = [[0, 0, 0, 0, 0],
        //       [1, 1, 1, 1, 1],
        //       [2, 2, 2, 2, 2],
        //       [3, 3, 3, 3, 3],
        //       [4, 4, 4, 4, 4]]
        // - Do MaxPool with strides=[2, 2], kernel=[2, 2], ceilMode=true,
        // output is a 3x3 array:
        // Y = [[1, 1, 1],
        //      [3, 3, 3],
        //      [4, 4, 4]]
        // - When computing Y[2, 0]:
        //    - In case of kernelIndex = 1, stride = 2
        //      - No dilation: spatialIndex = 2 * 2 + 1 = 5
        //        => out of bound
        //      - dilation = 2: spatialIndex = 2 * 2 + 2 * 1 = 6
        //        => out of bound
        Value outOfBound;
        if (isDilated or ceilMode) {
          for (int i = 2; i < dataIndices.size(); ++i) {
            Value upperIndex;
            if (inputShape[i] < 0) {
              auto inputDim = rewriter.create<DimOp>(loc, inputOperand, i);
              auto oneIndex =
                  emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);
              upperIndex = rewriter.create<SubIOp>(loc, inputDim, oneIndex);
            } else {
              upperIndex =
                  rewriter.create<ConstantIndexOp>(loc, inputShape[i] - 1);
            }
            if (outOfBound) {
              auto next = rewriter.create<CmpIOp>(
                  loc, CmpIPredicate::sgt, dataIndices[i], upperIndex);
              outOfBound =
                  rewriter.create<OrOp>(loc, outOfBound, next);
            } else {
              outOfBound = rewriter.create<CmpIOp>(
                  loc, CmpIPredicate::sgt, dataIndices[i], upperIndex);
            }
          }
          // Count the number of out-of-bound indices.
          if (outOfBound) {
            Value loadCount = rewriter.create<LoadOp>(loc, outOfBoundCount);
            auto one = emitConstantOp(rewriter, loc, resultElementType, 1);
            if (resultElementType.isa<FloatType>())
              loadCount = rewriter.create<SelectOp>(loc, outOfBound,
                  rewriter.create<AddFOp>(loc, loadCount, one), loadCount);
            else
              loadCount = rewriter.create<SelectOp>(loc, outOfBound,
                  rewriter.create<AddIOp>(loc, loadCount, one), loadCount);
            rewriter.create<StoreOp>(loc, loadCount, outOfBoundCount);
          }
          // TODO (tung): Count the number of pads.
        }

        Value loadData =
            rewriter.create<LoadOp>(loc, inputOperand, dataIndices);
        if (outOfBound)
          loadData =
              rewriter.create<SelectOp>(loc, outOfBound, identity, loadData);
        Value loadResult = rewriter.create<LoadOp>(loc, alloc, resultIndices);
        auto nextResult = mapToLowerScalarOp<PoolOp>(
            op, resultElementType, {loadResult, loadData}, rewriter);
        rewriter.create<StoreOp>(loc, nextResult, alloc, resultIndices);
      }
    }

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

void populateLoweringONNXPoolingOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPoolOpLowering<ONNXMaxPoolSingleOutOp>>(ctx);
  patterns.insert<ONNXPoolOpLowering<ONNXAveragePoolOp>>(ctx);
}
