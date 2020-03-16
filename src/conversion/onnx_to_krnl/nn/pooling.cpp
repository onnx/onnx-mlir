//===----- pooling.cpp - Lowering Pooling Ops -----------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Pooling Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/conversion/onnx_to_krnl/onnx_to_krnl_common.hpp"

using namespace mlir;

// Identity values
template <>
Value getIdentityValue<ONNXMaxPoolSingleOutOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return emitNegativeInfinityConstantOp(rewriter, loc, type);
}

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

struct ONNXMaxPoolSingleOutOpLowering : public ConversionPattern {
  ONNXMaxPoolSingleOutOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXMaxPoolSingleOutOp::getOperationName(), 1, ctx) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Match
    ONNXMaxPoolSingleOutOp poolOp = llvm::dyn_cast<ONNXMaxPoolSingleOutOp>(op);

    // Read kernel_shape attribute
    SmallVector<int, 4> kernelShape;
    auto kernelShapeAttribute = poolOp.kernel_shapeAttr();
    for (auto dim : kernelShapeAttribute.getValue())
      kernelShape.emplace_back(dim.cast<IntegerAttr>().getInt());

    // Read strides attribute
    SmallVector<int, 4> strides;
    auto stridesAttribute = poolOp.stridesAttr();
    for (auto stride : stridesAttribute.getValue())
      strides.emplace_back(stride.cast<IntegerAttr>().getInt());

    // Read ceil_mode attribute
    auto ceilMode = poolOp.ceil_mode().getSExtValue();

    // Read pads attribute
    SmallVector<int, 4> pads;
    auto padsAttribute = poolOp.padsAttr();
    for (auto pad : padsAttribute.getValue())
      pads.emplace_back(pad.cast<IntegerAttr>().getInt());

    // Read dilations attribute
    SmallVector<int, 4> dilations;
    auto dilationsAttribute = poolOp.dilationsAttr();
    for (auto dilation : dilationsAttribute.getValue())
      dilations.emplace_back(dilation.cast<IntegerAttr>().getInt());

    // Type information about the input and result of this operation.
    auto &inputOperand = operands[0];
    auto inputShape = inputOperand.getType().cast<MemRefType>().getShape();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto resultShape = memRefType.getShape();
    auto resultElementType = memRefType.getElementType();

    // Batch indices: N and C dimensions
    int batchRank = 2;

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      // Compute dimensions of the result of this operation.
      SmallVector<Value, 2> allocOperands;
      for (int i = 0; i < batchRank; ++i) {
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

      int spatialRank = resultShape.size() - batchRank;
      for (int i = batchRank; i < resultShape.size(); ++i) {
        if (resultShape[i] < 0) {
          // dim =
          //   let numerator = (input + pad - (kernel - 1) * dilation + 1)
          //   in let denomitor = stride
          //      in
          //        if (ceilMode)
          //          ceil(numerator / denominator) + 1
          //        else
          //          floor(numerator / denominator) + 1
          int spatialIndex = i - batchRank;

          // numerator = (input + pad - (kernel - 1) * dilation + 1)
          auto inputDim = rewriter.create<DimOp>(loc, inputOperand, i);
          auto inputVal = rewriter.create<IndexCastOp>(
              loc, inputDim, rewriter.getIntegerType(64));
          int64_t padKernelDilation =
              (pads[spatialIndex] + pads[spatialIndex + spatialRank]) -
              (kernelShape[spatialIndex] - 1) * dilations[spatialIndex] + 1;
          auto padKernelDilationVal = rewriter.create<ConstantOp>(
              loc, rewriter.getIntegerAttr(
                       rewriter.getIntegerType(64), padKernelDilation));
          auto numeratorVal =
              rewriter.create<AddIOp>(loc, inputVal, padKernelDilationVal);
          // denominator
          auto denominatorVal = rewriter.create<ConstantOp>(
              loc, rewriter.getIntegerAttr(
                       rewriter.getIntegerType(64), strides[spatialIndex]));

          // numerator / denominator
          Value dimVal =
              rewriter.create<SignedDivIOp>(loc, numeratorVal, denominatorVal);

          if (ceilMode) {
            auto remainder = rewriter.create<SignedRemIOp>(
                loc, numeratorVal, denominatorVal);
            auto isZero = rewriter.create<CmpIOp>(
                loc, CmpIPredicate::eq, remainder, zero);
            auto dimPlusOne = rewriter.create<AddIOp>(loc, dimVal, one);
            dimVal = rewriter.create<SelectOp>(loc, isZero, dimVal, dimPlusOne);
          }

          dimVal = rewriter.create<AddIOp>(loc, dimVal, one);
          allocOperands.emplace_back(rewriter.create<IndexCastOp>(
              loc, dimVal, rewriter.getIndexType()));
        }
      }
      alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
      if (insertDealloc) {
        auto *parentBlock = alloc.getDefiningOp()->getBlock();
        auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }

    // R = MaxPool(D)
    //
    // The input/output shapes will look like this:
    //
    // D (NxCxHxW) -> R (NxCxRHxRW)
    //
    // The loop nest will look as follows:
    //
    // strides = [s1, s2]
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
    //   k1, k2: inner loop nest indices
    //
    // TODO: handle padding.
    //

    // 1. Define outer loops and emit empty optimization block.
    auto nOuterLoops = resultShape.size();
    BuildKrnlLoop outerLoops(rewriter, loc, nOuterLoops);
    outerLoops.createDefineOptimizeAndIterateOp(alloc);

    rewriter.setInsertionPointToStart(outerLoops.getIterateBlock());
    {
      // 2. Emit the body of the outer loop nest.
      SmallVector<Value, 4> resultIndices;
      for (int i = 0; i < nOuterLoops; ++i)
        resultIndices.emplace_back(outerLoops.getInductionVar(i));

      // 2.1 Emit: R[n][c][r1][r2] = negative_infinity;
      Value identity = getIdentityValue<ONNXMaxPoolSingleOutOp>(
          rewriter, loc, resultElementType);
      rewriter.create<StoreOp>(loc, identity, alloc, resultIndices);

      // 2.2 Define inner loops.
      int nInnerLoops = kernelShape.size();
      BuildKrnlLoop innerLoops(rewriter, loc, nInnerLoops);
      innerLoops.createDefineAndOptimizeOp();
      //   for Kx = 0 .. KX
      for (int i = 0; i < nInnerLoops; ++i)
        innerLoops.pushBounds(0, kernelShape[i]);

      // 2.3 Emit inner loop nest.
      innerLoops.createIterateOp();
      rewriter.setInsertionPointToStart(innerLoops.getIterateBlock());
      {
        // 3. Emit inner loop body
        // t = D[n][c][s1 * r1 + k1][s2 * r2 + k2];
        // R[n][c][r1][r2] = max(R[n][c][r1][r2], t);

        // 3.1 Prepare indices for accesing the data tensor.
        SmallVector<Value, 4> dataIndices;
        // 3.1.1 Batch indices: n, c
        for (int i = 0; i < batchRank; ++i)
          dataIndices.emplace_back(outerLoops.getInductionVar(i));
        // 3.1.2 Insert spatial indices: sX * rX + kX
        // Spatial index may go out of the input index space due to dilation or
        // ceilMode. Hence, we should keep trace of out-of-index.
        Value outOfIndex = nullptr;
        for (int i = batchRank; i < nOuterLoops; ++i) {
          Value spatialIndex = outerLoops.getInductionVar(i);
          // If strides are present (not defulat) then emit the correct access
          // index.
          // sX * rX
          if (strides[i - batchRank] > 1) {
            auto strideIndex = emitConstantOp(
                rewriter, loc, rewriter.getIndexType(), strides[i - batchRank]);
            spatialIndex = rewriter.create<MulIOp>(
                loc, strideIndex, outerLoops.getInductionVar(i));
          }

          // sX * rX + kX
          spatialIndex = rewriter.create<AddIOp>(
              loc, spatialIndex, innerLoops.getInductionVar(i - batchRank));

          // Dilate the kernel index only if the dilation value is not one (not
          // default). If the kernel dimension has a real center, keep the index
          // at the center unchanged/undilated.
          if (dilations[i - batchRank] > 1) {
            bool hasCenter = (kernelShape[i - batchRank] % 2) == 1;
            int64_t centerIndex = (hasCenter)
                                      ? ((kernelShape[i - batchRank] - 1) / 2)
                                      : (kernelShape[i - batchRank] / 2);
            auto centerKernelIndex = emitConstantOp(
                rewriter, loc, rewriter.getIndexType(), centerIndex);
            Value kernelIndex = innerLoops.getInductionVar(i - batchRank);

            Value lessThanCenter, greaterThanCenter;
            if (hasCenter) {
              lessThanCenter = rewriter.create<CmpIOp>(
                  loc, CmpIPredicate::slt, kernelIndex, centerKernelIndex);
              greaterThanCenter = rewriter.create<CmpIOp>(
                  loc, CmpIPredicate::sgt, kernelIndex, centerKernelIndex);
            } else {
              lessThanCenter = rewriter.create<CmpIOp>(
                  loc, CmpIPredicate::slt, kernelIndex, centerKernelIndex);
              greaterThanCenter = rewriter.create<CmpIOp>(
                  loc, CmpIPredicate::sge, kernelIndex, centerKernelIndex);
            }

            auto dilationIndex = emitConstantOp(rewriter, loc,
                rewriter.getIndexType(), dilations[i - batchRank] - 1);
            // Dilate forward.
            auto spatialAddDilationIndex =
                rewriter.create<AddIOp>(loc, spatialIndex, dilationIndex);
            // Dilate backward.
            auto spatialMinusDilationIndex =
                rewriter.create<SubIOp>(loc, spatialIndex, dilationIndex);
            // Switch between forward and backward mode depending relationship
            // between the spatial index and the center index.
            spatialIndex = rewriter.create<SelectOp>(
                loc, lessThanCenter, spatialMinusDilationIndex, spatialIndex);
            spatialIndex = rewriter.create<SelectOp>(
                loc, greaterThanCenter, spatialAddDilationIndex, spatialIndex);
          }

          // Check whether the spatial index goes out of the input index space
          // or not.
          if (dilations[i - batchRank] > 1 or ceilMode) {
            Value lowerIndex =
                emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
            Value upperIndex;
            if (inputShape[i] < 0) {
              Value inputDim = rewriter.create<DimOp>(loc, inputOperand, i);
              Value one = rewriter.create<ConstantIndexOp>(loc, 1);
              upperIndex = rewriter.create<SubIOp>(loc, inputDim, one);
            } else {
              upperIndex =
                  rewriter.create<ConstantIndexOp>(loc, inputShape[i] - 1);
            }
            auto lessThanLowerBound = rewriter.create<CmpIOp>(
                loc, CmpIPredicate::slt, spatialIndex, lowerIndex);
            auto greaterThanUpperBound = rewriter.create<CmpIOp>(
                loc, CmpIPredicate::sgt, spatialIndex, upperIndex);
            if (outOfIndex) {
              auto next = rewriter.create<OrOp>(
                  loc, lessThanLowerBound, greaterThanUpperBound);
              outOfIndex = rewriter.create<OrOp>(loc, outOfIndex, next);
            } else {
              outOfIndex = rewriter.create<OrOp>(
                  loc, lessThanLowerBound, greaterThanUpperBound);
            }
          }

          dataIndices.emplace_back(spatialIndex);
        }

        // 3.2 Do pooling.
        Value loadData =
            rewriter.create<LoadOp>(loc, inputOperand, dataIndices);
        if (outOfIndex)
          loadData =
              rewriter.create<SelectOp>(loc, outOfIndex, identity, loadData);
        auto loadPartialResult =
            rewriter.create<LoadOp>(loc, alloc, resultIndices);
        Value result = mapToLowerScalarOp<ONNXMaxPoolSingleOutOp>(
            op, resultElementType, {loadPartialResult, loadData}, rewriter);
        rewriter.create<StoreOp>(loc, result, alloc, resultIndices);
      }
    }
    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

void populateLoweringONNXPoolingOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXMaxPoolSingleOutOpLowering>(ctx);
}
