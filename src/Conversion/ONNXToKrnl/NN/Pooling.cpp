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
Value emitScalarOpFor<ONNXMaxPoolSingleOutOp>(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    Type elementType, ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  auto max = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, max, lhs, rhs);
  return result;
}

struct ONNXMaxPoolSingleOutOpLowering : public ConversionPattern {
  ONNXMaxPoolSingleOutOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXMaxPoolSingleOutOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXMaxPoolSingleOutOpOperandAdaptor operandAdaptor(operands);
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
    auto inputOperand = operandAdaptor.X();
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
          //   let numerator = (input + pad - (kernel - 1) * dilation - 1)
          //   in let denomitor = stride
          //      in
          //        if (ceilMode)
          //          ceil(numerator / denominator) + 1
          //        else
          //          floor(numerator / denominator) + 1
          int spatialIndex = i - batchRank;

          // numerator = (input + pad - (kernel - 1) * dilation - 1)
          auto inputDim = rewriter.create<DimOp>(loc, inputOperand, i);
          auto inputVal = rewriter.create<IndexCastOp>(
              loc, inputDim, rewriter.getIntegerType(64));
          int64_t padKernelDilation =
              (pads[spatialIndex] + pads[spatialIndex + spatialRank]) -
              (kernelShape[spatialIndex] - 1) * dilations[spatialIndex] - 1;
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
    //             t = D[n][c][s1 * r1 + k1 * d1][s2 * r2 + k2 * d2];
    //             R[n][c][r1][r2] = max(R[n][c][r1][r2], t);
    //
    // Naming:
    //   n, c, r1, r2: outer loop nest indices
    //   k1, k2: inner loop nest indices
    //   s1, s2: strides
    //   d1, d2: dilations
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
        // t = D[n][c][s1 * r1 + k1 * d1][s2 * r2 + k2 * d2];
        // R[n][c][r1][r2] = max(R[n][c][r1][r2], t);

        // 3.1 Prepare indices for accesing the data tensor.
        SmallVector<Value, 4> dataIndices;
        // 3.1.1 Batch indices: n, c
        for (int i = 0; i < batchRank; ++i)
          dataIndices.emplace_back(outerLoops.getInductionVar(i));
        // 3.1.2 Insert spatial indices: sX * rX + kX * dX
        for (int i = batchRank; i < nOuterLoops; ++i) {
          // Get index along the inner loop's induction variables.
          // It is used to obtain kernel/pad/stride/dilation index.
          int j = i - batchRank;

          Value spatialIndex = outerLoops.getInductionVar(i);
          // If strides are present (not default) then emit the correct access
          // index.
          // sX *= rX
          if (strides[i - batchRank] > 1) {
            auto strideIndex = emitConstantOp(
                rewriter, loc, rewriter.getIndexType(), strides[j]);
            spatialIndex = rewriter.create<MulIOp>(
                loc, strideIndex, outerLoops.getInductionVar(i));
          }

          // Dilate the kernel index only if the dilation value is not one (not
          // default). Otherwise, just add kernelIndex.
          auto kernelIndex = innerLoops.getInductionVar(j);
          if (dilations[j] > 1) {
            // sX += dX * kW
            auto dilationIndex = emitConstantOp(
                rewriter, loc, rewriter.getIndexType(), dilations[j]);
            auto dilationKernelIndex =
                rewriter.create<MulIOp>(loc, dilationIndex, kernelIndex);
            spatialIndex =
                rewriter.create<AddIOp>(loc, spatialIndex, dilationKernelIndex);
          } else {
            // sX += kX
            spatialIndex =
                rewriter.create<AddIOp>(loc, spatialIndex, kernelIndex);
          }

          // If ceil mode or dilation is enabled, then the calculated access
          // index may exceed its dimension. In such a case, we will use the
          // maximum index, which causes multiple visits to the element of the
          // maximum index.
          // TODO: Avoid multiple visits.
          // Example of out-of-bound.
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
          if (dilations[j] > 1 or ceilMode) {
            Value upperIndex;
            if (inputShape[i] < 0) {
              Value inputDim = rewriter.create<DimOp>(loc, inputOperand, i);
              Value one = rewriter.create<ConstantIndexOp>(loc, 1);
              upperIndex = rewriter.create<SubIOp>(loc, inputDim, one);
            } else {
              upperIndex =
                  rewriter.create<ConstantIndexOp>(loc, inputShape[i] - 1);
            }
            auto greaterCondition = rewriter.create<CmpIOp>(
                loc, CmpIPredicate::sgt, spatialIndex, upperIndex);
            spatialIndex = rewriter.create<SelectOp>(
                loc, greaterCondition, upperIndex, spatialIndex);
          }

          dataIndices.emplace_back(spatialIndex);
        }

        // 3.2 Do pooling.
        auto loadData = rewriter.create<LoadOp>(loc, inputOperand, dataIndices);
        auto loadPartialResult =
            rewriter.create<LoadOp>(loc, alloc, resultIndices);
        Value result = emitScalarOpFor<ONNXMaxPoolSingleOutOp>(rewriter, loc,
            op, resultElementType, {loadPartialResult, loadData});
        rewriter.create<StoreOp>(loc, result, alloc, resultIndices);
      }
    }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXPoolingOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXMaxPoolSingleOutOpLowering>(ctx);
}
