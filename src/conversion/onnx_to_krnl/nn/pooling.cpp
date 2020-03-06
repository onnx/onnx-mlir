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
float getIdentityValue<float, ONNXMaxPoolSingleOutNoPadsOp>() {
  return (float)-std::numeric_limits<float>::infinity();
}

template <>
int getIdentityValue<int, ONNXMaxPoolSingleOutNoPadsOp>() {
  return std::numeric_limits<int>::min();
}

template <>
Value mapToLowerScalarOp<ONNXMaxPoolSingleOutNoPadsOp>(Operation *op,
    ArrayRef<Type> result_types, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  Value lhs = operands[0];
  Value rhs = operands[1];
  auto max = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, max, lhs, rhs);
  return result;
}

struct ONNXMaxPoolSingleOutNoPadsOpLowering : public ConversionPattern {
  ONNXMaxPoolSingleOutNoPadsOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXMaxPoolSingleOutNoPadsOp::getOperationName(), 1, ctx) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Match
    ONNXMaxPoolSingleOutNoPadsOp poolOp =
        llvm::dyn_cast<ONNXMaxPoolSingleOutNoPadsOp>(op);

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
      Value identity;
      if (resultElementType.isa<FloatType>()) {
        identity = rewriter.create<ConstantOp>(
            loc, FloatAttr::get(resultElementType,
                     getIdentityValue<float, ONNXMaxPoolSingleOutNoPadsOp>()));
      } else if (resultElementType.isa<IntegerType>()) {
        identity = rewriter.create<ConstantOp>(
            loc, IntegerAttr::get(resultElementType,
                     getIdentityValue<int, ONNXMaxPoolSingleOutNoPadsOp>()));
      } else {
        emitError(loc, "unsupported element type");
      }
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
        // Batch indices: n, c
        for (int i = 0; i < batchRank; ++i)
          dataIndices.emplace_back(outerLoops.getInductionVar(i));
        // Spatial indices: sX * rX + kX
        for (int i = batchRank; i < nOuterLoops; ++i) {
          Value spatialIndex = outerLoops.getInductionVar(i);
          // If strides are present then emit the correct access index.
          if (stridesAttribute && strides[i - batchRank] > 1) {
            spatialIndex = rewriter.create<MulIOp>(loc,
                rewriter.create<ConstantIndexOp>(loc, strides[i - batchRank]),
                outerLoops.getInductionVar(i));
          }
          spatialIndex = rewriter.create<AddIOp>(
              loc, spatialIndex, innerLoops.getInductionVar(i - batchRank));
          // If ceil mode is enabled, then the calculated access index may
          // exceed its dimension. In such a case, we will use the maximum
          // index, which causes multiple visits to the element of the
          // maximum index.
          // TODO: Avoid multiple visits.
          if (ceilMode) {
            Value inputIndex;
            if (inputShape[i] < 0) {
              Value inputDim = rewriter.create<DimOp>(loc, inputOperand, i);
              Value one = rewriter.create<ConstantIndexOp>(loc, 1);
              inputIndex = rewriter.create<SubIOp>(loc, inputDim, one);
            } else {
              inputIndex =
                  rewriter.create<ConstantIndexOp>(loc, inputShape[i] - 1);
            }
            auto greaterCondition = rewriter.create<CmpIOp>(
                loc, CmpIPredicate::sgt, spatialIndex, inputIndex);
            spatialIndex = rewriter.create<SelectOp>(
                loc, greaterCondition, inputIndex, spatialIndex);
          }
          dataIndices.emplace_back(spatialIndex);
        }

        // 3.2 Do pooling.
        auto loadData = rewriter.create<LoadOp>(loc, inputOperand, dataIndices);
        auto loadPartialResult =
            rewriter.create<LoadOp>(loc, alloc, resultIndices);
        Value result = mapToLowerScalarOp<ONNXMaxPoolSingleOutNoPadsOp>(
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
  patterns.insert<ONNXMaxPoolSingleOutNoPadsOpLowering>(ctx);
}
