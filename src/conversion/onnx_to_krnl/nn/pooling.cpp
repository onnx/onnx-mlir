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
float getIdentityValue<float, ONNXMaxPoolSingleOutOp>(){
  return (float)-std::numeric_limits<float>::infinity();
}

template <>
int getIdentityValue<int, ONNXMaxPoolSingleOutOp>(){
  return std::numeric_limits<int>::min();
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
    if (kernelShapeAttribute)
      for (auto dim : kernelShapeAttribute.getValue())
        kernelShape.emplace_back(dim.cast<IntegerAttr>().getInt());
    else
      emitError(loc, "kernel_shape is a mandatory attribute for which there is "
                     "no default.");

    // Read strides attribute
    SmallVector<int, 4> strides;
    auto stridesAttribute = poolOp.stridesAttr();
    if (stridesAttribute)
      for (auto stride : stridesAttribute.getValue())
        strides.emplace_back(stride.cast<IntegerAttr>().getInt());

    // Read ceil_mode attribute
    auto ceilMode = poolOp.ceil_mode().getSExtValue();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, {operands[0]});

    auto resultShape = memRefType.getShape();
    auto resultElementType = memRefType.getElementType();
    auto &inputOperand = operands[0];
    auto inputShape = inputOperand.getType().cast<MemRefType>().getShape();

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
    //           for k1 = 0 .. KH:
    //             for k2 = 0 .. KW:
    //               t = D[n][c][s1 * r1 + k1][s2 * r2 + k2];
    //               R[n][c][r1][r2] = max(R[n][c][r1][r2], t);
    //
    // Naming:
    //   n, c: outer loop nest indices
    //   r1, r2: spatial loop nest indices
    //   k1, k2: inner loop nest indices
    //
    // TODO: handle padding.
    //

    // 1. Define outer loops and emit empty optimization block:
    int batchRank = 2;  // N and C dimensions
    BuildKrnlLoop outerLoops(rewriter, loc, batchRank);
    outerLoops.createDefineAndOptimizeOp();
    int nIndex = outerLoops.pushBounds(0, inputOperand, 0);
    int cIndex = outerLoops.pushBounds(0, inputOperand, 1);
    // Outer loop iteration
    outerLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outerLoops.getIterateBlock());
    {
      // 2. Emit the body of the outer loop nest.

      // 2.1 Define spatial loops
      int nSpatialLoops = resultShape.size() - batchRank;
      BuildKrnlLoop spatialLoops(rewriter, loc, nSpatialLoops);
      spatialLoops.createDefineAndOptimizeOp();
      for (int i = batchRank; i < resultShape.size(); ++i)
        spatialLoops.pushBounds(0, alloc, i);

      // 2.2 Emit loop nest over output spatial dimensions.
      spatialLoops.createIterateOp();
      rewriter.setInsertionPointToStart(spatialLoops.getIterateBlock());
      {
        // 3. Emit the body of the spatial loop nest.
        // 3.1 Emit: R[n][c][r1][r2] = negative_infinity;
        SmallVector<Value, 4> resultIndices;
        // n
        resultIndices.emplace_back(outerLoops.getInductionVar(nIndex));
        // c
        resultIndices.emplace_back(outerLoops.getInductionVar(cIndex));
        // rX
        for (auto arg : spatialLoops.getIterateBlock()->getArguments())
          resultIndices.emplace_back(arg);
        // Store initializer value into output location.
        Value identity;
        if (resultElementType.isa<FloatType>()) {
          identity = rewriter.create<ConstantOp>(
              loc, FloatAttr::get(resultElementType,
                       getIdentityValue<float, ONNXMaxPoolSingleOutOp>()));
        } else if (resultElementType.isa<IntegerType>()) {
          identity = rewriter.create<ConstantOp>(
              loc, IntegerAttr::get(resultElementType,
                       getIdentityValue<int, ONNXMaxPoolSingleOutOp>()));
        } else {
          emitError(loc, "unsupported element type");
        }
        rewriter.create<StoreOp>(loc, identity, alloc, resultIndices);

        // 3.2 Define inner loops.
        int nInnerLoops = kernelShape.size();
        BuildKrnlLoop innerLoops(rewriter, loc, nInnerLoops);
        innerLoops.createDefineAndOptimizeOp();
        //   for Kx = 0 .. KX
        for (int i = 0; i < nInnerLoops; ++i)
          innerLoops.pushBounds(0, kernelShape[i]);

        // 3.3 Emit inner loop nest.
        innerLoops.createIterateOp();
        rewriter.setInsertionPointToStart(innerLoops.getIterateBlock());

        {
          // 4. Emit inner loop body
          // t = D[n][c][s1 * r1 + k1][s2 * r2 + k2];
          // R[n][c][r1][r2] = max(R[n][c][r1][r2], t);

          // 4.1 Prepare indices for accesing the data tensor.
          SmallVector<Value, 4> dataIndices;
          // n
          dataIndices.emplace_back(outerLoops.getInductionVar(nIndex));
          // c
          dataIndices.emplace_back(outerLoops.getInductionVar(cIndex));
          // sX * rX + kX
          for (int i = 0; i < kernelShape.size(); ++i) {
            Value spatialIndex = spatialLoops.getInductionVar(i);
            // If strides are present then emit the correct access index.
            if (stridesAttribute && strides[i] > 1) {
              spatialIndex = rewriter.create<MulIOp>(loc,
                  rewriter.create<ConstantIndexOp>(loc, strides[i]),
                  spatialLoops.getInductionVar(i));
            }
            spatialIndex = rewriter.create<AddIOp>(
                loc, spatialIndex, innerLoops.getInductionVar(i));
            // If ceil mode is enabled, then the calculated access index may
            // exceed its dimension. In such a case, we will use the maximum
            // index, which causes multiple visits to the element of the
            // maximum index.
            // TODO: Avoid multiple visits.
            if (ceilMode) {
              auto inputIndex = rewriter.create<ConstantIndexOp>(
                  loc, inputShape[batchRank + i] - 1);
              auto greaterCondition = rewriter.create<CmpIOp>(
                  loc, CmpIPredicate::sgt, spatialIndex, inputIndex);
              spatialIndex = rewriter.create<SelectOp>(
                  loc, greaterCondition, inputIndex, spatialIndex);
            }
            dataIndices.emplace_back(spatialIndex);
          }

          // 4.2 Do pooling.
          auto loadData =
              rewriter.create<LoadOp>(loc, inputOperand, dataIndices);
          auto loadPartialResult =
              rewriter.create<LoadOp>(loc, alloc, resultIndices);
          Value result = mapToLowerScalarOp<ONNXMaxPoolSingleOutOp>(
              op, resultElementType, {loadPartialResult, loadData}, rewriter);
          // 4.3 Store computed value into output location.
          rewriter.create<StoreOp>(loc, result, alloc, resultIndices);
        }
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
