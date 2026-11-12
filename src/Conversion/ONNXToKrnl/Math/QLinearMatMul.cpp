/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- QLinearMatMul.cpp - Lowering QLinearMatMul Op --------------===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX QLinearMatMul Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXQLinearMatMulOpLowering
    : public OpConversionPattern<ONNXQLinearMatMulOp> {
public:
  ONNXQLinearMatMulOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableSIMD, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx), enableSIMD(enableSIMD),
        enableParallel(
            enableParallel &&
            OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps
                .isEnabled(ONNXQLinearMatMulOp::getOperationName())) {}

  LogicalResult matchAndRewrite(ONNXQLinearMatMulOp qlmmOp,
      ONNXQLinearMatMulOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    using LocalDialectBuilder = MultiDialectBuilder<IndexExprBuilderForKrnl,
        OnnxBuilder, KrnlBuilder, MemRefBuilder, MathBuilder>;
    Operation *op = qlmmOp.getOperation();
    Location loc = ONNXLoc<ONNXQLinearMatMulOp>(op);
    LocalDialectBuilder create(rewriter, loc);

    ValueRange operands = adaptor.getOperands();
    Value A = adaptor.getA();
    Value aScale = adaptor.getAScale();
    Value aZeroPoint = adaptor.getAZeroPoint();
    Value B = adaptor.getB();
    Value bScale = adaptor.getBScale();
    Value bZeroPoint = adaptor.getBZeroPoint();
    Value yScale = adaptor.getYScale();
    Value yZeroPoint = adaptor.getYZeroPoint();

    // Now only support integer8 for inputs and zeropoints, and support float32
    // for scale.
    if (!getElementType(A.getType()).isInteger(8))
      return failure();
    if (!getElementType(B.getType()).isInteger(8))
      return failure();
    if (!getElementType(aScale.getType()).isF32())
      return failure();
    if (!getElementType(bScale.getType()).isF32())
      return failure();
    if (!getElementType(yScale.getType()).isF32())
      return failure();
    if (!getElementType(aZeroPoint.getType()).isInteger(8))
      return failure();
    if (!getElementType(bZeroPoint.getType()).isInteger(8))
      return failure();
    if (!getElementType(yZeroPoint.getType()).isInteger(8))
      return failure();

    // Common types.
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();
    auto resMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(qlmmOp.getResult().getType()));
    Type resElementType = resMemRefType.getElementType();

    // Get shape.
    ONNXQLinearMatMulOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Prepare input A.
    Value AI8 = create.onnx.getOrCastToI8(A);
    Value AI32 = create.onnx.cast(AI8, i32Ty);
    auto aZeroPointType = mlir::cast<ShapedType>(aZeroPoint.getType());
    int64_t aZeroPointRank = aZeroPointType.getRank();
    Value aZeroPointI8 = create.onnx.getOrCastToI8(aZeroPoint);
    Value aZeroPointI32 = create.onnx.cast(aZeroPointI8, i32Ty);
    // If broadcasting, e.g. A is [MxK], zeroPoint is [M], M != 1.
    // Unsqueeze zeroPoint to [Mx1] to make shapes compatible.
    // There is no need to handle scalar zeroPoint (e.g. tensor<dtype> or
    // tensor<1xdtype>), which is always true for broadcasting.
    if ((aZeroPointRank == 1) && (aZeroPointType.getShape()[0] != 1)) {
      SmallVector<int64_t, 4> unsqueezeShape(aZeroPointType.getShape());
      unsqueezeShape.emplace_back(1);
      aZeroPointI32 =
          create.onnx.unsqueeze(RankedTensorType::get(unsqueezeShape, i32Ty),
              aZeroPointI32, create.onnx.constantInt64({aZeroPointRank}));
    }
    AI32 = create.onnx.sub(AI32, aZeroPointI32);

    // Prepare input B.
    Value BI8 = create.onnx.getOrCastToI8(B);
    Value BI32 = create.onnx.cast(BI8, i32Ty);
    // K is the broadcating dim: [KxN] - [N] = [KxN] - [1xN]
    Value bZeroPointI8 = create.onnx.getOrCastToI8(bZeroPoint);
    Value bZeroPointI32 = create.onnx.cast(bZeroPointI8, i32Ty);
    BI32 = create.onnx.sub(BI32, bZeroPointI32);

    // Prepare output Y
    Value yZeroPointI8 = create.onnx.getOrCastToI8(yZeroPoint);
    Value yZeroPointI32 = create.onnx.cast(yZeroPointI8, i32Ty);

    // Emit MatMul.
    Value resI32 = create.onnx.matmul(
        RankedTensorType::get(resMemRefType.getShape(), i32Ty), AI32, BI32);

    // Scale the output.
    Value resF32 = create.onnx.cast(resI32, f32Ty);
    Value scale = create.onnx.div(create.onnx.mul(aScale, bScale), yScale);
    resF32 = create.onnx.mul(resF32, scale);

    // Saturate and add zero point.
    Value roundToEven = create.onnx.round(resF32);
    resI32 = create.onnx.cast(roundToEven, i32Ty);
    SmallVector<Value, 2> finalInputs = {
        create.onnx.toMemref(resI32), create.onnx.toMemref(yZeroPointI32)};
    ONNXBroadcastOpShapeHelper finalShape(op, finalInputs, &create.krnlIE);
    finalShape.computeShapeAndAssertOnFailure();
    DimsExpr outputDims = finalShape.getOutputDims();
    int64_t rank = outputDims.size();
    int64_t alignment = KrnlTypeConverter::getDefaultAllocAlignment(
        qlmmOp.getResult().getType());
    Value output =
        create.mem.alignedAlloc(resMemRefType, outputDims, alignment);
    bool isUnsigned = resElementType.isUnsignedInteger(8);
    Value qMin = create.math.constant(i32Ty, isUnsigned ? 0 : -128);
    Value qMax = create.math.constant(i32Ty, isUnsigned ? 255 : 127);
    Value cst128 = isUnsigned ? create.math.constant(i32Ty, 128) : Value();
    auto emitFinal = [&](const KrnlBuilder &kb, ArrayRef<Value> inputs,
                         int64_t VL) {
      MultiDialectBuilder<MathBuilder> inner(kb);
      Value zeroPoint = inputs[1];
      // Undo getOrCastToI8's unsigned re-centering in the i32 domain.
      if (isUnsigned)
        zeroPoint = inner.math.add(zeroPoint, cst128);
      Value adjusted = inner.math.add(inputs[0], zeroPoint);
      Value clipped = inner.math.clip(adjusted, qMin, qMax);
      Type type =
          VL > 1 ? VectorType::get({VL}, resElementType) : resElementType;
      return inner.math.cast(type, clipped);
    };

    int64_t VL = 1, simdTripCount = 0;
    bool simdOnly = false;
    int64_t collapsedLoops, literalSize;
    IndexExpr dynamicSize;
    if (enableSIMD && rank > 0 && !hasNonIdentityLayout(finalInputs) &&
        finalShape.hasManageableBroadcastForInnerDims(
            collapsedLoops, literalSize, dynamicSize, nullptr)) {
      SmallVector<int64_t, 4> shape;
      IndexExpr::getShape(outputDims, shape);
      auto i32OutputType = MemRefType::get(shape, i32Ty);
      GenOpMix mix = {{GenericOps::ArithmeticGop, isUnsigned ? 2 : 1},
          {GenericOps::MinMaxGop, 2}, {GenericOps::ConversionGop, 1}};
      VL = computeSuitableSimdUnrollFactor(i32OutputType,
          /*collapsedInnermostLoops*/ 1, mix, /*canOverCompute*/ false,
          simdTripCount, simdOnly);
    }
    onnxToKrnlSimdReport(op, VL > 1, VL > 1 ? VL : 0, simdTripCount,
        "QLinearMatMul output adjustment and saturation");

    int64_t loopRank = VL > 1 ? rank - 1 : rank;
    ValueRange loops = create.krnl.defineLoops(loopRank);
    DimsExpr lbs(loopRank, LitIE(0));
    DimsExpr ubs(outputDims.begin(), outputDims.begin() + loopRank);
    auto plan = KrnlParallelPlan::noCollapse(
        loops, /*first*/ 0, /*last excl*/ std::min<int64_t>(2, loopRank));
    if (enableParallel && loopRank > 0)
      plan.tryCreateParallel(
          create.krnl, op, "QLinearMatMul final output", lbs, ubs);
    create.krnl.iterateIE(loops, plan.optimizedLoopDef(), lbs, ubs,
        [&](const KrnlBuilder &kb, ValueRange indices) {
          IndexExprScope innerScope(kb, finalShape.getScope());
          DimsExpr outputAccess = DimListIE(indices);
          if (VL > 1)
            outputAccess.emplace_back(LitIE(0));
          SmallVector<DimsExpr, 2> inputAccess;
          for (int64_t i = 0; i < 2; ++i) {
            DimsExpr access;
            LogicalResult status = finalShape.getAccessExprs(finalInputs[i], i,
                outputAccess, access, /*flattenedInnerDims*/ VL > 1);
            assert(succeeded(status) && "Could not compute final input access");
            inputAccess.emplace_back(access);
          }
          if (VL > 1) {
            kb.simdIterateIE(LitIE(0), SymIE(outputDims.back()), VL, simdOnly,
                /*useParallel*/ false, finalInputs, inputAccess, {output},
                {outputAccess}, {emitFinal});
          } else {
            Value x = kb.loadIE(finalInputs[0], inputAccess[0]);
            Value z = kb.loadIE(finalInputs[1], inputAccess[1]);
            kb.storeIE(emitFinal(kb, {x, z}, 1), output, outputAccess);
          }
        });
    rewriter.replaceOp(op, {output});
    return success();
  }

private:
  bool enableSIMD;
  bool enableParallel;
};

void populateLoweringONNXQLinearMatMulOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD,
    bool enableParallel) {
  patterns.insert<ONNXQLinearMatMulOpLowering>(
      typeConverter, ctx, enableSIMD, enableParallel);
}

} // namespace onnx_mlir
