/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- MatMulInteger.cpp - Lowering MatMulInteger Op --------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX MatMulInteger Operator to Krnl dialect.
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

struct ONNXMatMulIntegerOpLowering
    : public OpConversionPattern<ONNXMatMulIntegerOp> {
public:
  ONNXMatMulIntegerOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXMatMulIntegerOp mmiOp,
      ONNXMatMulIntegerOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    using LocalDialectBuilder = MultiDialectBuilder<KrnlBuilder,
        IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder, OnnxBuilder>;
    Operation *op = mmiOp.getOperation();
    Location loc = ONNXLoc<ONNXMatMulIntegerOp>(op);
    LocalDialectBuilder create(rewriter, loc);

    ValueRange operands = adaptor.getOperands();
    Value A = adaptor.getA();
    Value B = adaptor.getB();
    Value aZeroPoint = mmiOp.getAZeroPoint(); // Optional input.
    Value bZeroPoint = mmiOp.getBZeroPoint(); // Optional input.

    if (!isNoneValue(aZeroPoint)) {
      ShapedType stype = aZeroPoint.getType().dyn_cast<ShapedType>();
      if ((stype.getRank() > 1) ||
          (stype.getRank() == 1 && (stype.getShape()[0] != 1))) {
        return rewriter.notifyMatchFailure(op, [&](mlir::Diagnostic &diag) {
          diag << "Does not support non-scalar for a_zero_point";
        });
      }
    }

    // Common types.
    auto resMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(mmiOp.getResult().getType()));
    Type resElementType = resMemRefType.getElementType();

    // Get shape.
    ONNXMatMulIntegerOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Type i32Ty = rewriter.getIntegerType(32);
    Type ui32Ty = rewriter.getIntegerType(32, /*isSigned=*/false);
    assert(resElementType == i32Ty && "Output type is not i32");

    Value AUInt32, BUInt32;
    // Use i32 for onnx.Sub because onnx.Sub does not support ui32.
    // It's because `arith` dialect does not have `subui` though it has `addui`
    // and `mului`.
    if (!isNoneValue(aZeroPoint)) {
      Value AInt32 = create.onnx.cast(A, i32Ty);
      Value aZeroPointInt32 = create.onnx.cast(aZeroPoint, i32Ty);
      // Note: `sub` is not true if aZeroPoint shape is [M], M != 1,
      // because [MxK] - [M] cannot be expressed by `sub`.
      // We require that aZeroPoint is scalar here (checked at the beginning of
      // the lowering).
      AInt32 = create.onnx.sub(AInt32, aZeroPointInt32);
      AUInt32 = create.onnx.cast(AInt32, ui32Ty);
    } else
      AUInt32 = create.onnx.cast(A, ui32Ty);

    if (!isNoneValue(bZeroPoint)) {
      Value BInt32 = create.onnx.cast(B, i32Ty);
      // K is the broadcating dim: [KxN] - [N] = [KxN] - [1xN]
      Value bZeroPointInt32 = create.onnx.cast(bZeroPoint, i32Ty);
      BInt32 = create.onnx.sub(BInt32, bZeroPointInt32);
      BUInt32 = create.onnx.cast(BInt32, ui32Ty);
    } else
      BUInt32 = create.onnx.cast(B, ui32Ty);

    // Emit MatMul for ui32.
    Value res = create.onnx.matmul(
        RankedTensorType::get(resMemRefType.getShape(), ui32Ty), AUInt32,
        BUInt32);
    // Output is i32.
    res = create.onnx.cast(res, resElementType);

    rewriter.replaceOp(op, {create.onnx.toMemref(res)});
    return success();
  }
};

void populateLoweringONNXMatMulIntegerOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulIntegerOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
