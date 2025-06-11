/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- HammingWindow.cpp - Lowering HammingWindow Op --------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX HammingWindow Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#include <cmath>

using namespace mlir;

namespace onnx_mlir {

struct ONNXHammingWindowOpLowering
    : public OpConversionPattern<ONNXHammingWindowOp> {
  ONNXHammingWindowOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXHammingWindowOp hammingOp,
      ONNXHammingWindowOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = hammingOp.getOperation();
    Location loc = ONNXLoc<ONNXHammingWindowOp>(op);
    Value input = adaptor.getSize();

    auto onnxType = static_cast<onnx::TensorProto_DataType>(
        dyn_cast<ONNXHammingWindowOp>(op).getOutputDatatype());
    mlir::Type outType = convertONNXTypeToMLIRType(rewriter, onnxType);

    // Formula = 0.54347 - 0.45653 * cos(2 * pi * i / (N - 1))
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder, SCFBuilder,
        IndexExprBuilderForKrnl>
        create(rewriter, loc);
    IndexExprScope scope(create.krnl);
    Value windowSizeVal = create.krnl.load(input, ValueRange{});
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    auto memRefType = cast<MemRefType>(convertedType);

    // If periodic, N += 1
    auto periodic = dyn_cast<ONNXHammingWindowOp>(op).getPeriodic();
    auto one = create.math.constant(rewriter.getI32Type(), 1);
    Value computesize =
        periodic ? create.math.add(windowSizeVal, one) : windowSizeVal;

    Value windowSizeForAlloc = create.math.castToIndex(windowSizeVal);
    Value alloc;
    SmallVector<Value, 1> dynamicDimValues;
    if (memRefType.getRank() > 0 && memRefType.isDynamicDim(0)) {
      dynamicDimValues.push_back(windowSizeForAlloc);
    }
    alloc = create.mem.alignedAlloc(memRefType, dynamicDimValues);

    // Constants used in the formula
    Value c0_54 = create.math.constant(outType, 0.54347);
    Value c0_46 = create.math.constant(outType, 0.45653);
    Value two_pi = create.math.constant(outType, 6.283185); // 2 * pi

    // denominator: N - 1
    Value denomF = create.math.sub(computesize, one);

    ValueRange normLoops = create.krnl.defineLoops(1);
    create.krnl.iterateIE(normLoops, normLoops, {LiteralIndexExpr(0)},
        {DimIndexExpr(windowSizeForAlloc)},
        [&](const KrnlBuilder &createKrnl, ValueRange loopIVs) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
          Value i = loopIVs[0];
          Value iFloat = create.math.cast(outType, i);
          Value mulValue = create.math.mul(two_pi, iFloat);
          Value castdemon = create.math.cast(outType, denomF);
          Value angle = create.math.div(mulValue, castdemon);
          //  cos(2 * pi * i / (N - 1))
          Value cosApprox = create.math.cos(angle);
          Value scaled =
              create.math.mul(c0_46, cosApprox); // 0.45653 * cos(angle)
          Value w_i =
              create.math.sub(c0_54, scaled); // 0.54347 - 0.45653 * cos(angle)
          create.krnl.store(w_i, alloc, {i});
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXHammingWindowOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXHammingWindowOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
