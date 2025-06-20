/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Window.cpp - Lowering Windows Op --------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Windows Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#include <functional>

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Window Compute Helper Templates
//===----------------------------------------------------------------------===//

template <typename ONNXOp>
Value emitWindowCompute(MultiDialectBuilder<KrnlBuilder, MathBuilder> &create,
    Value iFloat, Value denomF, Type outType);

template <>
Value emitWindowCompute<ONNXHammingWindowOp>(
    MultiDialectBuilder<KrnlBuilder, MathBuilder> &create, Value iFloat,
    Value denomF, Type outType) {
  // Formula: 0.54347 - 0.45653 * cos(2 * pi * i / (N - 1))
  Value c0_54347 = create.math.constant(outType, 0.54347);
  Value c0_45653 = create.math.constant(outType, 0.45653);
  Value c2pi = create.math.constant(outType, 2.0 * M_PI);
  Value num = create.math.mul(c2pi, iFloat);
  Value angle = create.math.div(num, denomF);
  Value cosAngle = create.math.cos(angle); // cos(2 * pi * i / (N - 1))
  Value term = create.math.mul(c0_45653, cosAngle);
  return create.math.sub(c0_54347, term);
}

template <>
Value emitWindowCompute<ONNXBlackmanWindowOp>(
    MultiDialectBuilder<KrnlBuilder, MathBuilder> &create, Value iFloat,
    Value denomF, Type outType) {
  // Formula: 0.42 - 0.5*cos(2*pi*i/(N-1)) + 0.08*cos(4*pi*i/(N-1))
  Value c0_42 = create.math.constant(outType, 0.42);
  Value c0_5 = create.math.constant(outType, 0.5);
  Value c0_08 = create.math.constant(outType, 0.08);
  Value c2pi = create.math.constant(outType, 2.0 * M_PI);
  Value c4pi = create.math.constant(outType, 4.0 * M_PI);

  Value num1 = create.math.mul(c2pi, iFloat);
  Value angle1 = create.math.div(num1, denomF);
  Value cosAngle1 = create.math.cos(angle1);      // cos(2*pi*i/(N-1)
  Value term1 = create.math.mul(c0_5, cosAngle1); // 0.5*cos(2*pi*i/(N-1))

  Value num2 = create.math.mul(c4pi, iFloat);
  Value angle2 = create.math.div(num2, denomF);
  Value cosAngle2 = create.math.cos(angle2);       // cos(4*pi*i/(N-1)
  Value term2 = create.math.mul(c0_08, cosAngle2); // 0.08*cos(4*pi*i/(N-1))

  Value result = create.math.sub(c0_42, term1);
  return create.math.add(result, term2);
}

//===----------------------------------------------------------------------===//

template <typename ONNXOp>
struct GenericWindowOpLowering : public OpConversionPattern<ONNXOp> {
  using OpConversionPattern<ONNXOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXOp windowOp,
      typename ONNXOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    Operation *op = windowOp.getOperation();
    Location loc = ONNXLoc<ONNXOp>(op);
    Value input = adaptor.getSize();

    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder, SCFBuilder,
        IndexExprBuilderForKrnl>
        create(rewriter, loc);

    // Get output data type from the op's attribute.
    auto onnxType =
        static_cast<onnx::TensorProto_DataType>(windowOp.getOutputDatatype());
    mlir::Type outType = convertONNXTypeToMLIRType(rewriter, onnxType);

    IndexExprScope scope(create.krnl);

    // Get window size from the input tensor.
    Value windowSizeVal = create.krnl.load(input, ValueRange{});
    Type convertedType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    auto memRefType = cast<MemRefType>(convertedType);

    // If periodic, N += 1
    auto periodic = windowOp.getPeriodic();
    auto oneI32 = create.math.constant(rewriter.getI32Type(), 1);
    Value computeSize =
        periodic ? create.math.add(windowSizeVal, oneI32) : windowSizeVal;

    // Allocate the output tensor of size N.
    Value windowSizeForAlloc = create.math.castToIndex(windowSizeVal);
    Value alloc;
    SmallVector<Value, 1> dynamicDimValues;
    if (memRefType.getRank() > 0 && memRefType.isDynamicDim(0)) {
      dynamicDimValues.push_back(windowSizeForAlloc);
    }
    alloc = create.mem.alignedAlloc(memRefType, dynamicDimValues);
    // denominator = (computeSize - 1) .
    Value computeSizeF = create.math.cast(outType, computeSize);
    Value oneF = create.math.constant(outType, 1.0);
    Value denomF = create.math.sub(computeSizeF, oneF);

    ValueRange loops = create.krnl.defineLoops(1);
    create.krnl.iterateIE(loops, loops, {LiteralIndexExpr(0)},
        {DimIndexExpr(windowSizeForAlloc)},
        [&](const KrnlBuilder &krnlBuilder, ValueRange loopIVs) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(krnlBuilder);
          Value i = loopIVs[0];
          Value iFloat = create.math.cast(outType, i);
          Value w_i =
              emitWindowCompute<ONNXOp>(create, iFloat, denomF, outType);
          krnlBuilder.store(w_i, alloc, {i});
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

using ONNXHammingWindowOpLowering =
    GenericWindowOpLowering<ONNXHammingWindowOp>;
using ONNXBlackmanWindowOpLowering =
    GenericWindowOpLowering<ONNXBlackmanWindowOp>;

void populateLoweringONNXWindowOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXHammingWindowOpLowering>(typeConverter, ctx);
  patterns.insert<ONNXBlackmanWindowOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir