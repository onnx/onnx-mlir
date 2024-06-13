/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Unsqueeze.cpp - Lowering Unsqueeze Op ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Unsqueeze Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <typename OP_TYPE, typename OP_ADAPTOR>
LogicalResult ONNXUnsqueezeOpLoweringCommon(OP_TYPE unsqueezeOp,
    OP_ADAPTOR adaptor, ConversionPatternRewriter &rewriter,
    const TypeConverter *typeConverter) {
  Operation *op = unsqueezeOp.getOperation();
  Location loc = ONNXLoc<OP_TYPE>(op);
  ValueRange operands = adaptor.getOperands();

  IndexExprBuilderForKrnl createIE(rewriter, loc);
  Value data = adaptor.getData();

  // Convert the output type to MemRefType.
  Type convertedType = typeConverter->convertType(*op->result_type_begin());
  assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
         "Failed to convert type to MemRefType");

  // Get shape.
  ONNXCommonUnsqueezeOpShapeHelper<OP_TYPE> shapeHelper(
      op, operands, &createIE);
  shapeHelper.computeShapeAndAssertOnFailure();

  // Lower to ReinterpretCastOp so that the data is never copied or modified.
  Value newView = emitMemRefReinterpretCastOp(
      rewriter, loc, data, shapeHelper.getOutputDims(), convertedType);
  rewriter.replaceOp(op, newView);
  onnxToKrnlSimdReport(op);
  return success();
}

struct ONNXUnsqueezeOpLowering : public OpConversionPattern<ONNXUnsqueezeOp> {
  ONNXUnsqueezeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXUnsqueezeOp unsqueezeOp,
      ONNXUnsqueezeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXUnsqueezeOpLoweringCommon<ONNXUnsqueezeOp,
        ONNXUnsqueezeOpAdaptor>(unsqueezeOp, adaptor, rewriter, typeConverter);
  }
};

struct ONNXUnsqueezeV11OpLowering
    : public OpConversionPattern<ONNXUnsqueezeV11Op> {
  ONNXUnsqueezeV11OpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXUnsqueezeV11Op unsqueezeOp,
      ONNXUnsqueezeV11OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXUnsqueezeOpLoweringCommon<ONNXUnsqueezeV11Op,
        ONNXUnsqueezeV11OpAdaptor>(
        unsqueezeOp, adaptor, rewriter, typeConverter);
  }
};

void populateLoweringONNXUnsqueezeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXUnsqueezeOpLowering>(typeConverter, ctx);
}

void populateLoweringONNXUnsqueezeV11OpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXUnsqueezeV11OpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
