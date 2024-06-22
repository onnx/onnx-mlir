/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Squeeze.cpp - Lowering Squeeze Op --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Squeeze Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <typename OP_TYPE, typename OP_ADAPTOR>
LogicalResult ONNXSqueezeOpLoweringCommon(OP_TYPE squeezeOp, OP_ADAPTOR adaptor,
    ConversionPatternRewriter &rewriter, const TypeConverter *typeConverter) {
  Operation *op = squeezeOp.getOperation();
  Location loc = ONNXLoc<OP_TYPE>(op);
  ValueRange operands = adaptor.getOperands();

  IndexExprBuilderForKrnl createIE(rewriter, loc);
  Value data = adaptor.getData();

  // Convert the output type to MemRefType.
  Type convertedType = typeConverter->convertType(*op->result_type_begin());
  assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
         "Failed to convert type to MemRefType");

  // Get shape.
  ONNXCommonSqueezeOpShapeHelper<OP_TYPE> shapeHelper(op, operands, &createIE);
  shapeHelper.computeShapeAndAssertOnFailure();

  // Lower to ReinterpretCastOp so that the data is never copied or modified.
  Value newView = emitMemRefReinterpretCastOp(
      rewriter, loc, data, shapeHelper.getOutputDims(), convertedType);
  rewriter.replaceOp(op, newView);
  onnxToKrnlSimdReport(op);
  return success();
}

struct ONNXSqueezeOpLowering : public OpConversionPattern<ONNXSqueezeOp> {
  ONNXSqueezeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXSqueezeOp squeezeOp,
      ONNXSqueezeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSqueezeOpLoweringCommon<ONNXSqueezeOp, ONNXSqueezeOpAdaptor>(
        squeezeOp, adaptor, rewriter, typeConverter);
  }
};

struct ONNXSqueezeV11OpLowering : public OpConversionPattern<ONNXSqueezeV11Op> {
  ONNXSqueezeV11OpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXSqueezeV11Op squeezeOp,
      ONNXSqueezeV11OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSqueezeOpLoweringCommon<ONNXSqueezeV11Op,
        ONNXSqueezeV11OpAdaptor>(squeezeOp, adaptor, rewriter, typeConverter);
  }
};

void populateLoweringONNXSqueezeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeOpLowering>(typeConverter, ctx);
}

void populateLoweringONNXSqueezeV11OpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeV11OpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
