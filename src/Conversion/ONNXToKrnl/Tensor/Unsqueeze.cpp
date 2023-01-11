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

template <typename OP_TYPE>
LogicalResult ONNXUnsqueezeOpLoweringCommon(Operation *op,
    ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
    TypeConverter *typeConverter) {
  typename OP_TYPE::Adaptor operandAdaptor(operands);
  // Op unsqueezeOp = dyn_cast_or_null<Op>(op);

  Location loc = op->getLoc();
  IndexExprBuilderForKrnl createIE(rewriter, loc);
  Value data = operandAdaptor.data();

  // Convert the output type to MemRefType.
  Type convertedType = typeConverter->convertType(*op->result_type_begin());
  assert(convertedType && convertedType.isa<MemRefType>() &&
         "Failed to convert type to MemRefType");

  // Get shape.
  ONNXCommonUnsqueezeOpShapeHelper<OP_TYPE> shapeHelper(
      op, operands, &createIE);
  shapeHelper.computeShapeAndAssertOnFailure();

  // Lower to ReinterpretCastOp so that the data is never copied or modified.
  Value newView = emitMemRefReinterpretCastOp(
      rewriter, loc, data, shapeHelper.getOutputDims(), convertedType);
  rewriter.replaceOp(op, newView);
  return success();
}

struct ONNXUnsqueezeOpLowering : public ConversionPattern {
  ONNXUnsqueezeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXUnsqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXUnsqueezeOpLoweringCommon<ONNXUnsqueezeOp>(
        op, operands, rewriter, typeConverter);
  }
};

struct ONNXUnsqueezeV11OpLowering : public ConversionPattern {
  ONNXUnsqueezeV11OpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXUnsqueezeV11Op::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXUnsqueezeOpLoweringCommon<ONNXUnsqueezeV11Op>(
        op, operands, rewriter, typeConverter);
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
