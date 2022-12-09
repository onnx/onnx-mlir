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

template <typename OP_TYPE>
LogicalResult ONNXSqueezeOpLoweringCommon(Operation *op,
    ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
    TypeConverter *typeConverter) {
  typename OP_TYPE::Adaptor operandAdaptor(operands);

  Location loc = op->getLoc();
  IndexExprBuilderForKrnl createIE(rewriter, loc);
  Value data = operandAdaptor.data();

  // Convert the output type to MemRefType.
  Type convertedType = typeConverter->convertType(*op->result_type_begin());
  assert(convertedType && convertedType.isa<MemRefType>() &&
         "Failed to convert type to MemRefType");

  // Get shape.
  ONNXCommonSqueezeOpShapeHelper<OP_TYPE> shapeHelper(op, operands, &createIE);
  shapeHelper.computeShapeAndAssertOnFailure();

  // Lower to ReinterpretCastOp so that the data is never copied or modified.
  Value newView = emitMemRefReinterpretCastOp(
      rewriter, loc, data, shapeHelper.getOutputDims(), convertedType);
  rewriter.replaceOp(op, newView);
  return success();
}

struct ONNXSqueezeOpLowering : public ConversionPattern {
  ONNXSqueezeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSqueezeOpLoweringCommon<ONNXSqueezeOp>(
        op, operands, rewriter, typeConverter);
  }
};

struct ONNXSqueezeV11OpLowering : public ConversionPattern {
  ONNXSqueezeV11OpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXSqueezeV11Op::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSqueezeOpLoweringCommon<ONNXSqueezeV11Op>(
        op, operands, rewriter, typeConverter);
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
