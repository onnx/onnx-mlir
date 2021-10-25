/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Squeeze.cpp - Lowering Squeeze Op --------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Squeeze Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

template <typename Adaptor, typename Op, typename ShapeHelper>
LogicalResult ONNXSqueezeOpLoweringCommon(Operation *op,
    ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) {
  Adaptor operandAdaptor(operands);
  Op squeezeOp = dyn_cast_or_null<Op>(op);

  auto loc = op->getLoc();
  auto memRefType = convertToMemRefType(*op->result_type_begin());
  Value data = operandAdaptor.data();

  ShapeHelper shapeHelper(&squeezeOp, &rewriter,
      getDenseElementAttributeFromKrnlValue, loadDenseElementArrayValueAtIndex);
  auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
  assert(succeeded(shapecomputed));

  // Lower to ReinterpretCastOp so that the data is never copied or modified.
  Value newView = emitMemRefReinterpretCastOp(
      rewriter, loc, data, memRefType, shapeHelper.dimsForOutput(0));
  rewriter.replaceOp(op, newView);
  return success();
}

struct ONNXSqueezeOpLowering : public ConversionPattern {
  ONNXSqueezeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSqueezeOpLoweringCommon<ONNXSqueezeOpAdaptor, ONNXSqueezeOp,
        ONNXSqueezeOpShapeHelper>(op, operands, rewriter);
  }
};

struct ONNXSqueezeV11OpLowering : public ConversionPattern {
  ONNXSqueezeV11OpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSqueezeV11Op::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSqueezeOpLoweringCommon<ONNXSqueezeV11OpAdaptor,
        ONNXSqueezeV11Op, ONNXSqueezeV11OpShapeHelper>(op, operands, rewriter);
  }
};

void populateLoweringONNXSqueezeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeOpLowering>(ctx);
}

void populateLoweringONNXSqueezeV11OpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeV11OpLowering>(ctx);
}
