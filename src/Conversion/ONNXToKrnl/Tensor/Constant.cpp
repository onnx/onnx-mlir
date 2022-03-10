/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Constant.cpp - Lowering Constant Op -----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Constant Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXConstantOpLowering : public ConversionPattern {
  ONNXConstantOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXConstantOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = ONNXLoc<ONNXConstantOp>(op);
    auto constantOp = cast<ONNXConstantOp>(op);

    if (constantOp.sparse_value().hasValue())
      return emitError(loc, "Only support dense values at this time");

    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());

    // Shape based computations.
    auto shape = memRefType.getShape();
    int64_t numElements = 1;
    for (size_t i = 0; i < shape.size(); ++i)
      numElements *= shape[i];

    // Emit the constant global in Krnl dialect.
    MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);
    Value constantGlobal = create.krnl.constant(
        memRefType, "constant_", constantOp.value().getValue());

    // Replace this operation with the generated krnl.global.
    rewriter.replaceOp(op, constantGlobal);

    return success();
  }
};

void populateLoweringONNXConstantOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOpLowering>(typeConverter, ctx);
}
