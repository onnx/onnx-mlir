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

namespace onnx_mlir {

struct ONNXConstantOpLowering : public ConversionPattern {
  ONNXConstantOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXConstantOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = ONNXLoc<ONNXConstantOp>(op);
    auto constantOp = cast<ONNXConstantOp>(op);

    if (constantOp.sparse_value().has_value())
      return emitError(loc, "Only support dense values at this time");

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();

    // Emit the constant global in Krnl dialect.
    MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);
    Value constantGlobal = create.krnl.constant(
        memRefType, "constant_", constantOp.value().value());

    // Replace this operation with the generated krnl.global.
    rewriter.replaceOp(op, constantGlobal);

    return success();
  }
};

void populateLoweringONNXConstantOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
