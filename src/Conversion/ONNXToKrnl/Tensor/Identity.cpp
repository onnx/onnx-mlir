/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Identity.cpp - Lowering Identity Op ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNXIdentity operator to the Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXIdentityOpLowering : public ConversionPattern {
  ONNXIdentityOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXIdentityOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXIdentityOpAdaptor operandAdaptor(operands);
    rewriter.replaceOp(op, operandAdaptor.input());
    return success();
  }
};

void populateLoweringONNXIdentityOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXIdentityOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
