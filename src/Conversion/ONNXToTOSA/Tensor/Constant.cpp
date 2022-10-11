/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX const operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXConstOpLoweringToTOSA : public OpConversionPattern<ONNXConstantOp> {
public:
  using OpConversionPattern<ONNXConstantOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXConstantOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ::mlir::Attribute valueAttr = *adaptor.value();
    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, resultType, valueAttr);
    return success();
  }
};


} // namespace

void populateLoweringONNXConstOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXConstOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir