/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Constant.cpp - Const Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX const operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
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
    auto valueAttr = adaptor.value();
    auto sparseAttr = adaptor.sparse_value();
    // Only one of the attributes can be present and one must be present.
    ::mlir::Attribute currentAttr = valueAttr.has_value() ? valueAttr.value() : sparseAttr.value();
    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (currentAttr.isa<ElementsAttr>()) {
      rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, resultType, currentAttr);
      return success();
    }
    return failure();
  }
};


} // namespace

void populateLoweringONNXConstOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXConstOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir