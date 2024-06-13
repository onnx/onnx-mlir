/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Constant.cpp - Const Op -----------------------------===//
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
    std::optional<Attribute> valueAttr = adaptor.getValue();
    std::optional<Attribute> sparseAttr = adaptor.getSparseValue();
    // Only one of the attributes can/must be present. If
    // sparse is there, value is not present. Currently sparse doesn't seem to
    // be supported by TOSA.
    if (sparseAttr.has_value()) {
      return rewriter.notifyMatchFailure(
          op, "tosa.const does not support sparse value");
    }
    Attribute currentAttr = valueAttr.value();
    if (!mlir::isa<ElementsAttr>(currentAttr)) {
      return rewriter.notifyMatchFailure(
          op, "tosa.const does not support non-tensor types");
    }
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(
        op, resultType, mlir::cast<ElementsAttr>(currentAttr));
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