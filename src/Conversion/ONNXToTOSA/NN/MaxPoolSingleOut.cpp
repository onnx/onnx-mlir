/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Maxpool.cpp - Maxpool Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX Maxpool operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXMaxPoolSingleOutOpLoweringToTOSA : public OpConversionPattern<ONNXMaxPoolSingleOutOp> {
public:
  using OpConversionPattern<ONNXMaxPoolSingleOutOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXMaxPoolSingleOutOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    MLIRContext *ctx = op->getContext();

    Value Input = adaptor.X();
    mlir::StringAttr autoPad = adaptor.auto_padAttr();
    //mlir::IntegerAttr ceilMode = adaptor.ceil_modeAttr();
    //mlir::IntegerAttr storage = adaptor.storage_orderAttr();
    //mlir::ArrayAttr dilations = adaptor.dilationsAttr();
    mlir::ArrayAttr kernelShape = adaptor.kernel_shapeAttr();
    mlir::ArrayAttr pads = adaptor.padsAttr();
    mlir::ArrayAttr strides = adaptor.stridesAttr();

    auto InputType = Input.getType().cast<RankedTensorType>();
    auto resultType = op.getResult().getType();

    if (autoPad != "NOTSET") {
      // LOG unsupported because deprecated? 
    }
    // ONNX Mlir uses NCHW as an input while TOSA expects NHWC. Insert a transpose
    // to change the format
    Value targetTensor = mlir::tosa::getConstTensor<int32_t>(rewriter, op, {0, 2, 3, 1}, {3}).value();
    Type outputType = RankedTensorType::get({-1, -1, -1, -1}, InputType.getElementType());
    Input = tosa::CreateOpAndInfer<tosa::TransposeOp>(rewriter, loc, outputType, Input, targetTensor).getResult();

    tosa::CreateReplaceOpAndInfer<tosa::MaxPool2dOp>(rewriter, op, resultType, Input, kernelShape, strides, pads);
    return success();
  }
};


} // namespace

void populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMaxPoolSingleOutOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir