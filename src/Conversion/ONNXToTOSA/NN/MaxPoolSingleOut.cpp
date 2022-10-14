/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- MaxPoolSingleOut.cpp - MaxPoolSingleOut Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX MaxpoolSingleOut operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXMaxPoolSingleOutOpLoweringToTOSA : public OpConversionPattern<ONNXMaxPoolSingleOutOp> {
public:
  using OpConversionPattern<ONNXMaxPoolSingleOutOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXMaxPoolSingleOutOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value x = adaptor.X(); // NCHW too ?  
    ArrayAttr kernelShape = adaptor.kernel_shapeAttr(); //Move to NCHW
    ArrayAttr dilations = adaptor.dilationsAttr(); // Errr
    ArrayAttr pads = adaptor.padsAttr();
    ArrayAttr strides = adaptor.stridesAttr(); //Move to NCHW
    int64_t ceilingMode = adaptor.ceil_mode();
    IntegerAttr ceilingModeAttr = adaptor.ceil_modeAttr(); 
    
    std::vector<Value> kernelFinalFormat;
    std::vector<Value> strideFinalFormat;

    


    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<tosa::MaxPool2dOp>(op, resultType, x, kernelShape, strides, pads);
    
    return success();
  }
};

} // namespace

void populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXMaxPoolSingleOutOpLoweringToTOSA>(typeConverter, ctx);
}

} //namespace onnx_mlir