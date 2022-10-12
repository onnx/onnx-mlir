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
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

  // Apply the transposeDims vector on input to generate a transposed form.
  Value transposeTensor(ONNXMaxPoolSingleOutOp op, ConversionPatternRewriter &rewriter,
                        Value input, ArrayRef<int32_t> transposeDims) const {
    auto inputTy = input.getType().template cast<RankedTensorType>();
    auto inputElemTy = inputTy.getElementType();
    auto inputShape = inputTy.getShape();
    auto inputRank = inputTy.getRank();

    llvm::Optional<Value> transposeDimsConst = tosa::getConstTensor<int32_t>(
        rewriter, op,
        /*vec=*/transposeDims,
        /*shape=*/{static_cast<int32_t>(inputRank)});

    SmallVector<int64_t> transposedInputShape;
    for (auto &dim : transposeDims)
      transposedInputShape.push_back(inputShape[dim]);
    auto transposedInputType =
        RankedTensorType::get(transposedInputShape, inputElemTy);
    return rewriter
        .create<tosa::TransposeOp>(op->getLoc(), transposedInputType, input,
                                   transposeDimsConst.value())
        .getResult();
  }

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