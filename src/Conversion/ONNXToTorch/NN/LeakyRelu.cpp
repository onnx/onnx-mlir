/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- LeakyReluOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// ========================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===-----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;

// ONNX LeakyRelu operation
//
// LeakyRelu takes input data (Tensor) and an argument alpha,
// and produces one" "output data (Tensor) where the function
// `f(x) = alpha * x for x < 0`, `f(x) = x for x >= 0`, is applied to
// the data tensor elementwise.
//
// Operands :
//   X   tensor of 16-bit/32-bit/64-bit float values or memref of any
//       type values Output: Y tensor of 16-bit/32-bit/64-bit float
//       values or memref of any type values
//
// Attributes
// alpha    32-bit float attribute
//
// Result:
//   Y	tensor of 16-bit/32-bit/64-bit float values or memref of
//      any type values
//
// Validation
// ----------
// /scripts/docker/build_with_docker.py --external-build --build-dir build
// --command
// "build/Ubuntu1804-Release/third-party/onnx-mlir/Release/bin/onnx-mlir
// --EmitONNXIR --debug --run-torch-pass
// third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/node/
// test_leakyrelu/model.onnx"
//
class ONNXLeakyReluOpToTorchLowering
    : public OpConversionPattern<ONNXLeakyReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXLeakyReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value x = adaptor.X();
    FloatAttr alpha = adaptor.alphaAttr();
    FloatAttr negSlopeFloatAttr = convertToIEEEDouble(alpha);
    Value negSlopeConstFloat =
        rewriter.create<Torch::ConstantFloatOp>(loc, negSlopeFloatAttr);

    rewriter.replaceOpWithNewOp<Torch::AtenLeakyReluOp>(op,
        typeConverter->convertType(op.getResult().getType()), x,
        negSlopeConstFloat);
    return success();
  }
};

void populateLoweringONNXToTorchLeakyReluOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLeakyReluOpToTorchLowering>(typeConverter, ctx);
}
