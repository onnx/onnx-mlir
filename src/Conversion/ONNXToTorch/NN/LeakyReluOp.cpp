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

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 *
 * ONNX LeakyRelu operation
 *
 * “LeakyRelu takes input data (Tensor) and an argument alpha,
 * and produces one" "output data (Tensor) where the function
 * `f(x) = alpha * x for x < 0`,""`f(x) = x for x >= 0`, is applied to
 * the data tensor elementwise."
 *
 * Operands :
 *   X   tensor of 16-bit/32-bit/64-bit float values or memref of any
 *       type values Output: Y tensor of 16-bit/32-bit/64-bit float
 *       values or memref of any type values
 *
 * Attributes
 * alpha    32-bit float attribute
 *
 * Result:
 *   Y	tensor of 16-bit/32-bit/64-bit float values or memref of
 *      any type values
 *
 * Validation
 * ----------
 * /scripts/docker/build_with_docker.py --external-build --build-dir build
 * --command
 * "build/Ubuntu1804-Release/third-party/onnx-mlir/Release/bin/onnx-mlir
 * --EmitONNXIR --debug --run-torch-pass
 * third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/node/
 * test_leakyrelu/model.onnx"
 *
 */

class ONNXLeakyReluOpToTorchLowering : public ConversionPattern {
public:
  ONNXLeakyReluOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXLeakyReluOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context = op->getContext();
    ONNXLeakyReluOp op1 = llvm::dyn_cast<ONNXLeakyReluOp>(op);
    ONNXLeakyReluOpAdaptor adapter(op1);

    Value x = op1.X();

    auto alpha = adapter.alphaAttr(); // mlir::FloatAttr
    auto negSlope = alpha.getValue(); // APSFloat
    auto negSlopeFloatValue = FloatAttr::get(
        mlir::FloatType::getF64(op->getContext()), negSlope.convertToFloat());
    Value negSlopeConstFloat =
        rewriter.create<ConstantFloatOp>(loc, negSlopeFloatValue);

    TensorType xTensorType = x.getType().cast<TensorType>();
    TensorType opTensorType = op->getResult(0).getType().cast<TensorType>();

    auto xType = Torch::ValueTensorType::get(
        context, xTensorType.getShape(), xTensorType.getElementType());
    auto xTorchTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, xType, x);
    auto resultType = Torch::ValueTensorType::get(op1.getContext(),
        opTensorType.getShape(), opTensorType.getElementType());

    Value result = rewriter.create<AtenLeakyReluOp>(
        loc, resultType, xTorchTensor, negSlopeConstFloat);

    llvm::outs() << "ATENLEAKYRELU CREATED is " << result << "\n";

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);
    return success();
  }
};

void populateLoweringONNXToTorchLeakyReluOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLeakyReluOpToTorchLowering>(typeConverter, ctx);
}
