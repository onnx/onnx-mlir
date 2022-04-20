/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SinOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file lowers Sin op from Onnx to torch
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

#ifdef _WIN32
#include <io.h>
#endif

/*
# ONNX Sin operation

Calculates the sine of the given input tensor, element-wise.
Version

This version of the operator has been available since version 7 of the default ONNX operator set.
Inputs

input (differentiable) : T
    Input tensor

Outputs

output (differentiable) : T
    The sine of the input tensor computed element-wise

Type Constraints

T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.

*/

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

class ONNXSinOpToTorchLowering : public ConversionPattern {
public:
  ONNXSinOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSinOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSinOp sinOp = llvm::dyn_cast_or_null<ONNXSinOp>(op);

    assert(sinOp && "Expecting op to have type ONNXSin");

    Location loc = sinOp.getLoc();
    Value x = sinOp.input();
    mlir::MLIRContext *context = sinOp.getContext();

    TensorType x_tensor_type = x.getType().cast<TensorType>();
    TensorType op_tensor_type = sinOp.getResult().getType().cast<TensorType>();

    auto xTy = Torch::ValueTensorType::get(context,
                                           x_tensor_type.getShape(),
                                           x_tensor_type.getElementType());
    auto xtt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, xTy, x);

    auto resultTy = Torch::ValueTensorType::get(context,
                                                op_tensor_type.getShape(),
                                                op_tensor_type.getElementType());

    llvm::outs() << "Sin input is "
                 << xtt << "\n"
                 << "\n";

    // y = x^0.5
    Value atenSin = rewriter.create<AtenSinOp>(loc, resultTy, xtt);

    llvm::outs() << "ATENSin CREATED is "
                 << atenSin << "\n"
                 << "\n";

    Value result = atenSin;

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTy, result);

    return success();
  }
};

void populateLoweringONNXToTorchSinOpPattern(
    RewritePatternSet &patterns,
    TypeConverter &typeConverter,
    MLIRContext *ctx,
    bool enableTiling
) {
    patterns.insert<ONNXSinOpToTorchLowering>(typeConverter, ctx);
}
