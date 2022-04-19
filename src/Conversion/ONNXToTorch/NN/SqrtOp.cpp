/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SqrtOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file lowers Sqrt op from Onnx to torch
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

#ifdef _WIN32
#include <io.h>
#endif

/*
# ONNX Sqrt operation

Square root takes one input data (Tensor) and produces one output data (Tensor) where the square root is, y = x^0.5, is applied to the tensor elementwise. If x is negative, then it will return NaN.
Version

This version of the operator has been available since version 13 of the default ONNX operator set.

Other versions of this operator: 1, 6
# Inputs

X (differentiable) : T
    Input tensor

# Outputs

Y (differentiable) : T
    Output tensor

# Type Constraints

T : tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
    Constrain input and output types to float tensors.
*/

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

class ONNXSqrtOpToTorchLowering : public ConversionPattern {
public:
  ONNXSqrtOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSqrtOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSqrtOp sqrtOp = llvm::dyn_cast_or_null<ONNXSqrtOp>(op);

    assert(sqrtOp && "Expecting op to have type ONNXSqrt");

    Location loc = sqrtOp.getLoc();
    Value x = sqrtOp.X();
    mlir::MLIRContext *context =  sqrtOp.getContext();

    TensorType x_tensor_type = x.getType().cast<TensorType>();
    TensorType op_tensor_type = sqrtOp.getResult().getType().cast<TensorType>();

    auto xTy = Torch::ValueTensorType::get(context,
                                           x_tensor_type.getShape(),
                                           x_tensor_type.getElementType());
    auto xtt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, xTy, x);

    auto resultTy = Torch::ValueTensorType::get(context,
                                                op_tensor_type.getShape(),
                                                op_tensor_type.getElementType());

    llvm::outs() << "Sqrt input is "
                 << xtt << "\n"
                 << "\n";

    // y = x^0.5
    Value atenSqrt = rewriter.create<AtenSqrtOp>(loc, resultTy, xtt);

    llvm::outs() << "ATENSqrt CREATED is "
                 << atenSqrt << "\n"
                 << "\n";

    Value result = atenSqrt;

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTy, result);

    return success();
  }
};

void populateLoweringONNXToTorchSqrtOpPattern(
    RewritePatternSet &patterns,
    TypeConverter &typeConverter,
    MLIRContext *ctx,
    bool enableTiling
) {
    patterns.insert<ONNXSqrtOpToTorchLowering>(typeConverter, ctx);
}
