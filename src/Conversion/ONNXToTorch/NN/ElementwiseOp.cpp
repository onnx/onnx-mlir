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

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Element-wise unary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ONNXUnaryOp, typename TorchUnaryOp>
struct ONNXToTorchElementwiseUnaryOpLowering : public ConversionPattern {
  ONNXToTorchElementwiseUnaryOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXUnaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXUnaryOp unaryOp = llvm::dyn_cast_or_null<ONNXUnaryOp>(op);

    assert(unaryOp && "Expecting op to have type ONNXSqrt");

    Location loc = unaryOp.getLoc();
    Value x = operands[0];
    mlir::MLIRContext *context =  unaryOp.getContext();

    TensorType x_tensor_type = x.getType().cast<TensorType>();
    TensorType op_tensor_type = unaryOp.getResult().getType().template cast<TensorType>();

    auto xTy = Torch::ValueTensorType::get(context,
                                           x_tensor_type.getShape(),
                                           x_tensor_type.getElementType());
    auto xtt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, xTy, x);

    auto resultTy = Torch::ValueTensorType::get(context,
                                                op_tensor_type.getShape(),
                                                op_tensor_type.getElementType());

    llvm::outs() << "Unary input is "
                 << xtt << "\n"
                 << "\n";

    // y = x^0.5
    Value atenUnary = rewriter.create<TorchUnaryOp>(loc, resultTy, xtt);

    llvm::outs() << "Unary CREATED is "
                 << atenUnary << "\n"
                 << "\n";

    Value result = atenUnary;

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTy, result);

    return success();
  }
};

void populateLoweringONNXToTorchElementwiseOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAbsOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAtanOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCastOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCeilOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCosOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCoshOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXEluOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXErfOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAcosOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAcoshOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAsinOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAsinhOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAtanhOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXExpOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXFloorOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXHardSigmoidOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXLeakyReluOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXLogOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXNegOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXNotOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXReciprocalOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXReluOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXRoundOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSeluOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSigmoidOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSignOp>,
      ONNXToTorchElementwiseUnaryOpLowering<ONNXSinOp, AtenSinOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSinhOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSoftplusOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSoftsignOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXTanhOp>,
  //     // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXTanOp>,
      ONNXToTorchElementwiseUnaryOpLowering<ONNXSqrtOp, AtenSqrtOp>>(typeConverter, ctx);
}
