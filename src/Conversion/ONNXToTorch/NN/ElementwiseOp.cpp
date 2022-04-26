/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SqrtOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file lowers most unary operators from torch to onnx using a template
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

    assert(unaryOp && "Expecting op to have a strong type");

    Location loc = unaryOp.getLoc();
    Value operand = unaryOp.getOperand();
    mlir::MLIRContext *context =  unaryOp.getContext();

    auto operandTensorType = operand.getType().template dyn_cast<TensorType>();
    TensorType resultTensorType = unaryOp.getResult().getType().template dyn_cast<TensorType>();

    auto operandType = Torch::ValueTensorType::get(context,
                                           operandTensorType.getShape(),
                                           operandTensorType.getElementType());

    auto operandTensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, operandType, operand);

    auto resultType = Torch::ValueTensorType::get(context,
                                                resultTensorType.getShape(),
                                                resultTensorType.getElementType());

    llvm::outs() << "Unary input is "
                 << operandTensor
                 << "\n";

    Value result = rewriter.create<TorchUnaryOp>(loc, resultType, operandTensor);

    llvm::outs() << "Unary CREATED is "
                 << result
                 << "\n";

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);

    return success();
  }
};

void populateLoweringONNXToTorchElementwiseOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAtanOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCosOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCoshOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXErfOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAcosOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAcoshOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAsinOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAsinhOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAtanhOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXExpOp, AtenExpOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXLogOp, AtenLogOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSignOp>,
      ONNXToTorchElementwiseUnaryOpLowering<ONNXSinOp, AtenSinOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSinhOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSoftsignOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXTanhOp, AtenTanhOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXTanOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXAbsOp, AtenAbsOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXCeilOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXFloorOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXNegOp, AtenNegOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXNotOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXReciprocalOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXReluOp, AtenReluOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXRoundOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSigmoidOp, AtenSigmoidOp>,
      // ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSoftplusOp>,
      ONNXToTorchElementwiseUnaryOpLowering<mlir::ONNXSqrtOp, AtenSqrtOp>>(typeConverter, ctx);
}
