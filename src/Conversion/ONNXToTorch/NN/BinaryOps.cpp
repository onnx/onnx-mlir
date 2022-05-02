/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SqrtOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file lowers most binary operators from torch to onnx using a template
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Element-wise binary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ONNXBinaryOp, typename TorchBinaryOp>
struct ONNXToTorchElementwiseBinaryOpLowering : public ConversionPattern {
  ONNXToTorchElementwiseBinaryOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXBinaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXBinaryOp binaryOp = llvm::dyn_cast_or_null<ONNXBinaryOp>(op);

    assert(binaryOp && "Expecting op to have a strong type");

    Location loc = binaryOp.getLoc();

    Value operandA = binaryOp.getOperand(0);
    Value operandB = binaryOp.getOperand(1);

    mlir::MLIRContext *context =  binaryOp.getContext();

    auto operandAType = toTorchType(context, operandA.getType());
    auto operandBType = toTorchType(context, operandB.getType());
    auto resultType = toTorchType(context, binaryOp.getResult().getType());

    auto operandATensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, operandAType, operandA);
    auto operandBTensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, operandBType, operandB);

    llvm::outs() << "Binary input is "
                 << operandATensor
                 << "\n"
                 << operandBTensor
                 << "\n";

    Value result = rewriter.create<TorchBinaryOp>(loc,
                                                  resultType,
                                                  operandATensor,
                                                  operandBTensor);

    llvm::outs() << "Binary CREATED is "
                 << result
                 << "\n";

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);

    return success();
  }
};

void populateLoweringONNXToTorchBinaryOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<
    ONNXToTorchElementwiseBinaryOpLowering<mlir::ONNXMatMulOp, AtenMatmulOp>>(typeConverter, ctx);
}
