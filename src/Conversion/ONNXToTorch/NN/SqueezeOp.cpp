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

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"


using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXToTorchSqueezeOpLowering : public ConversionPattern {
  ONNXToTorchSqueezeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXSqueezeOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeOp squeezeOp = llvm::dyn_cast_or_null<ONNXSqueezeOp>(op);

    assert(squeezeOp && "Expecting op to have a strong type");

    Location loc = squeezeOp.getLoc();
    Value operandA = squeezeOp.getOperand(0);
    Value operandB = squeezeOp.getOperand(1);
    mlir::MLIRContext *context =  squeezeOp.getContext();

    auto operandAType = toTorchType(context, operandA.getType());
    auto operandBType = toTorchType(context, operandB.getType());
    auto resultType = toTorchType(context, squeezeOp.getResult().getType());

    auto operandATensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, operandAType, operandA);
    auto operandBTensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, operandBType, operandB);

    auto zero = 1;
    auto ty = IntegerType::get(context, 64);
    auto zeroAttr = IntegerAttr::get(ty, zero);
    Value dim = rewriter.create<ConstantIntOp>(loc, zeroAttr);

    llvm::outs() << "Unary input is "
                 << operandATensor
                 << "\n";

    Value result = rewriter.create<AtenSqueezeDimOp>(loc, resultType, operandATensor, dim);

    llvm::outs() << "Unary CREATED is "
                 << result
                 << "\n";

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);

    return success();
  }
};

void populateLoweringONNXToTorchSqueezeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXToTorchSqueezeOpLowering>(typeConverter, ctx);
}
