/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AddOp.cpp - ONNX Op Transform ------------------===//
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
 * ONNX Add operation
 *
 * “Performs element-wise binary addition (with Numpy-style broadcasting
 * support).” “” “This operator supports multidirectional
 * (i.e. Numpy-style) broadcasting; for more details please check the doc.”
 *
 * Operands :
 *    A	  tensor of 32-bit/64-bit unsigned integer values or
 *        tensor of 32-bit/64-bit signless integer values or
 *        tensor of 16-bit/32-bit/64-bit float values or
 *        tensor of bfloat16 type values or memref of any type values
 *  Map this A operand with input parameter in torch side.
 *    B   tensor of 32-bit/64-bit unsigned integer values or
 *        tensor of 32-bit/64-bit signless integer values or
 *        tensor of 16-bit/32-bit/64-bit float values or
 *        tensor of bfloat16 type values or memref of any type values
 *  Map this B operand with other parameter in torch side.
 *
 * Results:
 *   C    tensor of 32-bit/64-bit unsigned integer values or
 *        tensor of 32-bit/64-bit signless integer values or
 *        tensor of 16-bit/32-bit/64-bit float values or
 *        tensor of bfloat16 type values or memref of any type values
 *
 */

class ONNXAddOpToTorchLowering : public ConversionPattern {
public:
  ONNXAddOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXAddOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context = op->getContext();
    ONNXAddOp op1 = llvm::dyn_cast<ONNXAddOp>(op);

    auto a = op1.A();
    auto b = op1.B();
    TensorType aTensorType = a.getType().cast<TensorType>();
    TensorType bTensorType = b.getType().cast<TensorType>();
    TensorType resultTensorType =
	    op->getResult(0).getType().cast<TensorType>();
    auto I64type = IntegerType::get(op1.getContext(), 64);
    auto one = 1;
    auto oneIntAttr = IntegerAttr::get(I64type, one);
    Value alphaDefaultValue = 
	rewriter.create<ConstantIntOp>(loc, oneIntAttr);
    auto aType = Torch::ValueTensorType::get(
	context, aTensorType.getShape(), aTensorType.getElementType());
    auto aTensor = 
	rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>
	(loc, aType, a);
    auto bType = Torch::ValueTensorType::get(context,
	bTensorType.getShape(), bTensorType.getElementType());
    auto bTensor =
	rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>
	(loc, bType, b);
    auto resultType = Torch::ValueTensorType::get(op1.getContext(),
        resultTensorType.getShape(), resultTensorType.getElementType());
    Value result =
        rewriter.create<AtenAddTensorOp>(loc, resultType, aTensor, bTensor,
			alphaDefaultValue);

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);
    return success();
  }
};

void populateLoweringONNXToTorchAddOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXAddOpToTorchLowering>(typeConverter, ctx);
}
