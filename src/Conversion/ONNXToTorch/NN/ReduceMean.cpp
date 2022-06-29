/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ReduceMean.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

/*
 * ONNX ReduceMean operation
 *
 * Computes the mean of the input tensor’s element along the provided axes.
 * The resulted” “tensor has the same rank as the input if keepdims equal 1.
 * If keepdims equal 0, then” “the resulted tensor have the reduced
 * dimension pruned. The above behavior is similar to numpy, with the
 * exception that numpy default keepdims to” “False instead of True.
 *
 * Attributes:
 *   axes	::mlir::ArrayAttr	64-bit integer array attribute
 *  keepdims	::mlir::IntegerAttr	64-bit signed integer attribute
 *
 *  Operands:
 *  data	tensor of 32-bit/64-bit unsigned integer values or
 *  		tensor of 32-bit/64-bit signless integer values or
 *  		tensor of 16-bit/32-bit/64-bit float values or
 *  		tensor of bfloat16 type values or memref of any type values.
 *
 *  Results:
 *  reduced	tensor of 32-bit/64-bit unsigned integer values or
 *  		tensor of 32-bit/64-bit signless integer values or
 *  		tensor of 16-bit/32-bit/64-bit float values or tensor
 *  		of bfloat16 type values or memref of any type values
 */

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXReduceMeanOpToTorchLowering : public ConversionPattern {
public:
  ONNXReduceMeanOpToTorchLowering(TypeConverter &typeConverter,
                                  MLIRContext *ctx)
      : ConversionPattern(typeConverter,
                          mlir::ONNXReduceMeanOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    ONNXReduceMeanOp reduceMean = llvm::dyn_cast_or_null<ONNXReduceMeanOp>(op);
    assert(reduceMean && "Expecting op to have a strong type");

    mlir::MLIRContext *context = reduceMean.getContext();
    Location loc = reduceMean.getLoc();

    auto axis = mlir::extractFromI64ArrayAttr(reduceMean.axesAttr());
    if(!(axis.size() == 2 && axis[0] == 2 && axis[1] == 3))
      assert("Not implemented yet for general axis sizes");

    auto keepDims = reduceMean.keepdimsAttr(); // ::mlir::IntegerAttr
    auto keepDimsVal = rewriter.create<ConstantBoolOp>(loc, true);


    auto dataTensor = getTorchTensor(reduceMean.data(), rewriter, context, loc);
    auto resultType = toTorchType(context, reduceMean.getResult().getType());


    auto sintType = IntegerType::get(getContext(), 64, IntegerType::SignednessSemantics::Signed);
    std::vector<Value> axisVal =
        createArrayAttribute(reduceMean.axesAttr(), sintType, loc, rewriter, 1);
    Value axisList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{axisVal});
    
    Value dtype = rewriter.create<ConstantNoneOp>(loc);
    Value result = rewriter.create<AtenMeanDimOp>(loc, resultType, dataTensor, axisList, keepDimsVal, dtype);

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);

    return success();
  }
};

void populateLoweringONNXToTorchReduceMeanOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXReduceMeanOpToTorchLowering>(typeConverter, ctx);
}
