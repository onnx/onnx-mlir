/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Softmax.cpp - Softmax Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXSoftmaxLoweringToTOSA : public OpConversionPattern<ONNXSoftmaxOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXSoftmaxOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXSoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // softmax = exp(logits) / reduce_sum(exp(logits), -1)
    RankedTensorType outputType =
        op.getResult().getType().dyn_cast<RankedTensorType>();
    RankedTensorType inputType =
        adaptor.input().getType().dyn_cast<RankedTensorType>();
    IntegerAttr axisAttr = adaptor.axisAttr();

    // reduce_sum on last dimension
    int32_t inputRank = inputType.getShape().size();

    // Get ONNX softmax axis
    int axis = axisAttr.getSInt();
    // Tosa only supports positive values
    if (axis < 0) {
      axis += inputRank;
    }
    // The legalization below is based on convertSoftmaxOp in
    // tensorflow tosa/transforms/legalize_common.cc, with the
    // addition of handling for axis.

    // Not a ranked tensor input/output
    if (!outputType || !inputType) {
      return rewriter.notifyMatchFailure(
          op, "input and result not ranked tensors");
    }

    SmallVector<int64_t> rsumShapeV(
        inputType.getShape().begin(), inputType.getShape().end());
    // Differs from TF
    rsumShapeV[axis] = 1;
    ArrayRef<int64_t> rsumShape(rsumShapeV);

    // Floating-point lowering is more direct:
    //
    // op1 = exp(logits)
    // op2 = reduce_sum(op1, -1)
    // op3 = reciprocal(op2)
    // op4 = mul(op1, op3)
    auto op1ExpIn = tosa::CreateOpAndInfer<tosa::ExpOp>(
        rewriter, op->getLoc(), outputType, adaptor.input());
    RankedTensorType rsumType =
        RankedTensorType::get(rsumShape, outputType.getElementType());

    // Keep dims so we don't need to reshape later
    auto op2ReducesumOp1 =
        tosa::CreateOpAndInfer<tosa::ReduceSumOp>(rewriter, op->getLoc(),
            rsumType, op1ExpIn.getResult(), rewriter.getI64IntegerAttr(axis));
    auto op3ReciprocalOp2 = tosa::CreateOpAndInfer<tosa::ReciprocalOp>(rewriter,
        op->getLoc(), op2ReducesumOp1.getType(), op2ReducesumOp1.getResult());

    tosa::CreateReplaceOpAndInfer<tosa::MulOp>(rewriter, op, outputType,
        op1ExpIn.getResult(), op3ReciprocalOp2.getResult(), 0);

    return success();
  }
};

} // namespace

void populateLoweringONNXSoftmaxOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSoftmaxLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
