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

template <typename Softmax>
Value convertSoftmax(PatternRewriter &rewriter, Operation *op,
    RankedTensorType rsumType, const Value &op1ExpIn, int axis,
    int32_t inputRank) = delete;

// Before opset 13, softmax reduces axis and every dimension following.
template <>
Value convertSoftmax<ONNXSoftmaxV11Op>(PatternRewriter &rewriter, Operation *op,
    RankedTensorType rsumType, const Value &op1ExpIn, int axis,
    int32_t inputRank) {
  // Create shared outputType with dynamic shape. Infer method when creating
  // ops will insert a static shape if possible
  Type outputType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(inputRank, -1), rsumType.getElementType());
  // Create first reduce with input from function operands
  Value reducedSum = tosa::CreateOpAndInfer<mlir::tosa::ReduceSumOp>(rewriter,
      op->getLoc(), outputType, op1ExpIn, rewriter.getI64IntegerAttr(axis));
  // Loop over all following dimensions with last reduce as input
  for (int i = axis + 1; i < inputRank; i++) {
    reducedSum = tosa::CreateOpAndInfer<mlir::tosa::ReduceSumOp>(rewriter,
        op->getLoc(), outputType, reducedSum, rewriter.getI64IntegerAttr(i));
  }
  return reducedSum;
}

// From opset 13, softmax uses axis as the reduce axis.
template <>
Value convertSoftmax<ONNXSoftmaxOp>(PatternRewriter &rewriter, Operation *op,
    RankedTensorType rsumType, const Value &op1ExpIn, int axis,
    int32_t inputRank) {
  return tosa::CreateOpAndInfer<mlir::tosa::ReduceSumOp>(rewriter, op->getLoc(),
      rsumType, op1ExpIn, rewriter.getI64IntegerAttr(axis));
}

template <typename SoftmaxOp>
class ONNXSoftmaxLoweringToTOSA : public OpConversionPattern<SoftmaxOp> {
public:
  using OpConversionPattern<SoftmaxOp>::OpConversionPattern;
  using OpAdaptor = typename SoftmaxOp::Adaptor;
  LogicalResult matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // softmax = exp(logits) / reduce_sum(exp(logits), -1)
    auto outputType = op.getResult().getType().template dyn_cast<TensorType>();
    auto inputType = adaptor.input().getType().template dyn_cast<TensorType>();
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

    // Floating-point lowering is more direct:
    //
    // op1 = exp(logits)
    // op2 = reduce_sum(op1, -1)
    // op3 = reciprocal(op2)
    // op4 = mul(op1, op3)
    auto op1ExpIn = tosa::CreateOpAndInfer<mlir::tosa::ExpOp>(
        rewriter, op->getLoc(), outputType, adaptor.input());
    RankedTensorType rsumType = RankedTensorType::get(
        llvm::SmallVector<int64_t, 4>(inputType.getShape().size(), -1),
        outputType.getElementType());

    Value op2ReducesumOp1 = convertSoftmax<SoftmaxOp>(
        rewriter, op, rsumType, op1ExpIn.getResult(), axis, inputRank);

    auto op3ReciprocalOp2 = tosa::CreateOpAndInfer<mlir::tosa::ReciprocalOp>(
        rewriter, op->getLoc(), op2ReducesumOp1.getType(), op2ReducesumOp1);

    tosa::CreateReplaceOpAndInfer<mlir::tosa::MulOp>(rewriter, op, outputType,
        op1ExpIn.getResult(), op3ReciprocalOp2.getResult(), 0);

    return success();
  }
};

} // namespace

void populateLoweringONNXSoftmaxOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSoftmaxLoweringToTOSA<ONNXSoftmaxOp>,
      ONNXSoftmaxLoweringToTOSA<ONNXSoftmaxV11Op>>(typeConverter, ctx);
}

} // namespace onnx_mlir