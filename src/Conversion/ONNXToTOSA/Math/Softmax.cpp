/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Softmax.cpp - Softmax Op ----------------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Before opset 13, softmax reduces axis and every dimension following.
template <typename ReductionOp>
Value computeReduction(PatternRewriter &rewriter, ONNXSoftmaxV11Op op,
    RankedTensorType rsumType, const Value &op1ExpIn, int axis) {
  const int64_t inputRank = rsumType.getRank();
  // Create shared outputType with dynamic shape. Infer method when creating
  // ops will insert a static shape if possible
  Type outputType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(inputRank, ShapedType::kDynamic),
      rsumType.getElementType());
  // Create first reduce with input from function operands
  Value reducedSum = tosa::CreateOpAndInfer<ReductionOp>(rewriter, op->getLoc(),
      outputType, op1ExpIn, rewriter.getI32IntegerAttr(axis));
  // Loop over all following dimensions with last reduce as input
  for (int i = axis + 1; i < inputRank; i++) {
    reducedSum = tosa::CreateOpAndInfer<ReductionOp>(rewriter, op->getLoc(),
        outputType, reducedSum, rewriter.getI32IntegerAttr(i));
  }
  return reducedSum;
}

// From opset 13, softmax uses axis as the reduce axis.
template <typename ReductionOp>
Value computeReduction(PatternRewriter &rewriter, ONNXSoftmaxOp op,
    RankedTensorType rsumType, const Value &op1ExpIn, int axis) {
  return tosa::CreateOpAndInfer<ReductionOp>(rewriter, op->getLoc(), rsumType,
      op1ExpIn, rewriter.getI32IntegerAttr(axis));
}

template <typename SoftmaxOp>
class ONNXSoftmaxLoweringToTOSA : public OpConversionPattern<SoftmaxOp> {
public:
  using OpConversionPattern<SoftmaxOp>::OpConversionPattern;
  using OpAdaptor = typename SoftmaxOp::Adaptor;
  LogicalResult matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getInput();
    // softmax = exp(logits) / reduce_sum(exp(logits), -1)
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    auto inputType =
        mlir::dyn_cast<RankedTensorType>(adaptor.getInput().getType());

    // Not a ranked tensor input/output
    if (!outputType || !inputType) {
      return rewriter.notifyMatchFailure(
          op, "input and result not ranked tensors");
    }

    // Get ONNX softmax axis
    int64_t axis = adaptor.getAxis();
    // Tosa only supports positive values
    int64_t inputRank = inputType.getRank();
    axis = tosa::convertNegativeAxis(axis, inputRank);
    // The legalization below is based on convertSoftmaxOp in
    // tensorflow tosa/transforms/legalize_common.cc, with the
    // addition of handling for axis.

    // Floating-point lowering is more direct:
    //
    // m   = reduce_max(logits)
    // op1 = exp(logits - m)
    // op2 = reduce_sum(op1, -1)
    // op3 = reciprocal(op2)
    // op4 = mul(op1, op3)
    RankedTensorType rsumType = RankedTensorType::get(
        llvm::SmallVector<int64_t, 4>(inputRank, ShapedType::kDynamic),
        outputType.getElementType());

    Value reduceMax = computeReduction<mlir::tosa::ReduceMaxOp>(
        rewriter, op, rsumType, input, axis);

    Value xLessMax = tosaBuilder.binaryOp<mlir::tosa::SubOp>(input, reduceMax);

    Value op1ExpIn = tosa::CreateOpAndInfer<mlir::tosa::ExpOp>(
        rewriter, loc, outputType, xLessMax);

    Value op2ReducesumOp1 = computeReduction<mlir::tosa::ReduceSumOp>(
        rewriter, op, rsumType, op1ExpIn, axis);

    Value op3ReciprocalOp2 = tosa::CreateOpAndInfer<mlir::tosa::ReciprocalOp>(
        rewriter, loc, op2ReducesumOp1.getType(), op2ReducesumOp1);

    Value mulOp = tosaBuilder.mul(op1ExpIn, op3ReciprocalOp2);
    rewriter.replaceOp(op, {mulOp});

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
