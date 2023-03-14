/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Concat Op --------------------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX ConcatOp to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXConcatLoweringToTOSA : public OpConversionPattern<ONNXConcatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXConcatOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXConcatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    ValueRange inputs = adaptor.getInputs();
    int64_t axis = adaptor.getAxis();
    auto resultType = op.getResult().getType();

    for (const auto &input : inputs) {
      if (!onnx_mlir::isRankedShapedType(input.getType()))
        return rewriter.notifyMatchFailure(
            op, "inputs are not ranked shaped tensors");
    }
    int64_t inputRank = onnx_mlir::getRank(inputs[0].getType());

    // onnx allows values beetween [-r, r-1] where r is the rank.
    axis = tosa::convertNegativeAxis(axis, inputRank);

    Type newConcatOutputType =
        RankedTensorType::get(llvm::SmallVector<int64_t, 4>(inputRank, -1),
            resultType.cast<ShapedType>().getElementType());

    tosa::CreateReplaceOpAndInfer<mlir::tosa::ConcatOp>(
        rewriter, op, newConcatOutputType, inputs, axis);
    return success();
  }
};

} // namespace

void populateLoweringONNXConcatOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXConcatLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
