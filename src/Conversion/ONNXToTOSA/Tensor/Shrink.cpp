/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Shrink.cpp - Shrink Op-------------------------------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers the ONNX Shrink operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXShrinkOpLoweringToTOSA : public OpConversionPattern<ONNXShrinkOp> {
public:
  using OpConversionPattern<ONNXShrinkOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXShrinkOp shrinkOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = shrinkOp->getLoc();

    auto lambd = adaptor.getLambdAttr();
    auto bias = adaptor.getBiasAttr();
    auto input = adaptor.getInput();

    TosaBuilder tosaBuilder(rewriter, loc);

    auto inputRankedTensorTy = dyn_cast<RankedTensorType>(input.getType());
    if (!inputRankedTensorTy) {
      return rewriter.notifyMatchFailure(
          loc, "Expected RankedTensorType for input data of ShrinkOp");
    }

    // lambd and bias have float type so it is safe to conver it to float
    const float lambdAsFloat = lambd.getValue().convertToFloat();
    const float biasAsFloat = bias.getValue().convertToFloat();
    auto lambdConstOp = tosaBuilder.getSplattedConst(lambdAsFloat,
        inputRankedTensorTy.getElementType(), inputRankedTensorTy.getShape());
    auto negatedLambdConstOp = tosaBuilder.getSplattedConst(-lambdAsFloat,
        inputRankedTensorTy.getElementType(), inputRankedTensorTy.getShape());
    auto biasConstOp = tosaBuilder.getSplattedConst(biasAsFloat,
        inputRankedTensorTy.getElementType(), inputRankedTensorTy.getShape());
    auto zeroConstOp = tosaBuilder.getSplattedConst(0,
        inputRankedTensorTy.getElementType(), inputRankedTensorTy.getShape());

    // Formula to be implemented:
    // { x < -lambd, then y = x + bias
    // { x > lambd,  then y = x - bias
    // { otherwise,  then y = 0

    auto firstCmp = tosaBuilder.compareOp<mlir::tosa::GreaterOp>(
        rewriter, loc, negatedLambdConstOp, input);
    auto firstFormula =
        tosaBuilder.binaryOp<mlir::tosa::AddOp>(input, biasConstOp);
    auto firstSelect = tosaBuilder.select(firstCmp, firstFormula, zeroConstOp);

    auto secondCmp = tosaBuilder.compareOp<mlir::tosa::GreaterOp>(
        rewriter, loc, input, lambdConstOp);
    auto secondFormula =
        tosaBuilder.binaryOp<mlir::tosa::SubOp>(input, biasConstOp);
    auto secondSelect =
        tosaBuilder.select(secondCmp, secondFormula, firstSelect);

    rewriter.replaceOp(shrinkOp, secondSelect);

    return success();
  }
};

} // namespace

void populateLoweringONNXShrinkOpToTOSAPattern(ConversionTarget & /*target*/,
    RewritePatternSet &patterns, TypeConverter & /*typeConverter*/,
    MLIRContext *ctx) {
  patterns.insert<ONNXShrinkOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
