/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Clip.cpp - Lowering Clip Op ------------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Clip Operator to StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStableHlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStableHlo/ONNXToStableHloCommon.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXClipOp
//===----------------------------------------------------------------------===//

struct ONNXClipOpLoweringToStableHlo : public ConversionPattern {
  ONNXClipOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(ONNXClipOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    ONNXClipOpAdaptor operandAdaptor(operands);
    IndexExprBuilderForStableHlo createIE(rewriter, loc);
    ONNXClipOpShapeHelper shapeHelper(op, operands, &createIE);
    auto shapeComputed = shapeHelper.computeShape();
    assert(succeeded(shapeComputed) && "Could not compute output shape");

    Value input = operandAdaptor.getInput();
    Value min = operandAdaptor.getMin();
    Value max = operandAdaptor.getMax();

    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    Type elemType = outputShapedType.getElementType();

    MathBuilder createMath(rewriter, loc);
    if (isNoneValue(min)) {
      min = rewriter.create<stablehlo::ConstantOp>(
          loc, DenseElementsAttr::get(mlir::RankedTensorType::get({}, elemType),
                   createMath.negativeInfAttr(elemType)));
    }
    if (isNoneValue(max)) {
      max = rewriter.create<stablehlo::ConstantOp>(
          loc, DenseElementsAttr::get(mlir::RankedTensorType::get({}, elemType),
                   createMath.positiveInfAttr(elemType)));
    }

    Value result =
        rewriter.create<stablehlo::ClampOp>(loc, outputType, min, input, max);
    rewriter.replaceOp(op, result);
    return success();
  }
};

void populateLoweringONNXClipOpToStableHloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXClipOpLoweringToStableHlo>(ctx);
}

} // namespace onnx_mlir
