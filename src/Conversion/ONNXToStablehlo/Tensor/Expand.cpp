/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Expand.cpp - Lowering Expand Op----------------------=== //
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Expand Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

// ONNXExpandOp(A) is mainly implemented using Stablehlo mulOp(A, 1s)
struct ONNXExpandOpLoweringToStablehlo : public ConversionPattern {
  ONNXExpandOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXExpandOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXExpandOpAdaptor operandAdaptor(operands);
    ONNXExpandOp expandOp = llvm::dyn_cast<ONNXExpandOp>(op);
    Value input = expandOp.getInput();
    Value shape = expandOp.getShape();
    Location loc = op->getLoc();

    // Cannot be used because ExpandOp Shape helper scans for onnx ops in the
    // inputs, and Stablehlo conversion has already removed them.

    // IndexExprBuilderForStablehlo createIE(rewriter, loc);
    // ONNXExpandOpShapeHelper shapeHelper(op, operands, &createIE);
    // LogicalResult shapeComputed = shapeHelper.computeShape();
    // assert(succeeded(shapeComputed) && "Failed to compute shape");

    Type inputType = input.getType();
    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(inputType) && "Expected Ranked ShapedType");
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    Type elementType = outputShapedType.getElementType();
    int64_t outputRank = outputShapedType.getRank();

    Operation *shapeDefOp = shape.getDefiningOp();
    Value ones;
    if (elementType.isa<IntegerType>())
      ones = rewriter.create<stablehlo::ConstantOp>(
          loc, rewriter.getIntegerAttr(elementType, 1));
    else
      ones = rewriter.create<stablehlo::ConstantOp>(
          loc, rewriter.getFloatAttr(elementType, 1.0));
    Value broadcastedOnes;
    if (ONNXShapeOp shapeOp = dyn_cast_or_null<ONNXShapeOp>(shapeDefOp)) {
      assert(shapeOp.getData().getType().isa<ShapedType>() &&
             "ShapeOp's input data should be of ShapedType");
      int64_t shapeRank =
          shapeOp.getData().getType().cast<ShapedType>().getRank();
      SmallVector<int64_t, 4> onesShape(shapeRank, ShapedType::kDynamic);
      RankedTensorType onesType = RankedTensorType::get(onesShape, elementType);
      broadcastedOnes = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
          loc, onesType, ones, shape, rewriter.getI64TensorAttr({}));
    } else if (mlir::ElementsAttr constShape =
                   getElementAttributeFromConstValue(shape)) {
      llvm::SmallVector<int64_t, 4> shapeValues;
      for (mlir::IntegerAttr element : constShape.getValues<IntegerAttr>())
        shapeValues.push_back(element.getInt());
      RankedTensorType broadcastedType =
          RankedTensorType::get(shapeValues, elementType);
      broadcastedOnes = rewriter.create<stablehlo::BroadcastInDimOp>(
          loc, broadcastedType, ones, rewriter.getI64TensorAttr({}));
    } else {
      assert(
          false &&
          "Shape argument of Expand is the output of an unexpected operation. "
          "Supported operations are: onnx.Constant and onnx.Shape");
    }
    llvm::SmallVector<Value, 4> newOperands = {input, broadcastedOnes};
    llvm::SmallVector<Value, 4> broadcastedOperands = getBroadcastedOperands(
        newOperands, outputType, rewriter, loc, outputRank);
    Value result = rewriter.create<stablehlo::MulOp>(
        loc, op->getResultTypes(), broadcastedOperands);
    rewriter.replaceOp(op, result);

    return success();
  }
};

void populateLoweringONNXExpandOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXExpandOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
