/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Expand.cpp - Lowering Expand Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Expand Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

// ONNXExpandOp(A) is mainly implemented using MHLO mulOp(A, 1s)
struct ONNXExpandOpLoweringToMhlo : public ConversionPattern {
  ONNXExpandOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXExpandOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXExpandOpAdaptor operandAdaptor(operands);
    ONNXExpandOp expandOp = llvm::dyn_cast<ONNXExpandOp>(op);
    Value input = expandOp.input();
    Value shape = expandOp.shape();
    Location loc = op->getLoc();
    ONNXExpandOpShapeHelper shapeHelper(&expandOp);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Failed to compute shape");

    // Convert the output type to MemRefType.
    Type inputType = input.getType();
    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(inputType) && "Expected Ranked ShapedType");
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    Type elementType = outputShapedType.getElementType();
    int64_t outputRank = outputShapedType.getRank();

    Operation *shapeDefOp = shape.getDefiningOp();
    Value ones = rewriter.create<mhlo::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, 1.0));
    Value broadcastedOnes;
    if (ONNXShapeOp shapeOp = dyn_cast_or_null<ONNXShapeOp>(shapeDefOp)) {
      assert(shapeOp.data().getType().isa<ShapedType>() &&
             "ShapeOp's input data should be of ShapedType");
      int64_t shapeRank = shapeOp.data().getType().cast<ShapedType>().getRank();
      SmallVector<int64_t, 4> onesShape(shapeRank, ShapedType::kDynamicSize);
      RankedTensorType onesType = RankedTensorType::get(onesShape, elementType);
      broadcastedOnes = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
          loc, onesType, ones, shape, rewriter.getI64TensorAttr({}));
    } else if (ONNXConstantOp shapeOp =
                   dyn_cast_or_null<ONNXConstantOp>(shapeDefOp)) {
      llvm::SmallVector<int64_t, 4> shapeValues;
      mlir::DenseElementsAttr constShape =
          getONNXConstantOp(shapeOp)
              .valueAttr()
              .dyn_cast_or_null<mlir::DenseElementsAttr>();
      for (mlir::IntegerAttr element : constShape.getValues<IntegerAttr>())
        shapeValues.push_back(element.getInt());
      RankedTensorType broadcastedType =
          RankedTensorType::get(shapeValues, elementType);
      broadcastedOnes = rewriter.create<mhlo::BroadcastInDimOp>(
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
    Value result = rewriter.create<mhlo::MulOp>(
        loc, op->getResultTypes(), broadcastedOperands);
    rewriter.replaceOp(op, result);

    return success();
  }
};

void populateLoweringONNXExpandOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXExpandOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
