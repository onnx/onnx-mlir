/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gather.cpp - Lowering Gather Op ---------------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXGatherOp is mainly implemented using MHLO TorchIndexSelectOp
struct ONNXGatherOpLoweringToMhlo : public ConversionPattern {
  ONNXGatherOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXGatherOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXGatherOpAdaptor operandAdaptor(operands);
    ONNXGatherOp gatherOp = cast<ONNXGatherOp>(op);
    Location loc = op->getLoc();

    ONNXGatherOpShapeHelper shapeHelper(&gatherOp);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");

    // Operands and attributes.
    Value data = operandAdaptor.data();
    Value indices = operandAdaptor.indices();
    int64_t axisLit = gatherOp.axis();

    ShapedType inputType = data.getType().cast<ShapedType>();
    int64_t dataRank = inputType.getRank();
    ShapedType indicesType = indices.getType().cast<ShapedType>();
    // Negative value means counting dimensions from the back.
    axisLit = axisLit < 0 ? axisLit + dataRank : axisLit;

    // start indices
    Value zero = getShapedZero(loc, rewriter, indices);
    Value axisDimSize;
    if (inputType.hasStaticShape()) {
      int64_t axisDimSizeLit = inputType.getShape()[axisLit];
      axisDimSize = getShapedInt(loc, rewriter, axisDimSizeLit, indices);
    } else {
      Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, data);
      Value indicesShape = rewriter.create<shape::ShapeOfOp>(loc, indices);
      Value axisDimSizeIndexValue =
          rewriter.create<shape::GetExtentOp>(loc, inputShape, axisLit);
      Value axisDimSizeValue = rewriter.create<arith::IndexCastOp>(
          loc, indicesType.getElementType(), axisDimSizeIndexValue);
      axisDimSize =
          rewriter.create<mhlo::DynamicBroadcastInDimOp>(loc, indicesType,
              axisDimSizeValue, indicesShape, rewriter.getI64TensorAttr({}));
    }
    Value greaterOp = rewriter.create<mhlo::CompareOp>(
        loc, indices, zero, mhlo::ComparisonDirection::LT);
    Value positiveIndices =
        rewriter.create<mhlo::AddOp>(loc, indicesType, indices, axisDimSize);
    Value startIndices = rewriter.create<mhlo::SelectOp>(
        loc, indicesType, greaterOp, positiveIndices, indices);

    Value gatherValue = rewriter.create<mhlo::TorchIndexSelectOp>(loc,
        outputType, data, startIndices, rewriter.getI64IntegerAttr(axisLit),
        rewriter.getI64IntegerAttr(0));
    rewriter.replaceOp(op, gatherValue);
    return success();
  }
};

} // namespace

void populateLoweringONNXGatherOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGatherOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
