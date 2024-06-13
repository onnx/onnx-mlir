/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Normalization.cpp - Lowering Normalization Ops -----------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers ONNX Conv Operators to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXConvOpLoweringToStablehlo : public ConversionPattern {
  ONNXConvOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConvOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    ONNXConvOpAdaptor operandAdaptor(operands);
    ONNXConvOp convOp = llvm::dyn_cast<ONNXConvOp>(op);
    Location loc = op->getLoc();

    IndexExprBuilderForStablehlo createStablehloIE(rewriter, loc);
    ONNXConvOpShapeHelper shapeHelper(op, operands, &createStablehloIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    llvm::SmallVector<IndexExpr, 2> kernelShape = shapeHelper.kernelShape;
    llvm::SmallVector<int64_t, 2> strides = shapeHelper.strides;
    llvm::SmallVector<int64_t, 2> dilations = shapeHelper.dilations;
    DimsExpr outputDims = shapeHelper.getOutputDims();
    int outputRank = shapeHelper.getOutputDims().size();

    Value inputOperand = operandAdaptor.getX();
    Value filterOperand = operandAdaptor.getW();
    Value biasOperand = operandAdaptor.getB();
    bool hasBias = !mlir::isa<NoneType>(biasOperand.getType());
    int64_t groupNum = convOp.getGroup();

    assert(isRankedShapedType(inputOperand.getType()) &&
           "Expected Ranked ShapedType");
    ShapedType inputType = mlir::cast<ShapedType>(inputOperand.getType());
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    Type outputType = *op->result_type_begin();
    // Onnx Input is NCHW
    int64_t spatialOffset = 2;
    int64_t rank = inputType.getRank();
    int64_t kernelSize = kernelShape.size();

    SmallVector<int64_t> inputSpatialDimensions;
    for (int64_t i = spatialOffset; i < rank; i++) {
      inputSpatialDimensions.push_back(i);
    }

    SmallVector<int64_t> kernelDimensions;
    for (int64_t i = spatialOffset; i < spatialOffset + kernelSize; i++) {
      kernelDimensions.push_back(i);
    }

    SmallVector<int64_t> outputSpatialDimensions;
    for (int64_t i = spatialOffset; i < outputRank; i++) {
      outputSpatialDimensions.push_back(i);
    }

    // paddings
    DimsExpr pads = shapeHelper.pads;
    llvm::StringRef padding = convOp.getAutoPad();
    int64_t spatialRank = rank - spatialOffset;
    SmallVector<int64_t> flattenPaddings;
    bool needPadding = (padding == "NOTSET");

    // currently only support static shape

    if (!needPadding) {
      if (!IndexExpr::isLiteral(pads))
        return failure();
    } else {
      if (!IndexExpr::isLiteral(kernelShape) || !IndexExpr::isLiteral(pads) ||
          !IndexExpr::isLiteral(outputDims))
        return failure();
      if (!inputType.hasStaticShape())
        return failure();
    }

    for (int64_t i = 0; i < spatialRank; i++) {
      if (!needPadding) {
        flattenPaddings.push_back(pads[i].getLiteral());
        flattenPaddings.push_back(pads[i + spatialRank].getLiteral());
      } else {
        int64_t kdTerm = (kernelShape[i].getLiteral() - 1) * dilations[i] + 1;
        int64_t padFront = pads[i].getLiteral();
        int64_t padBack =
            (outputDims[i + spatialOffset].getLiteral() - 1) * strides[i] +
            kdTerm - inputShape[i + spatialOffset] - padFront;
        flattenPaddings.push_back(padFront);
        flattenPaddings.push_back(padBack);
      }
    }

    stablehlo::ConvDimensionNumbersAttr dimension_numbers =
        stablehlo::ConvDimensionNumbersAttr::get(rewriter.getContext(), 0, 1,
            inputSpatialDimensions, 1, 0, kernelDimensions, 0, 1,
            outputSpatialDimensions);

    Value convResult =
        rewriter.create<stablehlo::ConvolutionOp>(loc, outputType, inputOperand,
            filterOperand, rewriter.getDenseI64ArrayAttr(strides),
            DenseIntElementsAttr::get(
                RankedTensorType::get({spatialRank, 2}, rewriter.getI64Type()),
                flattenPaddings),
            DenseI64ArrayAttr(), rewriter.getDenseI64ArrayAttr(dilations),
            nullptr, dimension_numbers, groupNum, 1, nullptr);

    Value result;
    if (!hasBias) {
      result = convResult;
    } else {
      Value finalB;
      Value resultShape = rewriter.create<shape::ShapeOfOp>(loc, convResult);
      finalB =
          rewriter.create<stablehlo::DynamicBroadcastInDimOp>(loc, outputType,
              biasOperand, resultShape, rewriter.getDenseI64ArrayAttr({1}));
      result = rewriter.create<stablehlo::AddOp>(loc, convResult, finalB);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXConvOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
