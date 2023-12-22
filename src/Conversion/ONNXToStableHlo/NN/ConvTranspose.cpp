/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ConvTranspose.cpp - Lowering ConvTranspose Op ------------===//
//
// Copyright 2022-2023
//
// =============================================================================
//
// This file lowers ONNX ConvTranspose Operators to StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStableHlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStableHlo/ONNXToStableHloCommon.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXConvTransposeOpLoweringToStableHlo : public ConversionPattern {
  ONNXConvTransposeOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXConvTransposeOp::getOperationName(), 1, ctx) {}

  Value reshapeFilter(ConversionPatternRewriter &rewriter, Location loc,
      Value filterOperand, int64_t groupNum, int rank) const {
    assert(isRankedShapedType(filterOperand.getType()) &&
           "Expected Ranked ShapedType");
    ShapedType filterType = filterOperand.getType().cast<ShapedType>();
    assert(filterType.hasStaticShape() && "Expected static shape for filter");
    ArrayRef<int64_t> filterShape = filterType.getShape();
    Type elemType = filterType.getElementType();

    // 1. [IC, OC//G, H, W, ...] => [G, IC//G, OC//G, H, W, ...]
    SmallVector<int64_t> newFilterShape(filterShape.begin(), filterShape.end());
    newFilterShape[0] /= groupNum;
    newFilterShape.insert(newFilterShape.begin(), groupNum);
    filterOperand = rewriter.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get(newFilterShape, elemType), filterOperand);

    // 2. [G, IC//G, OC//G, H, W, ...] => [G, OC//G, IC//G, H, W, ...]
    llvm::SmallVector<int64_t> transposeDims(rank + 1);
    for (int64_t i = 0; i <= rank; i++)
      transposeDims[i] = i;
    std::swap(transposeDims[1], transposeDims[2]);
    filterOperand = rewriter.create<stablehlo::TransposeOp>(
        loc, filterOperand, rewriter.getDenseI64ArrayAttr(transposeDims));

    // 3. [G, OC//G, IC//G, H, W, ...] => [OC, IC//G, H, W, ...]
    std::swap(newFilterShape[1], newFilterShape[2]);
    newFilterShape.erase(newFilterShape.begin());
    newFilterShape[0] *= groupNum;
    filterOperand = rewriter.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get(newFilterShape, elemType), filterOperand);

    return filterOperand;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    ONNXConvTransposeOpAdaptor operandAdaptor(
        operands, op->getAttrDictionary());
    ONNXConvTransposeOp convOp = llvm::dyn_cast<ONNXConvTransposeOp>(op);
    Location loc = op->getLoc();

    IndexExprBuilderForStableHlo createIE(rewriter, loc);
    ONNXConvTransposeOpShapeHelper shapeHelper(op, operands, &createIE);
    LogicalResult shapecomputed = shapeHelper.computeShape();
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    llvm::SmallVector<IndexExpr, 2> kernelShape = shapeHelper.kernelShape;
    llvm::SmallVector<int64_t, 2> strides = shapeHelper.strides;
    llvm::SmallVector<int64_t, 2> dilations = shapeHelper.dilations;
    llvm::SmallVector<int64_t, 2> outputPadding = shapeHelper.outputPadding;
    bool needOutputPadding = std::any_of(outputPadding.begin(),
        outputPadding.end(), [](int64_t i) { return i != 0; });

    Value inputOperand = operandAdaptor.getX();
    Value filterOperand = operandAdaptor.getW();
    Value biasOperand = operandAdaptor.getB();
    bool hasBias = !biasOperand.getType().isa<NoneType>();
    int64_t groupNum = convOp.getGroup();

    assert(isRankedShapedType(inputOperand.getType()) &&
           "Expected Ranked ShapedType");
    ShapedType inputType = inputOperand.getType().cast<ShapedType>();
    Type elemType = inputType.getElementType();
    // Onnx Input is NCHW
    int64_t spatialOffset = 2;
    int64_t rank = inputType.getRank();
    int64_t kernelSize = kernelShape.size();

    Type outputType = *op->result_type_begin();
    Type convOutputType;
    if (!needOutputPadding)
      convOutputType = outputType;
    else {
      // use the shape inference result of shapeHelper
      llvm::SmallVector<IndexExpr, 2> dimsNoOutputPadding =
          shapeHelper.dimsNoOutputPadding;
      SmallVector<int64_t> convOutputShape;
      for (int i = 0; i < rank; ++i) {
        if (dimsNoOutputPadding[i].isLiteral())
          convOutputShape.emplace_back(dimsNoOutputPadding[i].getLiteral());
        else
          convOutputShape.emplace_back(ShapedType::kDynamic);
      }
      convOutputType = RankedTensorType::get(convOutputShape, elemType);
    }

    SmallVector<int64_t> spatialDimensions;
    for (int64_t i = spatialOffset; i < rank; i++) {
      spatialDimensions.push_back(i);
    }
    SmallVector<int64_t> kernelDimensions;
    for (int64_t i = spatialOffset; i < spatialOffset + kernelSize; i++) {
      kernelDimensions.push_back(i);
    }

    // paddings
    DimsExpr pads = shapeHelper.pads;
    int64_t spatialRank = rank - spatialOffset;
    SmallVector<int64_t> flattenPaddings;
    // currently only support static spatial dims
    if (!IndexExpr::isLiteral(kernelShape) || !IndexExpr::isLiteral(pads))
      return failure();
    for (int64_t i = 0; i < spatialRank; i++) {
      flattenPaddings.push_back(
          dilations[i] * (kernelShape[i].getLiteral() - 1) -
          pads[i].getLiteral());
      flattenPaddings.push_back(
          dilations[i] * (kernelShape[i].getLiteral() - 1) -
          pads[i + spatialRank].getLiteral());
    }

    stablehlo::ConvDimensionNumbersAttr dimension_numbers =
        stablehlo::ConvDimensionNumbersAttr::get(rewriter.getContext(), 0, 1,
            spatialDimensions, 1, 0, kernelDimensions, 0, 1, spatialDimensions);

    // Reverse and transpose filterOperand
    filterOperand = rewriter.create<stablehlo::ReverseOp>(
        loc, filterOperand, rewriter.getDenseI64ArrayAttr(spatialDimensions));
    if (groupNum > 1)
      filterOperand =
          reshapeFilter(rewriter, loc, filterOperand, groupNum, rank);
    else {
      // Transpose filterOperand from [i, o, ...] to [o, i, ...]
      llvm::SmallVector<int64_t> transposeDims(rank);
      for (int64_t i = 0; i < rank; i++)
        transposeDims[i] = i;
      std::swap(transposeDims[0], transposeDims[1]);
      filterOperand = rewriter.create<stablehlo::TransposeOp>(
          loc, filterOperand, rewriter.getDenseI64ArrayAttr(transposeDims));
    }

    Value convResult = rewriter.create<stablehlo::ConvolutionOp>(loc,
        convOutputType, inputOperand, filterOperand,
        DenseIntElementsAttr::get(
            RankedTensorType::get({spatialRank}, rewriter.getI64Type()),
            SmallVector<int64_t>(spatialRank, 1)),
        DenseIntElementsAttr::get(
            RankedTensorType::get({spatialRank, 2}, rewriter.getI64Type()),
            flattenPaddings),
        DenseIntElementsAttr::get(
            RankedTensorType::get({spatialRank}, rewriter.getI64Type()),
            strides),
        DenseIntElementsAttr::get(
            RankedTensorType::get({spatialRank}, rewriter.getI64Type()),
            dilations),
        nullptr, dimension_numbers, groupNum, 1, nullptr);

    Value padResult;
    if (!needOutputPadding) {
      padResult = convResult;
    } else {
      SmallVector<int64_t> edgePaddingLowVec(rank, 0);
      SmallVector<int64_t> edgePaddingHighVec(rank, 0);
      SmallVector<int64_t> interiorPaddingVec(rank, 0);
      std::copy(outputPadding.begin(), outputPadding.end(),
          edgePaddingHighVec.begin() + 2);
      Value zeroPaddingValue = rewriter.create<stablehlo::ConstantOp>(
          loc, DenseElementsAttr::get(mlir::RankedTensorType::get({}, elemType),
                   rewriter.getZeroAttr(elemType)));
      mlir::DenseI64ArrayAttr edgePaddingLow =
          rewriter.getDenseI64ArrayAttr(edgePaddingLowVec);
      mlir::DenseI64ArrayAttr edgePaddingHigh =
          rewriter.getDenseI64ArrayAttr(edgePaddingHighVec);
      mlir::DenseI64ArrayAttr interiorPadding =
          rewriter.getDenseI64ArrayAttr(interiorPaddingVec);
      padResult = rewriter.create<stablehlo::PadOp>(loc, outputType, convResult,
          zeroPaddingValue, edgePaddingLow, edgePaddingHigh, interiorPadding);
    }

    Value addBiasResult;
    if (!hasBias) {
      addBiasResult = padResult;
    } else {
      Value finalB;
      Value resultShape = rewriter.create<shape::ShapeOfOp>(loc, padResult);
      finalB = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(loc,
          outputType, biasOperand, resultShape, rewriter.getI64TensorAttr({1}));
      addBiasResult = rewriter.create<stablehlo::AddOp>(loc, padResult, finalB);
    }

    rewriter.replaceOp(op, addBiasResult);
    return success();
  }
};

} // namespace

void populateLoweringONNXConvTransposeOpToStableHloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConvTransposeOpLoweringToStableHlo>(ctx);
}

} // namespace onnx_mlir
