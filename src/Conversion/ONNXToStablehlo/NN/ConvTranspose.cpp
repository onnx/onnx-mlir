/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ConvTranspose.cpp - Lowering ConvTranspose Op ------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers ONNX ConvTranspose Operators to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXConvTransposeOpLoweringToStablehlo : public ConversionPattern {
  ONNXConvTransposeOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXConvTransposeOp::getOperationName(), 1, ctx) {}

  Value reshapeFilter(ConversionPatternRewriter &rewriter, Location loc,
      Value filterOperand, int64_t groupNum, int rank) const {
    assert(isRankedShapedType(filterOperand.getType()) &&
           "Expected Ranked ShapedType");
    ShapedType filterType = mlir::cast<ShapedType>(filterOperand.getType());
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

    IndexExprBuilderForStablehlo createIE(rewriter, loc);
    ONNXConvTransposeOpShapeHelper shapeHelper(op, operands, &createIE);
    LogicalResult shapecomputed = shapeHelper.computeShape();
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    llvm::SmallVector<IndexExpr, 2> kernelShape = shapeHelper.kernelShape;
    llvm::SmallVector<int64_t, 2> strides = shapeHelper.strides;
    llvm::SmallVector<int64_t, 2> dilations = shapeHelper.dilations;
    llvm::SmallVector<int64_t, 2> outputPadding = shapeHelper.outputPadding;

    Value inputOperand = operandAdaptor.getX();
    Value filterOperand = operandAdaptor.getW();
    Value biasOperand = operandAdaptor.getB();
    bool hasBias = !mlir::isa<NoneType>(biasOperand.getType());
    int64_t groupNum = convOp.getGroup();

    assert(isRankedShapedType(inputOperand.getType()) &&
           "Expected Ranked ShapedType");
    ShapedType inputType = mlir::cast<ShapedType>(inputOperand.getType());
    // Onnx Input is NCHW
    int64_t spatialOffset = 2;
    int64_t rank = inputType.getRank();
    int64_t kernelSize = kernelShape.size();

    Type convOutputType = *op->result_type_begin();

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
          pads[i + spatialRank].getLiteral() + outputPadding[i]);
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
        rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(spatialRank, 1)),
        DenseIntElementsAttr::get(
            RankedTensorType::get({spatialRank, 2}, rewriter.getI64Type()),
            flattenPaddings),
        rewriter.getDenseI64ArrayAttr(strides),
        rewriter.getDenseI64ArrayAttr(dilations), nullptr, dimension_numbers,
        groupNum, 1, nullptr);

    Value addBiasResult;
    if (!hasBias) {
      addBiasResult = convResult;
    } else {
      Value finalB;
      Value resultShape = rewriter.create<shape::ShapeOfOp>(loc, convResult);
      finalB = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(loc,
          convOutputType, biasOperand, resultShape,
          rewriter.getDenseI64ArrayAttr({1}));
      addBiasResult =
          rewriter.create<stablehlo::AddOp>(loc, convResult, finalB);
    }

    rewriter.replaceOp(op, addBiasResult);
    return success();
  }
};

} // namespace

void populateLoweringONNXConvTransposeOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConvTransposeOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
