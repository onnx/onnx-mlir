/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Matmul.cpp - Lowering Matmul Op --------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

int64_t getLiteralValue(const IndexExpr &idx) {
  return idx.isLiteral() ? idx.getLiteral() : ShapedType::kDynamic;
}

struct ONNXMatMulOpLoweringToStablehlo : public ConversionPattern {
  ONNXMatMulOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXMatMulOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    IndexExprBuilderForStablehlo createIE(rewriter, loc);
    ONNXMatMulOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    Type elementType = outputShapedType.getElementType();

    Value A(operandAdaptor.getA()), B(operandAdaptor.getB());
    auto aRank = A.getType().cast<ShapedType>().getRank();
    auto bRank = B.getType().cast<ShapedType>().getRank();
    // Size all the arrays to padded length.
    int paddedRank = std::max(aRank, bRank);
    paddedRank = std::max(paddedRank, 2);
    DimsExpr aDims = shapeHelper.aDims;
    DimsExpr bDims = shapeHelper.bDims;
    llvm::BitVector aPadDims = shapeHelper.aPadDims;
    llvm::BitVector bPadDims = shapeHelper.bPadDims;

    DimsExpr outputDims = shapeHelper.getOutputDims();
    llvm::SmallVector<int64_t, 4> aShapeList;
    llvm::SmallVector<int64_t, 4> bShapeList;
    llvm::SmallVector<int64_t, 4> outputShapeList;

    IndexExpr::getShape(outputDims, outputShapeList);
    IndexExpr::getShape(aDims, aShapeList);
    IndexExpr::getShape(bDims, bShapeList);

    llvm::SmallVector<int64_t, 4> aShape;
    llvm::SmallVector<int64_t, 4> bShape;

    for (int64_t i = 0; i < paddedRank - 2; i++) {
      aShape.push_back(getLiteralValue(outputDims[i]));
      bShape.push_back(getLiteralValue(outputDims[i]));
    }
    if (!aPadDims[paddedRank - 2])
      aShape.push_back(aShapeList[paddedRank - 2]);
    aShape.push_back(aShapeList[paddedRank - 1]);
    bShape.push_back(bShapeList[paddedRank - 2]);
    if (!bPadDims[paddedRank - 1])
      bShape.push_back(bShapeList[paddedRank - 1]);

    RankedTensorType outputAType = RankedTensorType::get(aShape, elementType);
    RankedTensorType outputBType = RankedTensorType::get(bShape, elementType);

    int64_t oneDPadA = aPadDims[paddedRank - 2];
    int64_t oneDPadB = bPadDims[paddedRank - 1];

    // TODO: Refactor this miss before merging
    Value broadcastedA;
    {
      SmallVector<int64_t, 4> broadcastDimensions =
          llvm::to_vector<4>(llvm::seq<int64_t>(
              paddedRank - oneDPadA - aRank, paddedRank - oneDPadA));
      if (!outputAType.hasStaticShape()) {
        SmallVector<Value> dimTensors(paddedRank - oneDPadA - aRank);
        for (int64_t i = 0; i < paddedRank - oneDPadA - aRank; i++) {
          Value dim = rewriter.create<tensor::DimOp>(loc, B, i);
          dim = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getI64Type(), dim);
          dimTensors[i] =
              rewriter.create<tensor::FromElementsOp>(loc, ValueRange{dim});
        }
        Value aShape = rewriter.create<shape::ShapeOfOp>(loc, A);
        aShape = rewriter.create<arith::IndexCastOp>(loc,
            RankedTensorType::get({A.getType().cast<ShapedType>().getRank()},
                rewriter.getI64Type()),
            aShape);
        dimTensors.push_back(aShape);
        Value fullShape = rewriter.create<stablehlo::ConcatenateOp>(loc,
            RankedTensorType::get(
                {outputAType.getRank()}, rewriter.getI64Type()),
            dimTensors, rewriter.getI64IntegerAttr(0));
        broadcastedA =
            rewriter.createOrFold<stablehlo::DynamicBroadcastInDimOp>(loc,
                outputAType, A, fullShape,
                rewriter.getI64VectorAttr(broadcastDimensions));
      } else {
        broadcastedA = rewriter.createOrFold<stablehlo::BroadcastInDimOp>(loc,
            outputAType, A, rewriter.getI64VectorAttr(broadcastDimensions));
      }
    }
    Value broadcastedB;
    {
      SmallVector<int64_t, 4> broadcastDimensions =
          llvm::to_vector<4>(llvm::seq<int64_t>(
              paddedRank - oneDPadB - bRank, paddedRank - oneDPadB));
      if (!outputBType.hasStaticShape()) {
        SmallVector<Value> dimTensors(paddedRank - oneDPadB - bRank);
        for (int64_t i = 0; i < paddedRank - oneDPadB - bRank; i++) {
          Value dim = rewriter.create<tensor::DimOp>(loc, A, i);
          dim = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getI64Type(), dim);
          dimTensors[i] =
              rewriter.create<tensor::FromElementsOp>(loc, ValueRange{dim});
        }
        Value bShape = rewriter.create<shape::ShapeOfOp>(loc, B);
        bShape = rewriter.create<arith::IndexCastOp>(loc,
            RankedTensorType::get({B.getType().cast<ShapedType>().getRank()},
                rewriter.getI64Type()),
            bShape);
        dimTensors.push_back(bShape);
        Value fullShape = rewriter.create<stablehlo::ConcatenateOp>(loc,
            RankedTensorType::get(
                {outputBType.getRank()}, rewriter.getI64Type()),
            dimTensors, rewriter.getI64IntegerAttr(0));
        broadcastedB =
            rewriter.createOrFold<stablehlo::DynamicBroadcastInDimOp>(loc,
                outputBType, B, fullShape,
                rewriter.getI64VectorAttr(broadcastDimensions));
      } else {
        broadcastedB = rewriter.createOrFold<stablehlo::BroadcastInDimOp>(loc,
            outputBType, B, rewriter.getI64VectorAttr(broadcastDimensions));
      }
    }

    Value dotProduct;
    if (paddedRank > 2)
      dotProduct = rewriter.create<stablehlo::DotGeneralOp>(loc, outputType,
          broadcastedA, broadcastedB,
          stablehlo::DotDimensionNumbersAttr::get(rewriter.getContext(),
              llvm::to_vector<4>(llvm::seq<int64_t>(0, paddedRank - 2)),
              llvm::to_vector<4>(llvm::seq<int64_t>(0, paddedRank - 2)),
              {paddedRank - 1 - oneDPadA}, {paddedRank - 2}),
          nullptr);
    else {
      dotProduct = rewriter.create<stablehlo::DotOp>(loc,
          op->getResultTypes().front(), broadcastedA, broadcastedB, nullptr);
    }
    rewriter.replaceOp(op, dotProduct);
    return success();
  }
};

} // namespace

void populateLoweringONNXMatMulOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
