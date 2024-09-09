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
    ShapedType outputShapedType = mlir::cast<ShapedType>(outputType);
    Type elementType = outputShapedType.getElementType();

    Value A(operandAdaptor.getA()), B(operandAdaptor.getB());
    auto aRank = mlir::cast<ShapedType>(A.getType()).getRank();
    auto bRank = mlir::cast<ShapedType>(B.getType()).getRank();
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

    int64_t oneDPadA = aPadDims[paddedRank - 2];
    int64_t oneDPadB = bPadDims[paddedRank - 1];

    // TODO: Some of the above logic could probably be absorbed into this
    // function but will require more refactoring
    auto broadCastTo = [&](const Value &operandToBroadcast,
                           const Value &operandToMatch,
                           ArrayRef<int64_t> shapeInts, int64_t oneDPad) {
      Value broadcasted;
      auto rank =
          mlir::cast<ShapedType>(operandToBroadcast.getType()).getRank();
      RankedTensorType broadCastedType =
          RankedTensorType::get(shapeInts, elementType);
      SmallVector<int64_t, 4> broadcastDimensions =
          llvm::to_vector<4>(llvm::seq<int64_t>(
              paddedRank - oneDPad - rank, paddedRank - oneDPad));
      if (!broadCastedType.hasStaticShape()) {
        SmallVector<Value> dimTensors(paddedRank - oneDPad - rank);
        for (int64_t i = 0; i < paddedRank - oneDPad - rank; i++) {
          Value dim = rewriter.create<tensor::DimOp>(loc, operandToMatch, i);
          dim = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getI64Type(), dim);
          dimTensors[i] =
              rewriter.create<tensor::FromElementsOp>(loc, ValueRange{dim});
        }
        Value broadcastedShape =
            rewriter.create<shape::ShapeOfOp>(loc, operandToBroadcast);
        broadcastedShape = rewriter.create<arith::IndexCastOp>(loc,
            RankedTensorType::get({rank}, rewriter.getI64Type()),
            broadcastedShape);
        dimTensors.push_back(broadcastedShape);
        Value fullShape = rewriter.create<stablehlo::ConcatenateOp>(loc,
            RankedTensorType::get(
                {broadCastedType.getRank()}, rewriter.getI64Type()),
            dimTensors, rewriter.getI64IntegerAttr(0));
        broadcasted = rewriter.createOrFold<stablehlo::DynamicBroadcastInDimOp>(
            loc, broadCastedType, operandToBroadcast, fullShape,
            rewriter.getDenseI64ArrayAttr(broadcastDimensions));
      } else {
        broadcasted = rewriter.createOrFold<stablehlo::BroadcastInDimOp>(loc,
            broadCastedType, operandToBroadcast,
            rewriter.getDenseI64ArrayAttr(broadcastDimensions));
      }
      return broadcasted;
    };

    Value broadcastedA = broadCastTo(A, B, aShape, oneDPadA);
    Value broadcastedB = broadCastTo(B, A, bShape, oneDPadB);

    Value dotProduct;
    if (paddedRank > 2)
      dotProduct = rewriter.create<stablehlo::DotGeneralOp>(loc, outputType,
          broadcastedA, broadcastedB,
          stablehlo::DotDimensionNumbersAttr::get(rewriter.getContext(),
              llvm::to_vector<4>(llvm::seq<int64_t>(0, paddedRank - 2)),
              llvm::to_vector<4>(llvm::seq<int64_t>(0, paddedRank - 2)),
              {paddedRank - 1 - oneDPadA}, {paddedRank - 2}),
          /*precision_config*/ nullptr, /*algorithm*/ nullptr);
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
