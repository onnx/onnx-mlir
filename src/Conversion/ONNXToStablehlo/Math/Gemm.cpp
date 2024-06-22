/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Gemm.cpp - Lowering Gemm Op ------------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Gemm Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

bool closeTo(float a, float b, int ulps = 2) {
  return std::fabs(a - b) <=
             std::numeric_limits<float>::epsilon() * std::fabs(a + b) * ulps ||
         std::fabs(a - b) < std::numeric_limits<float>::min();
}

// ONNXGemmOp(A,B,C) is implemented using Stablehlo a * Dot(A(T), B(T)) + b * C;
template <typename GemmOp>
struct ONNXGemmOpLoweringToStablehlo : public ConversionPattern {
  ONNXGemmOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(GemmOp::getOperationName(), 1, ctx) {}

  void replaceGemmOp(ONNXGemmOp &gemmOp, Operation *op,
      ONNXGemmOpAdaptor &operandAdaptor, Type elemType,
      ONNXGemmOpShapeHelper &shapeHelper, ConversionPatternRewriter &rewriter,
      Location loc) const {
    float alphaLit = gemmOp.getAlpha().convertToFloat();
    float betaLit = gemmOp.getBeta().convertToFloat();
    Value A(operandAdaptor.getA()), B(operandAdaptor.getB()),
        C(operandAdaptor.getC());
    Value transA = A;
    Value transB = B;
    if (gemmOp.getTransA() == 1)
      transA = rewriter.create<stablehlo::TransposeOp>(
          loc, A, rewriter.getDenseI64ArrayAttr({1, 0}));
    if (gemmOp.getTransB() == 1)
      transB = rewriter.create<stablehlo::TransposeOp>(
          loc, B, rewriter.getDenseI64ArrayAttr({1, 0}));
    ShapedType resultType =
        mlir::dyn_cast_or_null<ShapedType>(gemmOp.getType());
    Value dot = rewriter.create<stablehlo::DotOp>(
        loc, gemmOp.getType(), transA, transB, nullptr);
    bool hasBias = shapeHelper.hasBias;

    // alpha * dot
    Value dotResult;
    if (closeTo(alphaLit, 1.0f))
      dotResult = dot;
    else {
      if (resultType.hasStaticShape()) {
        Value alphaVal = rewriter.create<stablehlo::ConstantOp>(
            loc, DenseElementsAttr::get(
                     resultType, rewriter.getFloatAttr(elemType, alphaLit)));
        dotResult = rewriter.create<stablehlo::MulOp>(loc, dot, alphaVal);
      } else {
        Value alphaVal = rewriter.create<stablehlo::ConstantOp>(
            loc, rewriter.getFloatAttr(elemType, alphaLit));
        Value shape = rewriter.create<shape::ShapeOfOp>(loc, dot);
        Value broadcastedAlpha =
            rewriter.create<stablehlo::DynamicBroadcastInDimOp>(loc, resultType,
                alphaVal, shape, rewriter.getDenseI64ArrayAttr({}));
        dotResult =
            rewriter.create<stablehlo::MulOp>(loc, dot, broadcastedAlpha);
      }
    }

    Value resultOp;

    if (!hasBias)
      resultOp = dotResult;
    else {
      // + beta * C
      int cRank = shapeHelper.cRank;
      Value broadcastedC;
      Value finalC;
      if (resultType.hasStaticShape()) {
        if (cRank == 1)
          broadcastedC = rewriter.create<stablehlo::BroadcastInDimOp>(
              loc, resultType, C, rewriter.getDenseI64ArrayAttr({1}));
        else if (cRank == 0)
          broadcastedC = rewriter.create<stablehlo::BroadcastInDimOp>(
              loc, resultType, C, rewriter.getDenseI64ArrayAttr({}));
        else
          broadcastedC = C;
        if (!closeTo(betaLit, 1.0f)) {
          Value betaVal = rewriter.create<stablehlo::ConstantOp>(
              loc, DenseElementsAttr::get(
                       resultType, rewriter.getFloatAttr(elemType, betaLit)));
          finalC =
              rewriter.create<stablehlo::MulOp>(loc, broadcastedC, betaVal);
        } else
          finalC = broadcastedC;
      } else {
        Value shape = rewriter.create<shape::ShapeOfOp>(loc, dot);
        if (cRank == 1)
          broadcastedC = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
              loc, resultType, C, shape, rewriter.getDenseI64ArrayAttr({1}));
        else if (cRank == 0)
          broadcastedC = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
              loc, resultType, C, shape, rewriter.getDenseI64ArrayAttr({}));
        else
          broadcastedC = C;
        if (!closeTo(betaLit, 1.0f)) {
          Value betaVal = rewriter.create<stablehlo::ConstantOp>(
              loc, rewriter.getFloatAttr(elemType, gemmOp.getBeta()));
          Value broadcastedBeta =
              rewriter.create<stablehlo::DynamicBroadcastInDimOp>(loc,
                  resultType, betaVal, shape,
                  rewriter.getDenseI64ArrayAttr({}));
          finalC = rewriter.create<stablehlo::MulOp>(
              loc, broadcastedC, broadcastedBeta);
        } else
          finalC = broadcastedC;
      }
      resultOp = rewriter.create<stablehlo::AddOp>(loc, dotResult, finalC);
    }

    rewriter.replaceOp(op, resultOp);
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Get shape.
    ONNXGemmOp gemmOp = llvm::cast<ONNXGemmOp>(op);
    ONNXGemmOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Location loc = op->getLoc();
    // Shape helper version for analysis: does not generate code for lowering.
    ONNXGemmOpShapeHelper shapeHelper(op, {});
    shapeHelper.computeShapeAndAssertOnFailure();

    ShapedType outpType = mlir::dyn_cast<ShapedType>(gemmOp.getType());
    if (outpType == nullptr)
      return failure();
    Type elemType = outpType.getElementType();

    replaceGemmOp(
        gemmOp, op, operandAdaptor, elemType, shapeHelper, rewriter, loc);
    return success();
  }
};
} // namespace

void populateLoweringONNXGemmOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLoweringToStablehlo<ONNXGemmOp>>(ctx);
}

} // namespace onnx_mlir
