/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Gemm.cpp - Lowering Gemm Op ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gemm Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

bool closeTo(float a, float b, int ulps = 2) {
  return std::fabs(a - b) <=
             std::numeric_limits<float>::epsilon() * std::fabs(a + b) * ulps ||
         std::fabs(a - b) < std::numeric_limits<float>::min();
}

template <typename GemmOp>
struct ONNXGemmOpLoweringToMhlo : public ConversionPattern {
  ONNXGemmOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(GemmOp::getOperationName(), 1, ctx) {}

  void replaceGemmOp(ONNXGemmOp &gemmOp, Operation *op,
      ONNXGemmOpAdaptor &operandAdaptor, Type elemType,
      ONNXGemmOpShapeHelper &shapeHelper, ConversionPatternRewriter &rewriter,
      Location loc) const {
    float alphaLit = gemmOp.alpha().convertToFloat();
    float betaLit = gemmOp.beta().convertToFloat();
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(operandAdaptor.C());
    Value transA = A;
    Value transB = B;
    if (gemmOp.transA() == 1)
      transA = rewriter.create<mhlo::TransposeOp>(
          loc, A, rewriter.getI64VectorAttr({1, 0}));
    if (gemmOp.transB() == 1)
      transB = rewriter.create<mhlo::TransposeOp>(
          loc, B, rewriter.getI64VectorAttr({1, 0}));
    ShapedType resultType = gemmOp.getType().dyn_cast_or_null<ShapedType>();
    Value dot = rewriter.create<mhlo::DotOp>(loc, transA, transB, nullptr);
    bool hasBias = shapeHelper.hasBias;

    // alpha * dot
    Value dotResult;
    if (closeTo(alphaLit, 1.0f))
      dotResult = dot;
    else {
      if (resultType.hasStaticShape()) {
        Value alphaVal = rewriter.create<mhlo::ConstantOp>(
            loc, DenseElementsAttr::get(
                     resultType, rewriter.getFloatAttr(elemType, alphaLit)));
        dotResult = rewriter.create<mhlo::MulOp>(loc, dot, alphaVal);
      } else {
        Value alphaVal = rewriter.create<mhlo::ConstantOp>(
            loc, rewriter.getFloatAttr(elemType, alphaLit));
        Value shape = rewriter.create<shape::ShapeOfOp>(loc, dot);
        Value broadcastedAlpha = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
            loc, resultType, alphaVal, shape, rewriter.getI64TensorAttr({}));
        dotResult = rewriter.create<mhlo::MulOp>(loc, dot, broadcastedAlpha);
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
          broadcastedC = rewriter.create<mhlo::BroadcastInDimOp>(
              loc, resultType, C, rewriter.getI64TensorAttr({1}));
        else if (cRank == 0)
          broadcastedC = rewriter.create<mhlo::BroadcastInDimOp>(
              loc, resultType, C, rewriter.getI64TensorAttr({}));
        else
          broadcastedC = C;
        if (!closeTo(betaLit, 1.0f)) {
          Value betaVal = rewriter.create<mhlo::ConstantOp>(
              loc, DenseElementsAttr::get(
                       resultType, rewriter.getFloatAttr(elemType, betaLit)));
          finalC = rewriter.create<mhlo::MulOp>(loc, broadcastedC, betaVal);
        } else
          finalC = broadcastedC;
      } else {
        Value shape = rewriter.create<shape::ShapeOfOp>(loc, dot);
        if (cRank == 1)
          broadcastedC = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
              loc, resultType, C, shape, rewriter.getI64TensorAttr({1}));
        else if (cRank == 0)
          broadcastedC = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
              loc, resultType, C, shape, rewriter.getI64TensorAttr({}));
        else
          broadcastedC = C;
        if (!closeTo(betaLit, 1.0f)) {
          Value betaVal = rewriter.create<mhlo::ConstantOp>(
              loc, rewriter.getFloatAttr(elemType, gemmOp.beta()));
          Value broadcastedBeta =
              rewriter.create<mhlo::DynamicBroadcastInDimOp>(loc, resultType,
                  betaVal, shape, rewriter.getI64TensorAttr({}));
          finalC =
              rewriter.create<mhlo::MulOp>(loc, broadcastedC, broadcastedBeta);
        } else
          finalC = broadcastedC;
      }
      resultOp = rewriter.create<mhlo::AddOp>(loc, dotResult, finalC);
    }

    rewriter.replaceOp(op, resultOp);
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Get shape.
    ONNXGemmOp gemmOp = llvm::cast<ONNXGemmOp>(op);
    ONNXGemmOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Location loc = op->getLoc();
    ONNXGemmOpShapeHelper shapeHelper(&gemmOp);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    ShapedType outpType = gemmOp.getType().dyn_cast<ShapedType>();
    if (outpType == nullptr)
      return failure();
    Type elemType = outpType.getElementType();

    replaceGemmOp(
        gemmOp, op, operandAdaptor, elemType, shapeHelper, rewriter, loc);
    return success();
  }
};
} // namespace

void populateLoweringONNXGemmOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLoweringToMhlo<ONNXGemmOp>>(ctx);
}

} // namespace onnx_mlir
