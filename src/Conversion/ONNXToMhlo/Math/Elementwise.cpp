/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <>
struct MhloDialectOp<ONNXAddOp> {
  using Op = mhlo::AddOp;
};

namespace {

// Element-wise unary ops lowering to Mhlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLoweringToMhlo : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseUnaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(StringAttr::get(op->getContext(),
                                    ElementwiseUnaryOp::getOperationName()),
        op->getLoc());
    Value mhloOp = rewriter.create<MhloOp<ElementwiseUnaryOp>>(
        loc, op->getResultTypes(), op->getOperands());
    rewriter.replaceOp(op, mhloOp);
    return success();
  }
};

template <>
struct ONNXElementwiseUnaryOpLoweringToMhlo<ONNXReluOp>
    : public ConversionPattern {
  ONNXElementwiseUnaryOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(ONNXReluOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(
        StringAttr::get(op->getContext(), ONNXReluOp::getOperationName()),
        op->getLoc());
    ONNXReluOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value inp = adaptor.X();
    ShapedType inpType = inp.getType().dyn_cast_or_null<ShapedType>();
    if (inpType == nullptr)
      return failure();
    Type elemType = inpType.getElementType();
    Type resultType = *op->result_type_begin();
    Value broadcastedZero;
    if (inpType.hasStaticShape())
      broadcastedZero =
          rewriter.create<mhlo::ConstOp>(loc, rewriter.getZeroAttr(inpType));
    else {
      Value zero =
          rewriter.create<mhlo::ConstOp>(loc, rewriter.getZeroAttr(elemType));
      Value shape = rewriter.create<shape::ShapeOfOp>(loc, inp);
      broadcastedZero = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
          loc, resultType, zero, shape, rewriter.getI64TensorAttr({}));
    }
    Value resultOp = rewriter.create<mhlo::MaxOp>(loc, inp, broadcastedZero);
    rewriter.replaceOp(op, resultOp);
    return success();
  }
};

// Element-wise variadic ops lowering to Mhlo dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLoweringToMhlo : public ConversionPattern {
  ONNXElementwiseVariadicOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseVariadicOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(StringAttr::get(op->getContext(),
                                    ElementwiseVariadicOp::getOperationName()),
        op->getLoc());

    // TODO: check whether ONNX dialect has explicit broadcast feature
    Value mhloOp = rewriter.create<MhloOp<ElementwiseVariadicOp>>(
        loc, op->getResultTypes(), op->getOperands());

    rewriter.replaceOp(op, mhloOp);

    return success();
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseVariadicOpLoweringToMhlo<ONNXAddOp>,
      ONNXElementwiseUnaryOpLoweringToMhlo<ONNXReluOp>>(ctx);
}

} // namespace onnx_mlir
