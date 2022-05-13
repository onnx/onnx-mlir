/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SqrtOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file lowers most unary operators from torch to onnx using a template
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXToTorchSqueezeV11OpLowering : public ConversionPattern {
  ONNXToTorchSqueezeV11OpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXSqueezeOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeV11Op squeezeOp = llvm::dyn_cast_or_null<ONNXSqueezeV11Op>(op);

    assert(squeezeOp && "Expecting op to have a strong type");

    Location loc = squeezeOp.getLoc();

    Value data = squeezeOp.data();
    ArrayAttr axes = squeezeOp.axesAttr();

    mlir::MLIRContext *context = squeezeOp.getContext();

    auto dataType = toTorchType(context, data.getType());
    auto resultType = toTorchType(context, squeezeOp.getResult().getType());
    auto dataTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, dataType, data);
    Value result;

    if (axes) {
      for (auto i = 0; axes.size(); i++) {
        auto j = axes[i].dyn_cast<IntegerAttr>();
        Value dim = rewriter.create<ConstantIntOp>(loc, j);

        result =
            rewriter.create<AtenSqueezeDimOp>(loc, resultType, dataTensor, dim);
      }
    } else {
      result = rewriter.create<AtenSqueezeOp>(loc, resultType, dataTensor);
    }

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);

    return success();
  }
};

struct ONNXToTorchSqueezeOpLowering : public ConversionPattern {
  ONNXToTorchSqueezeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXSqueezeOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeOp squeezeOp = llvm::dyn_cast_or_null<ONNXSqueezeOp>(op);

    assert(squeezeOp && "Expecting op to have a strong type");

    Location loc = squeezeOp.getLoc();

    Value data = squeezeOp.data();
    // Value axes = squeezeOp.axes();

    mlir::MLIRContext *context = squeezeOp.getContext();

    auto dataType = toTorchType(context, data.getType());
    auto resultType = toTorchType(context, squeezeOp.getResult().getType());
    auto dataTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, dataType, data);
    Value result;

    // if (axes) {
    //   for (auto i = 0; axes.size(); i++) {
    //     auto j = axes[i].dyn_cast<IntegerAttr>();
    //     Value dim = rewriter.create<ConstantIntOp>(loc, j);

    //     result =
    //         rewriter.create<AtenSqueezeDimOp>(loc, resultType, dataTensor,
    //         dim);
    //   }
    // } else {
      result = rewriter.create<AtenSqueezeOp>(loc, resultType, dataTensor);
    // }

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType,
    result);

    return success();
  }
};

void populateLoweringONNXToTorchSqueezeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns
      .insert<ONNXToTorchSqueezeV11OpLowering, ONNXToTorchSqueezeOpLowering>(
          typeConverter, ctx);
}
