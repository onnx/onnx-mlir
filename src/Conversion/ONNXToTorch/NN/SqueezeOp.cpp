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
#include <cstdint>
#include <set>
#include <vector>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXToTorchSqueezeV11OpLowering : public ConversionPattern {
  ONNXToTorchSqueezeV11OpLowering(TypeConverter &typeConverter,
                                  MLIRContext *ctx)
      : ConversionPattern(typeConverter, ONNXSqueezeOp::getOperationName(), 1,
                          ctx) {}

  std::vector<int> toVector(mlir::ArrayAttr axes_unsorted) const {
    std::vector<int> axes;

    for (auto i : axes_unsorted) {
      auto j = i.dyn_cast<IntegerAttr>();
      int64_t k = j.getValue().getSExtValue();
      axes.push_back(k);
    }

    return axes;
  }

  std::vector<int> getAxes(mlir::ArrayAttr axes) const {
    auto axes_vec = toVector(axes);
    return getSortedWithNegativeAxes(axes_vec);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeV11Op squeezeOp = llvm::dyn_cast_or_null<ONNXSqueezeV11Op>(op);

    assert(squeezeOp && "Expecting op to have a strong type");

    Location loc = squeezeOp.getLoc();

    Value data = squeezeOp.data();
    auto axes = getAxes(squeezeOp.axesAttr());

    mlir::MLIRContext *context = squeezeOp.getContext();

    auto dataType = toTorchType(context, data.getType());
    auto resultType = toTorchType(context, squeezeOp.getResult().getType());
    auto dataTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, dataType, data);
    Value result =
        squeezeResult(axes, dataTensor, resultType, rewriter, context, loc);

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);

    return success();
  }
};

struct ONNXToTorchSqueezeOpLowering : public ConversionPattern {
  ONNXToTorchSqueezeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, ONNXSqueezeOp::getOperationName(), 1,
                          ctx) {}

  std::vector<int> toVector(mlir::Value axes_unsorted) const {
    std::vector<int> axes;

    // for (auto i : axes_unsorted) {
    //   auto j = i.dyn_cast<IntegerAttr>();
    //   axes.push_back(j);
    // }

    return axes;
  }

  std::vector<int> getAxes(mlir::Value axes) const {
    auto axes_vec = toVector(axes);
    return getSortedWithNegativeAxes(axes_vec);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeOp squeezeOp = llvm::dyn_cast_or_null<ONNXSqueezeOp>(op);

    assert(squeezeOp && "Expecting op to have a strong type");

    Location loc = squeezeOp.getLoc();

    Value data = squeezeOp.data();
    mlir::ValueRange x = squeezeOp.axes();

    auto axes = getAxes(squeezeOp.axes());

    mlir::MLIRContext *context = squeezeOp.getContext();

    auto dataType = toTorchType(context, data.getType());
    auto resultType = toTorchType(context, squeezeOp.getResult().getType());
    auto dataTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, dataType, data);

    auto result =
        squeezeResult(axes, dataTensor, resultType, rewriter, context, loc);

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);

    return success();
  }
};

void populateLoweringONNXToTorchSqueezeOpPattern(RewritePatternSet &patterns,
                                                 TypeConverter &typeConverter,
                                                 MLIRContext *ctx) {
  patterns
      .insert<ONNXToTorchSqueezeV11OpLowering, ONNXToTorchSqueezeOpLowering>(
          typeConverter, ctx);
}
