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

#include <cstdint>
#include <set>
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

  Value getIntValue(int val, ConversionPatternRewriter &rewriter,
      mlir::MLIRContext *context, Location loc) const {
    auto iType = IntegerType::get(context, 64);
    auto iVal = IntegerAttr::get(iType, val);
    return rewriter.create<ConstantIntOp>(loc, iVal);
  }

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
    Value result = dataTensor;

    // if (0) {
    //   for (auto i = 0; axes.size(); i++) {
    //     Value dim = getIntValue(0, rewriter, context, loc);

    //     result =
    //       rewriter.create<AtenSqueezeDimOp>(loc, resultType, result, dim);
    //   }
    // } else {
      result = rewriter.create<AtenSqueezeOp>(loc, resultType, dataTensor);
    // }

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);

    return success();
  }
};

struct ONNXToTorchSqueezeOpLowering : public ConversionPattern {
  ONNXToTorchSqueezeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXSqueezeOp::getOperationName(), 1, ctx) {}

  Value getIntValue(int val, ConversionPatternRewriter &rewriter,
      mlir::MLIRContext *context, Location loc) const {
    auto iType = IntegerType::get(context, 64);
    auto iVal = IntegerAttr::get(iType, val);
    return rewriter.create<ConstantIntOp>(loc, iVal);
  }

  std::vector<int> toVector(mlir::Value axes_unsorted) const {
    std::vector<int> axes;

    for (auto i : axes_unsorted) {
      auto j = i.dyn_cast<IntegerAttr>();
      axes.push_back(j);
    }

    return axes;
  }

  std::vector<int> toUniqueAndNonNegative(std::vector<int> axes) const {
    std::set<int> axesSet(axes.begin(), axes.end());
    std::vector<int> axesNonNeg;

    for (auto x : axesSet) {
      // positive integers are added as it
      // negative integers are normarlized to positive
      axesNonNeg.push_back((x > 0) ? x : (x + axesSet.size()));
    }
    return axesNonNeg;
  }

  std::vector<int> getSortedWithNegativeAxes(mlir::Value axesRaw) const {
    auto axesVector = toVector(axesRaw);
    auto axesNonNegative = toUniqueAndNonNegative(axesVector);
    auto axesSorted = axesNonNegative;

    std::sort(axesSorted.begin(), axesSorted.end());

    return axesSorted;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeOp squeezeOp = llvm::dyn_cast_or_null<ONNXSqueezeOp>(op);

    assert(squeezeOp && "Expecting op to have a strong type");

    Location loc = squeezeOp.getLoc();

    Value data = squeezeOp.data();
    mlir::ValueRange x = squeezeOp.axes();

    auto axes = getSortedWithNegativeAxes(squeezeOp.axes());

    mlir::MLIRContext *context = squeezeOp.getContext();

    auto dataType = toTorchType(context, data.getType());
    auto resultType = toTorchType(context, squeezeOp.getResult().getType());
    auto dataTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, dataType, data);
    Value result = dataTensor;

    if (axes.size() > 0) {
      for (auto i=0; i<axes.size(); i++) {
        // With every successive deleting on dimension, the input axis
        // changes to `axis = axis - number_of_dimensions_deleted`
        Value dim = getIntValue((axes[i] - i), rewriter, context, loc);
        result =
          rewriter.create<AtenSqueezeDimOp>(loc, resultType, result, dim);
      }
    } else {
      result = rewriter.create<AtenSqueezeOp>(loc, resultType, dataTensor);
    }

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
