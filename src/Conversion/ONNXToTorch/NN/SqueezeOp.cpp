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

std::vector<int> toUniqueAndNonNegative(std::vector<int> axes) {
  std::set<int> axesSet(axes.begin(), axes.end());
  std::vector<int> axesNonNeg;

  for (auto x : axesSet) {
    // positive integers are added as it
    // negative integers are normarlized to positive
    axesNonNeg.push_back((x > 0) ? x : (x + axesSet.size()));
  }
  return axesNonNeg;
}

std::vector<int> getSortedWithNegativeAxes(mlir::ArrayAttr axesRaw) {
  auto axesVector = toVector(axesRaw);
  auto axesNonNegative = toUniqueAndNonNegative(axesVector);
  auto axesSorted = axesNonNegative;

  std::sort(axesSorted.begin(), axesSorted.end());

  return axesSorted;
}

mlir::Value squeezeResult(std::vector<int> axes, mlir::Value dataTensor,
                          Torch::ValueTensorType resultType,
                          ConversionPatternRewriter &rewriter,
                          mlir::MLIRContext *context, Location loc) {
  Value result = dataTensor;

  if (axes.size() > 0) {
    for (auto i = 0; i < axes.size(); i++) {
      auto dataType = result.getType().dyn_cast<TensorType>();

      // With every successive deleting on dimension, the input axis
      // changes to `axis = axis - number_of_dimensions_deleted`
      // This works because, axes is sorted and normalized to possitive integers
      auto dim_raw = axes[i] - i;
      // assert((dataType.getShape()[dim_raw] == 1) && "Cannot squeeze for
      // dim");
      Value dim = getIntValue(dim_raw, rewriter, context, loc);
      result = rewriter.create<AtenSqueezeDimOp>(loc, resultType, result, dim);
    }
  } else {
    result = rewriter.create<AtenSqueezeOp>(loc, resultType, dataTensor);
  }

  return result;
}

struct ONNXToTorchSqueezeV11OpLowering : public ConversionPattern {
  ONNXToTorchSqueezeV11OpLowering(TypeConverter &typeConverter,
                                  MLIRContext *ctx)
      : ConversionPattern(typeConverter, ONNXSqueezeOp::getOperationName(), 1,
                          ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeV11Op squeezeOp = llvm::dyn_cast_or_null<ONNXSqueezeV11Op>(op);

    assert(squeezeOp && "Expecting op to have a strong type");

    Location loc = squeezeOp.getLoc();
    mlir::MLIRContext *context = squeezeOp.getContext();

    auto axes = getSortedWithNegativeAxes(squeezeOp.axesAttr());
    auto dataTensor = getTorchTensor(squeezeOp.data(), rewriter, context, loc);

    auto resultType = toTorchType(context, squeezeOp.getResult().getType());
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

  std::vector<int> getAxes(mlir::Value axes, mlir::MLIRContext *context) const {
    auto builder = mlir::Builder(context);
    auto axesConstOp = getONNXConstantOp(axes);
    auto axesAttr = createArrayAttrFromConstantOp(builder, axesConstOp);
    return getSortedWithNegativeAxes(axesAttr);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeOp squeezeOp = llvm::dyn_cast_or_null<ONNXSqueezeOp>(op);

    assert(squeezeOp && "Expecting op to have a strong type");

    Location loc = squeezeOp.getLoc();
    mlir::MLIRContext *context = squeezeOp.getContext();

    auto axes = getAxes(squeezeOp.axes(), context);
    auto dataTensor = getTorchTensor(squeezeOp.data(), rewriter, context, loc);

    auto resultType = toTorchType(context, squeezeOp.getResult().getType());
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
