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

std::vector<int> getAxes(ONNXSqueezeOp squeezeOp) {
  auto builder = mlir::Builder(squeezeOp.getContext());
  auto axesConstOp = getONNXConstantOp(squeezeOp.axes());
  auto axesAttr = createArrayAttrFromConstantOp(builder, axesConstOp);
  return getSortedWithNegativeAxes(axesAttr);
}

std::vector<int> getAxes(ONNXSqueezeV11Op squeezeOp) {
  return getSortedWithNegativeAxes(squeezeOp.axesAttr());
}

mlir::Value squeezeResult(std::vector<int> axes, mlir::Value dataTensor,
                          Torch::ValueTensorType resultType,
                          ConversionPatternRewriter &rewriter,
                          mlir::MLIRContext *context, Location loc) {
  Value result = dataTensor;

  if (axes.size() > 0) {
    for (uint64_t i = 0; i < axes.size(); i++) {
      // With every successive deleting on dimension, the input axis
      // changes to `axis = axis - number_of_dimensions_deleted`
      // This works because, axes is sorted and normalized to possitive integers
      auto dim_raw = axes[i] - i;
      // Fail when dimension squeeze is not 1?
      // assert((result.getType().dyn_cast<TensorType>().getShape()[dim_raw] ==
      // 1) && "Cannot squeeze for dim");
      Value dim = getIntValue(dim_raw, rewriter, context, loc);
      result = rewriter.create<AtenSqueezeDimOp>(loc, resultType, result, dim);
    }
  } else {
    result = rewriter.create<AtenSqueezeOp>(loc, resultType, dataTensor);
  }

  return result;
}

/**
 *
 * ONNX SqueezeV13/SqueezeV11
 *
 * Remove single-dimensional entries from the shape of a tensor Takes an input
 * axes with a list of axes to squeeze. If axes is not provided, all the
 * single dimensions will be removed from the shape. If an axis is selected
 * with shape entry not equal to one, an error is raised.
 *
 *
 * Operands :
 *   data       tensor of 8-bit unsigned integer values or tensor of 16-bit
 * unsigned integer values or tensor of 32-bit unsigned integer values or tensor
 * of 64-bit unsigned integer values or tensor of 8-bit signless integer values
 * or tensor of 16-bit signless integer values or tensor of 32-bit signless
 * integer values or tensor of 64-bit signless integer values or tensor of
 * bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit
 * float values or tensor of 64-bit float values or tensor of string type values
 * or tensor of 1-bit signless integer values or tensor of complex type with
 * 32-bit float elements values or tensor of complex type with 64-bit float
 * elements values or memref of any type values
 *
 *   axes       (attr/operand) tensor of 64-bit signless integer values or
 * memref of any type values or none type
 *
 *
 * Output   :
 *   output   	tensor of 8-bit unsigned integer
 * values or tensor of 16-bit unsigned integer values or tensor of 32-bit
 * unsigned integer values or tensor of 64-bit unsigned integer values or tensor
 * of 8-bit signless integer values or tensor of 16-bit signless integer values
 * or tensor of 32-bit signless integer values or tensor of 64-bit signless
 * integer values or tensor of bfloat16 type values or tensor of 16-bit float
 * values or tensor of 32-bit float values or tensor of 64-bit float values or
 * tensor of string type values or tensor of 1-bit signless integer values or
 * tensor of complex type with 32-bit float elements values or tensor of complex
 * type with 64-bit float elements values or memref of any type values
 *
 *
 */

template <typename SqueezeOp>
struct ONNXToTorchSqueezeOpLowering : public ConversionPattern {
  ONNXToTorchSqueezeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, SqueezeOp::getOperationName(), 1,
                          ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    SqueezeOp squeezeOp = llvm::dyn_cast_or_null<SqueezeOp>(op);

    assert(squeezeOp && "Expecting op to have a strong type");

    Location loc = squeezeOp.getLoc();
    mlir::MLIRContext *context = squeezeOp.getContext();

    auto axes = getAxes(squeezeOp);
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
  patterns.insert<ONNXToTorchSqueezeOpLowering<ONNXSqueezeV11Op>,
                  ONNXToTorchSqueezeOpLowering<ONNXSqueezeOp>>(typeConverter,
                                                               ctx);
}
