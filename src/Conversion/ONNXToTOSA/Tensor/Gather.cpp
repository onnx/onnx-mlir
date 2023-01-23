/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gather.cpp - Gather Op ------------------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX GatherOp to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXGatherLoweringToTOSA : public OpConversionPattern<ONNXGatherOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXGatherOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXGatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.data();
    Value indices = adaptor.indices();
    int64_t axis = adaptor.axis();
    auto inputType = input.getType();
    if (!onnx_mlir::isRankedShapedType(inputType))
      return rewriter.notifyMatchFailure(op, "input is not a ranked tensor");
    int64_t inputRank = onnx_mlir::getRank(inputType);

    // onnx allows values beetween [-r, r-1] where r is the rank
    if (axis < 0) {
      axis += inputRank;
    }

    auto indicesType = indices.getType().cast<ShapedType>();
    SmallVector<int32_t, 4> newIndicesValues;
    newIndicesValues.resize(indicesType.getNumElements());

    auto indicesValues = tosa::getValueFromTosaConst<ElementsAttr>(indices);

    ArrayRef<int64_t> inputShape = inputType.cast<ShapedType>().getShape();
    auto indicesAttrValues = indicesValues.getValues<APInt>();
    for (const auto [index, value] : llvm::enumerate(indicesAttrValues)) {
      int64_t numericalValue = value.getSExtValue();
      if (numericalValue < 0)
        newIndicesValues[index] = (int32_t)(numericalValue + inputShape[axis]);
      else
        newIndicesValues[index] = (int32_t)(numericalValue);
    }

    llvm::Optional<Value> newIndices = tosa::getConstTensor<int32_t>(
        rewriter, op, newIndicesValues, indicesType.getShape());
    if (!newIndices.has_value())
      return rewriter.notifyMatchFailure(
          op, "could not determine input indices");

    auto newGather = tosa::convertGatherOp(rewriter, op, op.getResult(), input,
        newIndices.value(), 0, (int32_t)axis);

    if (!newGather.has_value()) {
      return failure();
    }
    rewriter.replaceOp(op, newGather.value());

    return success();
  }
};

} // namespace

void populateLoweringONNXGatherOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXGatherLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
