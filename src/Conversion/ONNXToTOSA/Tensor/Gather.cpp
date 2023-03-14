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

    auto loc = op.getLoc();

    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getData();
    Value indices = adaptor.getIndices();
    int64_t axis = adaptor.getAxis();

    auto result = op.getResult();

    auto inputType = input.getType();
    if (!onnx_mlir::isRankedShapedType(inputType))
      return rewriter.notifyMatchFailure(op, "input is not a ranked tensor");
    int64_t inputRank = onnx_mlir::getRank(inputType);

    // onnx allows values beetween [-r, r-1] where r is the rank
    axis = tosa::convertNegativeAxis(axis, inputRank);

    auto indicesType = indices.getType().cast<ShapedType>();
    SmallVector<int32_t, 4> newIndicesValues;
    newIndicesValues.resize(indicesType.getNumElements());

    auto indicesValues = tosa::getValueFromTosaConst<ElementsAttr>(indices);

    // ONNX allows negative indices and TOSA doesn't. This iterates through each
    // index and rescales if necessary.
    ArrayRef<int64_t> inputShape = inputType.cast<ShapedType>().getShape();
    auto indicesAttrValues = indicesValues.getValues<APInt>();
    for (const auto &[index, value] : llvm::enumerate(indicesAttrValues)) {
      int64_t numericalValue = value.getSExtValue();
      if (numericalValue < 0)
        newIndicesValues[index] = (int32_t)(numericalValue + inputShape[axis]);
      else
        newIndicesValues[index] = (int32_t)(numericalValue);
    }

    Value newIndices =
        tosaBuilder.getConst(newIndicesValues, indicesType.getShape());

    auto newGather =
        tosaBuilder.gather(result, input, newIndices, 0, (int32_t)axis);

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
