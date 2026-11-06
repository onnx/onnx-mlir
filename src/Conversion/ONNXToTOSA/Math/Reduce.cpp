/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Reduce.cpp - Reduce Ops ---------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX reduce operators to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <numeric>

using namespace mlir;

namespace onnx_mlir {

namespace {

template <typename ONNXReduceOp, typename TosaReduceOp>
class ONNXReduceOpLoweringToTOSA : public OpConversionPattern<ONNXReduceOp> {
public:
  using OpConversionPattern<ONNXReduceOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXReduceOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getData();
    Value axesValue = adaptor.getAxes();
    auto keepDims = adaptor.getKeepdims();
    auto noOpIfAxesEmpty = adaptor.getNoopWithEmptyAxes();

    auto outputType = mlir::dyn_cast_or_null<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!outputType)
      return rewriter.notifyMatchFailure(
          op, "output type not a ranked tensor.");

    RankedTensorType inputType =
        mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input type not a ranked tensor.");

    // axes is mandatory for tosa
    llvm::SmallVector<int64_t, 4> axesVec;
    if (isNoneValue(axesValue)) {
      // if not present all axes are reduced
      if (!noOpIfAxesEmpty) {
        const int64_t numberOfAxes = inputType.getRank();
        axesVec.resize(numberOfAxes);
        std::iota(std::begin(axesVec), std::end(axesVec), 0);
      }
    } else if (axesValue.getDefiningOp<mlir::tosa::ConstOp>()) {
      // if input is a tosa const get axes
      auto axes = tosa::getValueFromTosaConst<ElementsAttr>(axesValue);
      for (int64_t axis : axes.getValues<int64_t>())
        axesVec.push_back(axis);
    } else {
      return rewriter.notifyMatchFailure(
          op, "only constant axes are supported.");
    }

    // Tosa needs a DenseElementsAttr
    const int64_t axesSize = axesVec.size();
    DenseElementsAttr newAxesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({axesSize}, rewriter.getI64Type()), axesVec);

    std::optional<Value> reduced =
        onnx_mlir::tosa::convertReduceOpCommon<TosaReduceOp>(rewriter, op,
            outputType, input, newAxesAttr, keepDims,
            inputType.getElementType());

    if (!reduced.has_value())
      return rewriter.notifyMatchFailure(
          op, "could not convert generic reduce op.");

    // Shape inference is handled by the helper functions.
    rewriter.replaceOp(op, {reduced.value()});
    return success();
  }
};

} // namespace

void populateLoweringONNXReduceOpsToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<
      ONNXReduceOpLoweringToTOSA<ONNXReduceMaxOp, mlir::tosa::ReduceMaxOp>,
      ONNXReduceOpLoweringToTOSA<ONNXReduceMinOp, mlir::tosa::ReduceMinOp>,
      ONNXReduceOpLoweringToTOSA<ONNXReduceProdOp, mlir::tosa::ReduceProductOp>,
      ONNXReduceOpLoweringToTOSA<ONNXReduceSumOp, mlir::tosa::ReduceSumOp>>(
      typeConverter, ctx);
}

} // namespace onnx_mlir
