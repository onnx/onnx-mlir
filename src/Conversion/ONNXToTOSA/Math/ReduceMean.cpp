/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ReduceMean.cpp - ReduceMean Op --------------------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX reduce mean operator to TOSA dialect.
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

class ONNXReduceMeanLoweringToTOSA
    : public OpConversionPattern<ONNXReduceMeanOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXReduceMeanOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXReduceMeanOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getData();
    Value axesValue = adaptor.getAxes();
    auto keepDims = adaptor.getKeepdims();
    auto noOpIfAxesEmpty = adaptor.getNoopWithEmptyAxes();

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    RankedTensorType inputType =
        mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input type not a ranked tensor.");

    // axes is mandatory for tosa
    llvm::SmallVector<int64_t, 4> axesVec;
    if (isNoneValue(axesValue) && !noOpIfAxesEmpty) {
      // if not present all axes are reduced
      const int64_t numberOfAxes =
          mlir::cast<ShapedType>(input.getType()).getRank();
      llvm::SmallVector<int64_t> allDims(numberOfAxes);
      std::iota(std::begin(allDims), std::end(allDims), 0);
      axesVec.append(allDims);
    } else if (axesValue.getDefiningOp<mlir::tosa::ConstOp>()) {
      // if input is a tosa const get axes
      auto axes = tosa::getValueFromTosaConst<ElementsAttr>(axesValue);
      auto axesElementsValues = axes.getValues<int64_t>();
      llvm::transform(axesElementsValues, std::back_inserter(axesVec),
          [](int64_t axesInt) { return axesInt; });
    }
    // Tosa needs a DenseElementsAttr
    const int64_t vecValuesSize = axesVec.size();
    DenseElementsAttr newAxesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({vecValuesSize}, rewriter.getI64Type()), axesVec);

    // reduce_mean is lowered as followed:
    // op1 = reduce_sum(input)
    // op2 = mul(op1, 1.0 / num_elements_on_reduced_axis)

    int64_t inputRank = inputType.getRank();
    int64_t numElemsOnReducedAxis = 1;
    for (int i = 0; i < newAxesAttr.getNumElements(); i++) {
      int64_t axisVal = newAxesAttr.getValues<mlir::IntegerAttr>()[i].getInt();
      if (axisVal < 0)
        axisVal += inputRank;
      numElemsOnReducedAxis *= inputType.getShape()[axisVal];
    }
    double divScale = 1.0 / static_cast<double>(numElemsOnReducedAxis);
    Type reduceElementType = inputType.getElementType();

    auto val = onnx_mlir::tosa::convertReduceOpCommon<mlir::tosa::ReduceSumOp>(
        rewriter, op, outputType, input, newAxesAttr, keepDims,
        reduceElementType);

    if (!val.has_value())
      return rewriter.notifyMatchFailure(
          op, "could not convert generic reduce op.");

    Value divConst = tosaBuilder.getSplattedConst(divScale);
    auto output = tosaBuilder.mul(val.value(), divConst);

    if (!output) {
      return rewriter.notifyMatchFailure(op, "could not be converted");
    }
    // Shape inference is handled by the helper functions
    rewriter.replaceOp(op, {output});
    return success();
  }
};

} // namespace

void populateLoweringONNXReduceMeanOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXReduceMeanLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
