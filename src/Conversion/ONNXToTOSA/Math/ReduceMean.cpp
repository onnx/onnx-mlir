/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ReduceMean.cpp - ReduceMean Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX reduce mean operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <iterator>
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

    Value input = adaptor.data();
    auto axes = adaptor.axes();
    auto keepDims = adaptor.keepdims();

    auto resultType = getTypeConverter()
                          ->convertType(op.getResult().getType())
                          .cast<RankedTensorType>();

    if (!axes) {
      const int64_t numberOfAxes = input.getType().cast<ShapedType>().getRank();
      std::vector<int64_t> allDims(numberOfAxes);
      std::iota(std::begin(allDims), std::end(allDims), 0);
      axes = rewriter.getI64ArrayAttr(allDims);
    }

    auto vecValues = extractFromI64ArrayAttr(axes.value());
    const int64_t vecValuesSize = vecValues.size();
    DenseElementsAttr newAxesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({vecValuesSize}, rewriter.getI64Type()),
        vecValues);

    auto output = tosa::convertReduceMeanOp(
        rewriter, op, resultType, input, newAxesAttr, keepDims);

    if (!output) {
      return rewriter.notifyMatchFailure(op, "Could not be converted");
    }
    rewriter.replaceOp(op, {output.value()});
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
