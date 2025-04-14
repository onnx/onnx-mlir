/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Padding.cpp - Padding Op --------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX padding operator to TOSA dialect.
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
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace onnx_mlir {

class ONNXPadOpLoweringToTOSA : public OpConversionPattern<ONNXPadOp> {
public:
  using OpConversionPattern<ONNXPadOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXPadOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXPadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op.getLoc();

    Value data = adaptor.getData();
    Value pads = adaptor.getPads();
    Value constValue = adaptor.getConstantValue();

    auto dataType = dyn_cast<RankedTensorType>(data.getType());
    if (!dataType || !dataType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "input type has no static shape");
    }

    auto elementDtype = dataType.getElementType();
    if (!isa<FloatType>(elementDtype) && !isTOSAInt(elementDtype)) {
      return rewriter.notifyMatchFailure(op, "unsupported type");
    }

    if (!adaptor.getAxes().getDefiningOp<ONNXNoneOp>()) {
      return rewriter.notifyMatchFailure(op, "only default axes are supported");
    }

    if (!(adaptor.getMode() == "constant")) {
      return rewriter.notifyMatchFailure(
          op, "only 'constant' mode is supported");
    }

    if (!pads.getDefiningOp<mlir::tosa::ConstOp>() ||
        !(constValue.getDefiningOp<mlir::tosa::ConstOp>() ||
            constValue.getDefiningOp<ONNXNoneOp>())) {
      return rewriter.notifyMatchFailure(
          op, "only tosa.const operands are supported");
    }
    // creating the DenseElementsAttr using pads values.
    auto denseAttr = tosa::getValueFromTosaConst<ElementsAttr>(pads);

    // Reading the ONNX side pads values and store in the array.
    llvm::SmallVector<int64_t, 8> intValues;
    bool paddingNeeded = false;
    for (auto n : denseAttr.getValues<APInt>()) {
      intValues.push_back(n.getZExtValue());
      if (!n.isZero())
        paddingNeeded = true;
    }
    if (!paddingNeeded) {
      // We do not need to represent the no-op pad in the resulting MLIR
      rewriter.replaceOp(op, {data});
      return success();
    }

    Value padsList1 =
        tosa::buildOnnxToTosaPaddingConstOp(rewriter, intValues, loc, {}, {});

    mlir::Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (!isa<NoneType>(constValue.getType())) {
      auto valueAttr = tosa::getValueFromTosaConst<ElementsAttr>(constValue);
      TosaBuilder tosaBuilder(rewriter, loc);

      Value constTosaTensor;
      if (isa<FloatType>(valueAttr.getElementType())) {
        auto valueIt = valueAttr.getValues<FloatAttr>().begin();
        const float valueFloat = cast<FloatAttr>(*valueIt).getValueAsDouble();
        constTosaTensor = tosaBuilder.getSplattedConst(
            valueFloat, valueAttr.getElementType(), 0);
      } else {
        assert(isTOSAInt(elementDtype) && "Already validated");
        auto valueIt = valueAttr.getValues<IntegerAttr>().begin();
        auto valueAsAPInt = cast<IntegerAttr>(*valueIt).getValue();
        auto asIntegerTy = cast<IntegerType>(valueAttr.getElementType());
        if (asIntegerTy.isUnsigned()) {
          constTosaTensor = tosaBuilder.getSplattedConst(
              valueAsAPInt.getZExtValue(), asIntegerTy, 0);
        } else {
          constTosaTensor = tosaBuilder.getSplattedConst(
              valueAsAPInt.getSExtValue(), asIntegerTy, 0);
        }
      }
      rewriter.replaceOpWithNewOp<mlir::tosa::PadOp>(
          op, resultType, data, padsList1, constTosaTensor);

    } else {
      auto constType = RankedTensorType::get({}, elementDtype);

      DenseElementsAttr constAttr;
      if (auto floatType = dyn_cast<FloatType>(elementDtype)) {
        constAttr = DenseElementsAttr::get(
            constType, APFloat::getZero(floatType.getFloatSemantics()));
      } else {
        assert(isTOSAInt(elementDtype) && "Already validated");
        auto tyAsInt = cast<IntegerType>(elementDtype);
        constAttr = DenseElementsAttr::get(constType,
            llvm::APInt(tyAsInt.getWidth(), 0, tyAsInt.getSignedness()));
      }

      rewriter.replaceOpWithNewOp<mlir::tosa::PadOp>(op, resultType, data,
          padsList1,
          rewriter.create<mlir::tosa::ConstOp>(
              op->getLoc(), constType, constAttr));
    }

    return success();
  }
};

void populateLoweringONNXPadOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXPadOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
