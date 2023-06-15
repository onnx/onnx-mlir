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
#include "llvm/ADT/None.h"
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

    float valueFloat = 0.0F;
    if (!constValue.getType().dyn_cast<NoneType>()) {
      auto valueAttr =
          tosa::getValueFromTosaConst<ElementsAttr>(constValue);
      auto valueIt = valueAttr.getValues<FloatAttr>().begin();
      // Need float for F32 Type
      float valueFloat = (*valueIt).cast<FloatAttr>().getValueAsDouble();

      TosaBuilder tosaBuilder(rewriter, loc);
      Value constTosaTensor =
          tosaBuilder.getSplattedConst(valueFloat);

      rewriter.replaceOpWithNewOp<mlir::tosa::PadOp>(
          op, resultType, data, padsList1, constTosaTensor);
    } else {
        auto constType = RankedTensorType::get({}, rewriter.getF32Type());
        auto constAttr = DenseElementsAttr::get(constType, valueFloat);
        Value constTosaTensor = rewriter.create<mlir::tosa::ConstOp>(
            op->getLoc(), constType, constAttr);

        rewriter.replaceOpWithNewOp<mlir::tosa::PadOp>(
            op, resultType, data, padsList1, constTosaTensor);
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