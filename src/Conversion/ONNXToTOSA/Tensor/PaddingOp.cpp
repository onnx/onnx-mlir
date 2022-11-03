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

    Value data = adaptor.data();
    Value pads = adaptor.pads();
    Value constValue = adaptor.constant_value();

    if (!(adaptor.mode() == "constant")) {
      return rewriter.notifyMatchFailure(
          op, "Only 'constant' mode is supported");
    }

    if (!pads.getDefiningOp<tosa::ConstOp>() ||
        !(constValue.getDefiningOp<tosa::ConstOp>() ||
            constValue.getDefiningOp<ONNXNoneOp>())) {
      return rewriter.notifyMatchFailure(
          op, "Only tosa.const operands are supported");
    }
    // creating the DenseElementsAttr using pads values.
    ElementsAttr denseAttr = mlir::onnx_mlir::getValueFromTosaConst<ElementsAttr>(pads);

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

    // Create a new pad vec in the right format
    // ONNX : [b1, b2, b3, b4, e1, e2, e3, e4]
    // TOSA :[[b1, e1], [b2, e2], [b3, e3], [b4, e4]]
    llvm::SmallVector<int64_t, 8> translatePadsList;

    const unsigned int dimSize = intValues.size() / 2;
    for (unsigned int i = 0; i < dimSize; i++) {
      translatePadsList.push_back(intValues[i]);
      translatePadsList.push_back(intValues[i + dimSize]);
    }

    const unsigned int numberOfDims = intValues.size() / 2;
    DenseElementsAttr paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({numberOfDims, 2}, rewriter.getI64Type()),
        translatePadsList);

    Value padsList1 =
        rewriter.create<tosa::ConstOp>(loc, paddingAttr.getType(), paddingAttr);

    mlir::Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (!constValue.getType().dyn_cast<NoneType>()) {
      ElementsAttr valueAttr =
          mlir::onnx_mlir::getValueFromTosaConst<ElementsAttr>(constValue);
      auto valueIt = valueAttr.getValues<FloatAttr>().begin();
      // Need float for F32 Type
      float valueFloat = (*valueIt).cast<FloatAttr>().getValueAsDouble();
      auto constType = RankedTensorType::get({}, rewriter.getF32Type());
      auto constAttr = DenseElementsAttr::get(constType, valueFloat);
      Value constTosaTensor =
          rewriter.create<tosa::ConstOp>(op->getLoc(), constType, constAttr);

      rewriter.replaceOpWithNewOp<tosa::PadOp>(
          op, resultType, data, padsList1, constTosaTensor);
    } else {
      rewriter.replaceOpWithNewOp<tosa::PadOp>(op, resultType, data, padsList1);
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