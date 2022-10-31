/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Reshape.cpp - Reshape Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX ReshapeOp to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXReshapeLoweringToTOSA : public OpConversionPattern<ONNXReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXReshapeOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto outputType =
        op.getResult().getType().dyn_cast<TensorType>();

    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "not a ranked tensor");
    }

    if (adaptor.allowzero() != 0) {
      return rewriter.notifyMatchFailure(op, "only allowZero = 0 is supported");
    }

    if (!adaptor.shape().getDefiningOp<tosa::ConstOp>()) {
      return rewriter.notifyMatchFailure(
          op, "only tosa.const operands are supported");
    }
    Value shapeConst = adaptor.shape().getDefiningOp<tosa::ConstOp>();
    auto shapeConstAttr =
        tosa::getValueFromTosaConst<ElementsAttr>(shapeConst);
    for (APInt i : shapeConstAttr.getValues<APInt>()) {
      if (i.getZExtValue() == 0) {
        return rewriter.notifyMatchFailure(op, "zero shape not allowed");
      }
    }

    llvm::SmallVector<int64_t> shapeValues;
    for (int i = 0; i < outputType.getShape().size(); i++) {
      shapeValues.push_back(outputType.getShape()[i]);
    }
    ArrayAttr shapeAttr = rewriter.getI64ArrayAttr(shapeValues);

    tosa::CreateReplaceOpAndInfer<tosa::ReshapeOp>(
        rewriter, op, outputType, adaptor.data(), shapeAttr);
    return success();
  }
};

} // namespace

void populateLoweringONNXReshapeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXReshapeLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
