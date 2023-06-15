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
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
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
    auto loc = op.getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);
    Value input = adaptor.getData();

    auto outputType = op.getResult().getType().dyn_cast<TensorType>();

    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "not a ranked tensor");
    }

    if (!outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "only static shapes are supported");
    }

    if (adaptor.getAllowzero() != 0) {
      return rewriter.notifyMatchFailure(op, "only allowZero = 0 is supported");
    }

    if (!adaptor.getShape().getDefiningOp<mlir::tosa::ConstOp>()) {
      return rewriter.notifyMatchFailure(
          op, "only tosa.const operands are supported");
    }

    Value reshapeOp = tosaBuilder.reshape(input, outputType.getShape());

    rewriter.replaceOp(op, reshapeOp);

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
