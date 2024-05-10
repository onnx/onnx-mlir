/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Unsqueeze.cpp - Unsqueeze Op ------------------------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX UnsqueezeOp to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

template <typename SqueezeOp, typename ShapeHelper>
class ONNXUnsqueezeSqueezeLoweringToTOSA
    : public OpConversionPattern<SqueezeOp> {
public:
  using OpConversionPattern<SqueezeOp>::OpConversionPattern;
  using OpAdaptor = typename SqueezeOp::Adaptor;
  LogicalResult matchAndRewrite(SqueezeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();

    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTy || !resultTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type with static shape");
    }

    TosaBuilder tosaBuilder(rewriter, loc);
    rewriter.replaceOp(
        op, tosaBuilder.reshape(adaptor.getData(), resultTy.getShape()));
    return success();
  }
};

} // namespace

void populateLoweringONNXSqueezeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXUnsqueezeSqueezeLoweringToTOSA<ONNXUnsqueezeOp,
                      ONNXUnsqueezeOpShapeHelper>,
      ONNXUnsqueezeSqueezeLoweringToTOSA<ONNXSqueezeOp,
          ONNXSqueezeOpShapeHelper>>(ctx);
}

} // namespace onnx_mlir
