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

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXUnsqueezeLoweringToTOSA
    : public OpConversionPattern<ONNXUnsqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXUnsqueezeOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXUnsqueezeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    // Get shape.
    IndexExprBuilderForTosa createTosaIE(rewriter, loc);
    ONNXUnsqueezeOpShapeHelper shapeHelper(op, {}, &createTosaIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.data();

    DimsExpr outputDims = shapeHelper.getOutputDims();
    auto outputDimsVec = tosa::createInt64VectorFromIndexExpr(outputDims);

    Value newUnsqueezeOp = tosaBuilder.reshape(input, outputDimsVec);

    rewriter.replaceOp(op, newUnsqueezeOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXUnsqueezeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXUnsqueezeLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
