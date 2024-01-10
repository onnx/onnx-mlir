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

template <typename SqueezeOp, typename ShapeHelper>
class ONNXUnsqueezeSqueezeLoweringToTOSA
    : public OpConversionPattern<SqueezeOp> {
public:
  using OpConversionPattern<SqueezeOp>::OpConversionPattern;
  using OpAdaptor = typename SqueezeOp::Adaptor;
  LogicalResult matchAndRewrite(SqueezeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    // Get shape.
    IndexExprBuilderForTosa createTosaIE(rewriter, loc);
    ShapeHelper shapeHelper(op, {}, &createTosaIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getData();

    DimsExpr outputDims = shapeHelper.getOutputDims();
    llvm::SmallVector<int64_t, 4> outputDimsVec;
    IndexExpr::getLiteral(outputDims, outputDimsVec);

    Value newUnsqueezeOp = tosaBuilder.reshape(input, outputDimsVec);

    rewriter.replaceOp(op, newUnsqueezeOp);
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
