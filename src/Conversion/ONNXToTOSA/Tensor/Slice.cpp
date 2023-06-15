/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Slice.cpp - Slice Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX SliceOp to TOSA dialect.
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

class ONNXSliceLoweringToTOSA : public OpConversionPattern<ONNXSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXSliceOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    // Get shape.
    IndexExprBuilderForTosa createTosaIE(rewriter, loc);
    ONNXSliceOpShapeHelper shapeHelper(op, {}, &createTosaIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getData();
    if (!(IndexExpr::isLiteral(shapeHelper.starts)))
      return rewriter.notifyMatchFailure(op, "starts has no literals.");
    if (!(IndexExpr::isLiteral(shapeHelper.ends)))
      return rewriter.notifyMatchFailure(op, "ends has no literals.");
    if (!(IndexExpr::isLiteral(shapeHelper.steps)))
      return rewriter.notifyMatchFailure(op, "steps has no literals.");

    llvm::SmallVector<int64_t, 4> starts;
    IndexExpr::getLiteral(shapeHelper.starts, starts);
    llvm::SmallVector<int64_t, 4> ends;
    IndexExpr::getLiteral(shapeHelper.ends, ends);
    llvm::SmallVector<int64_t, 4> steps;
    IndexExpr::getLiteral(shapeHelper.steps, steps);

    for (const int64_t step : steps) {
      if (step != 1)
        return rewriter.notifyMatchFailure(
            op, "TOSA only supports step size 1.");
    }

    llvm::SmallVector<int64_t, 4> size;
    size.resize(starts.size());
    for (size_t i = 0; i < starts.size(); i++) {
      size[i] = ends[i] - starts[i];
    }

    Value newSliceOp = tosaBuilder.slice(input, size, starts);

    rewriter.replaceOp(op, newSliceOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXSliceOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSliceLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
