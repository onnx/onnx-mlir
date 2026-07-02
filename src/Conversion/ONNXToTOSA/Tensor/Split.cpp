/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Split.cpp - Split Op -----------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX split operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXSplitOpLoweringToTOSA : public OpConversionPattern<ONNXSplitOp> {
public:
  using OpConversionPattern<ONNXSplitOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXSplitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op.getLoc());

    Value input = adaptor.getInput();
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    int64_t rank = inputType.getRank();
    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis is out of range");

    llvm::SmallVector<Value, 4> newResults;
    llvm::SmallVector<int64_t, 4> starts(rank, 0);
    int64_t offset = 0;
    for (Value result : op.getResults()) {
      auto resultType = mlir::dyn_cast<RankedTensorType>(result.getType());
      if (!resultType || !resultType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            op, "all outputs must have a static shape");

      llvm::SmallVector<int64_t, 4> sizes(resultType.getShape());
      starts[axis] = offset;
      newResults.push_back(tosaBuilder.slice(input, sizes, starts));
      offset += resultType.getShape()[axis];
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

} // namespace

void populateLoweringONNXSplitOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
