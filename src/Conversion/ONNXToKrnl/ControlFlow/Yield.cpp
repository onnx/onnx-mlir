/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- Yield.cpp - Lowering Yield Op ------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Yield Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXYieldOpLowering : public OpConversionPattern<ONNXYieldOp> {
  ONNXYieldOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXYieldOp yieldOp, ONNXYieldOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Operation *op = yieldOp.getOperation();
    Location loc = ONNXLoc<ONNXYieldOp>(op);

    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);

    ValueRange inputs = yieldOp.getOperands();
    llvm::SmallVector<Value> outputs;
    for (Value input : inputs) {
      Type inputType = input.getType();
      Type outputType = typeConverter->convertType(inputType);
      outputs.emplace_back(typeConverter->materializeTargetConversion(
          rewriter, loc, outputType, input));
    }

    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, outputs);

    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXYieldOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXYieldOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
