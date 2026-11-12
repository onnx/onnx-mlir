/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Attention.cpp - Lowering Attention Op ----------------===//
//
// Copyright 2024-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Attention Operator to Krnl dialect by decomposing
// it into basic ONNX operations (MatMul, Transpose, Softmax, Add, Mul, etc.).
// The resulting ONNX operations are then lowered to Krnl by their respective
// lowering patterns. The decomposition itself lives in
// src/Dialect/ONNX/ONNXOps/AttentionToONNXOps.cpp, shared with the NNPA
// RewriteONNXForZHigh fallback pattern.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/AttentionToONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXAttentionOpLowering : public OpConversionPattern<ONNXAttentionOp> {
  ONNXAttentionOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXAttentionOp>(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXAttentionOp attentionOp,
      ONNXAttentionOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    return lowerONNXAttentionOp(attentionOp, adaptor.getQ(), adaptor.getK(),
        adaptor.getV(), adaptor.getAttnMask(), adaptor.getPastKey(),
        adaptor.getPastValue(), adaptor.getNonpadKvSeqlen(), rewriter);
  }
};

void populateLoweringONNXAttentionOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXAttentionOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
