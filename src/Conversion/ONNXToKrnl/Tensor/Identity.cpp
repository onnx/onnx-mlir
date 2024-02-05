/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Identity.cpp - Lowering Identity Op ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNXIdentity operator to the Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXIdentityOpLowering : public OpConversionPattern<ONNXIdentityOp> {
  ONNXIdentityOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXIdentityOp identityOp,
      ONNXIdentityOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(identityOp, adaptor.getInput());
    // No work, no need to report on SIMD.
    return success();
  }
};

void populateLoweringONNXIdentityOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXIdentityOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
