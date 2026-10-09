//===------------ Identity.cpp - Lowering MatMul Op to Linalg -------------===//
//
// Copyright 2026 HighTec-EDV Systeme Gmbh Authors.
//
// =============================================================================
//
// This file lowers the ONNX Identity Operator to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXIdentityOpLoweringToLinalg
    : public OpRewritePattern<ONNXIdentityOp> {
  ONNXIdentityOpLoweringToLinalg(
      MLIRContext *ctx, const std::string &linalgOps, bool useLinalgPath)
      : OpRewritePattern<ONNXIdentityOp>(ctx), linalgOps(linalgOps),
        useLinalgPath(useLinalgPath) {}

  LogicalResult matchAndRewrite(
      ONNXIdentityOp identityOp, PatternRewriter &rewriter) const final {
    if (!shouldConvertToLinalg(
            identityOp.getOperation(), linalgOps, useLinalgPath)) {
      return rewriter.notifyMatchFailure(
          identityOp, "operation not selected for Linalg conversion");
    }

    if (identityOp.getInput().getType() != identityOp.getResult().getType())
      return rewriter.notifyMatchFailure(
          identityOp, "input and result types must match");

    // Simply replacing all uses of the value created by the Identity op with
    // the value given to Identity.
    rewriter.replaceOp(identityOp, identityOp.getInput());

    return success();
  }

private:
  std::string linalgOps;
  bool useLinalgPath;
};

void populateLoweringONNXIdentityOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx,
    const std::string &linalgOps, bool useLinalgPath) {
  patterns.add<ONNXIdentityOpLoweringToLinalg>(ctx, linalgOps, useLinalgPath);
}
} // namespace onnx_mlir
