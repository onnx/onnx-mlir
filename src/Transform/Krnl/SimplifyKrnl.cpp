/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

namespace {
struct RemoveRedundantKrnlDummyCast : public OpRewritePattern<KrnlDummyCastOp> {
public:
  using OpRewritePattern<KrnlDummyCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      KrnlDummyCastOp op, PatternRewriter &rewriter) const override {
    if (op.getOperand().getType() == op.getResult().getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }
    return failure();
  }
};
} // namespace

void KrnlDummyCastOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.insert<RemoveRedundantKrnlDummyCast>(context);
}
