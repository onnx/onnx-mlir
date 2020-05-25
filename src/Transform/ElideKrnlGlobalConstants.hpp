#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"

/*!
 *  RewritePattern that replaces existing constant Krnl global values
 *  with a similar operation which preserves all attributes except the value
 *  attribute.
 */
class KrnlConstGlobalValueElision
    : public mlir::OpRewritePattern<mlir::KrnlGlobalOp> {
public:
  using mlir::OpRewritePattern<mlir::KrnlGlobalOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::KrnlGlobalOp op, mlir::PatternRewriter &rewriter) const override;
};