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
  /*
   * A threshold value specifying the maximum number of elements a  constant
   * operation can hold as an attribute. If the number exceeds this threshold,
   * constants will be packed together and, in the case where `move-to-file`
   * option is enabled, stored as a  binary file on disk. This can help preserve
   * readability of IR dump and improve compilation speed.
   */
  static const int64_t kDefaultElisionThreshold;

  int64_t elisionThreshold;

  using mlir::OpRewritePattern<mlir::KrnlGlobalOp>::OpRewritePattern;

  explicit KrnlConstGlobalValueElision(mlir::MLIRContext *context,
      int64_t elisionThreshold)
      : OpRewritePattern(context), elisionThreshold(elisionThreshold) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::KrnlGlobalOp op, mlir::PatternRewriter &rewriter) const override;
};
