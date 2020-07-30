//===-------- EnableMemoryPool.cpp - Enable Memory Pool for MemRefs -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// For certain cases the number of individual memory allocations required for
// all internal tensors is large and needs to be mitigated. This pass enables a
// managed memory pool for allocating MemRefs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

class KrnlEliminateCmpI : public OpRewritePattern<CmpIOp> {
public:
  using OpRewritePattern<CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      CmpIOp cmpIOp, PatternRewriter &rewriter) const override {
    auto loc = cmpIOp.getLoc();
    Value replacement = nullptr;

    // Operands of the compare are equal.
    if (cmpIOp.getOperands()[0] == cmpIOp.getOperands()[1]) {
      // Cmp Ops which are false when operands are the same.
      if (cmpIOp.getPredicate() == CmpIPredicate::sgt ||
          cmpIOp.getPredicate() == CmpIPredicate::slt ||
          cmpIOp.getPredicate() == CmpIPredicate::ugt ||
          cmpIOp.getPredicate() == CmpIPredicate::ult ||
          cmpIOp.getPredicate() == CmpIPredicate::ne)
        replacement = rewriter.create<ConstantOp>(
            loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 0));

      // Cmp Ops which are true when operands are the same.
      if (cmpIOp.getPredicate() == CmpIPredicate::eq ||
          cmpIOp.getPredicate() == CmpIPredicate::sle ||
          cmpIOp.getPredicate() == CmpIPredicate::ule ||
          cmpIOp.getPredicate() == CmpIPredicate::sge ||
          cmpIOp.getPredicate() == CmpIPredicate::uge)
        replacement = rewriter.create<ConstantOp>(
            loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    }

    if (!replacement)
      return failure();

    rewriter.replaceOp(cmpIOp, replacement);
    return success();
  }
};

/*!
 *  Function pass that eliminates redundant instructions before leaving the
 *  Krnl abstraction level.
 */
class SimplifyKrnlPass : public PassWrapper<SimplifyKrnlPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<KrnlEliminateCmpI>(&getContext());

    applyPatternsAndFoldGreedily(function, patterns);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createSimplifyKrnlPass() {
  return std::make_unique<SimplifyKrnlPass>();
}
