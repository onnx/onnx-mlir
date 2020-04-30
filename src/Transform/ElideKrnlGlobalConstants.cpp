//===- ElideKrnlGlobalConstants.cpp - Krnl Constant lobal Value Elision ---===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// In practice, the constant values of Global Krnl operations may be large
// enough to hinder the readability of the MLIR intermediate representation.
//
// This file creates a pass which elides the explicit values of constant
// global operations. This pass has purely cosmetic purposes and should only be
// run to obtain a compact representation of the program when emitting Krnl
// dialect code. This pass should never be invoked on code meant to be run.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/*!
 *  RewritePattern that replaces existing constant Krnl global values
 *  with a similar operation which preserves all attributes except the value
 *  attribute.
 */

class KrnlConstGlobalValueElision : public OpRewritePattern<KrnlGlobalOp> {
public:
  using OpRewritePattern<KrnlGlobalOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlGlobalOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    if (op.value().hasValue()) {
      auto newGlobalOp = rewriter.create<KrnlGlobalOp>(
          loc, op.getResult().getType(), op.shape(), op.name(), nullptr);
      rewriter.replaceOp(op, newGlobalOp.getResult());
    }

    return success();
  }
};

/*!
 *  Function pass that performs constant value elision of Krnl globals.
 */
class ElideConstGlobalValuePass
    : public PassWrapper<ElideConstGlobalValuePass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<KrnlConstGlobalValueElision>(&getContext());

    applyPatternsAndFoldGreedily(function, patterns);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createElideConstGlobalValuePass() {
  return std::make_unique<ElideConstGlobalValuePass>();
}

static PassRegistration<ElideConstGlobalValuePass> pass("elide-krnl-constants",
    "Elide the constant values of the Global Krnl operations.");