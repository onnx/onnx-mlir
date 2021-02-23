/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- ElideConstants.cpp - Elide Constant Values ---------------------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// In practice, the constant values of Constant operations may be large enough
// to hinder the readability of the MLIR intermediate representation.
//
// This file creates a pass which elides the explicit values of Constant
// operations. This pass has purely cosmetic purposes and should only be run to
// obtain a compact representation of the program when emitting ONNX and KRNL
// Dialect code. This pass should never be invoked on code meant to be run.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/*!
 *  RewritePattern that replaces existing Constant operations
 *  with Constant operations with the same shape information but
 *  no values.
 */

class ConstantValueElision : public OpRewritePattern<ONNXConstantOp> {
public:
  using OpRewritePattern<ONNXConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConstantOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto constOp = llvm::dyn_cast<ONNXConstantOp>(&op);

    if (constOp->sparse_value().hasValue())
      return emitError(loc, "Only support dense values at this time");

    if (constOp->value().hasValue()) {
      auto newConstOp = rewriter.create<ONNXConstantOp>(
          loc, constOp->getResult().getType(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,nullptr, nullptr);
      rewriter.replaceOp(op, newConstOp.getResult());
    }
    return success();
  }
};

/*!
 *  Function pass that performs constant value elision.
 */
class ElideConstantValuePass
    : public PassWrapper<ElideConstantValuePass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<ConstantValueElision>(&getContext());

    applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};
} // end anonymous namespace

/*!
 * Create a Constant Value Elision pass.
 */
std::unique_ptr<mlir::Pass> mlir::createElideConstantValuePass() {
  return std::make_unique<ElideConstantValuePass>();
}
