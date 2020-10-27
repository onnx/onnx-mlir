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

#include "ElideKrnlGlobalConstants.hpp"

using namespace mlir;

const int64_t KrnlConstGlobalValueElision::kDefaultElisionThreshold = 32;

mlir::LogicalResult KrnlConstGlobalValueElision::matchAndRewrite(
    mlir::KrnlGlobalOp op, mlir::PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  if (op.value().hasValue()) {
    const auto &valAttr = op.valueAttr().dyn_cast_or_null<DenseElementsAttr>();
    if (valAttr.getNumElements() > elisionThreshold) {
      IntegerAttr offsetAttr = op.offset() ? op.offsetAttr() : nullptr;
      auto newGlobalOp = rewriter.create<KrnlGlobalOp>(loc,
          op.getResult().getType(), /*shape=*/op.shape(),
          /*name=*/op.name(), /*value=*/nullptr, /*offset=*/offsetAttr);
      rewriter.replaceOp(op, newGlobalOp.getResult());
    }
  }

  return success();
}

namespace {
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
    patterns.insert<KrnlConstGlobalValueElision>(
        &getContext(), KrnlConstGlobalValueElision::kDefaultElisionThreshold);

    applyPatternsAndFoldGreedily(function, patterns);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createElideConstGlobalValuePass() {
  return std::make_unique<ElideConstGlobalValuePass>();
}
