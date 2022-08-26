/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===- ElideKrnlGlobalConstants.cpp - Krnl Constant lobal Value Elision ---===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

#include "ElideKrnlGlobalConstants.hpp"

using namespace mlir;
using namespace onnx_mlir;

constexpr uint64_t KrnlConstGlobalValueElision::kDefaultElisionThreshold;

mlir::LogicalResult KrnlConstGlobalValueElision::matchAndRewrite(
    mlir::KrnlGlobalOp op, mlir::PatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // Only elide if value is available.
  if (!op.value().has_value())
    return success();

  // Only elide dense and dense resource attributes.
  if (!(op.value()->isa<DenseElementsAttr>() ||
          op.value()->isa<DenseResourceElementsAttr>()))
    return success();

  MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);

  bool elide = false;

  if (op.value()->isa<DenseElementsAttr>()) {
    const auto &valAttr = op.valueAttr().dyn_cast_or_null<DenseElementsAttr>();
    if (valAttr.getNumElements() > elisionThreshold && !valAttr.isSplat()) {
      elide = true;
    }
  } else {
    const auto &valAttr =
        op.valueAttr().dyn_cast_or_null<DenseResourceElementsAttr>();
    if (valAttr.getNumElements() > elisionThreshold) {
      elide = true;
    }
  }

  if (elide) {
    IntegerAttr offsetAttr = op.offset() ? op.offsetAttr() : nullptr;
    IntegerAttr alignmentAttr = op.alignment() ? op.alignmentAttr() : nullptr;
    auto newGlobalOp =
        create.krnl.constant(op.getResult().getType().cast<MemRefType>(),
            op.name(), None, offsetAttr, alignmentAttr);
    rewriter.replaceOp(op, newGlobalOp);
  }

  return success();
}

namespace {
/*!
 *  Function pass that performs constant value elision of Krnl globals.
 */
class ElideConstGlobalValuePass : public PassWrapper<ElideConstGlobalValuePass,
                                      OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElideConstGlobalValuePass)

  StringRef getArgument() const override { return "elide-krnl-constants"; }

  StringRef getDescription() const override {
    return "Elide the constant values of the Global Krnl operations.";
  }

  void runOnOperation() override {
    auto function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<KrnlConstGlobalValueElision>(
        &getContext(), KrnlConstGlobalValueElision::kDefaultElisionThreshold);
    // No need to test, its ok to fail the apply.
    LogicalResult res =
        applyPatternsAndFoldGreedily(function, std::move(patterns));
    assert((succeeded(res) || failed(res)) && "remove unused var warning");
  }
};

} // namespace

std::unique_ptr<Pass> onnx_mlir::createElideConstGlobalValuePass() {
  return std::make_unique<ElideConstGlobalValuePass>();
}
