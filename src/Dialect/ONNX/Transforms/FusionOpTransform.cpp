/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- FusionOpTransform.cpp - Fuse ONNX op chains -----------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================

#include "src/Dialect/ONNX/Transforms/FusionOpTransform.hpp"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/Transforms/FusionOpBasePattern.hpp"
#include "src/Dialect/ONNX/Transforms/ONNXFusionOpHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

void populateONNXFusionOpPatterns(RewritePatternSet &patterns,
    MLIRContext *context, DimAnalysis *dimAnalysis) {
  patterns.insert<FusedPatternForOpKind<ONNXConcatOp, SplitOpGatherFusionHelper>>(
      context, dimAnalysis);
}

#define GEN_PASS_DEF_FUSIONOPTRANSFORMPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"

} // namespace onnx_mlir

namespace {

class FusionOpTransformPass
    : public onnx_mlir::impl::FusionOpTransformPassBase<
          FusionOpTransformPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionOpTransformPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (disableFusedOp)
      return;

    DimAnalysis *dimAnalysis = new DimAnalysis(module);
    dimAnalysis->analyze();

    RewritePatternSet patterns(&getContext());
    populateONNXFusionOpPatterns(patterns, &getContext(), dimAnalysis);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
