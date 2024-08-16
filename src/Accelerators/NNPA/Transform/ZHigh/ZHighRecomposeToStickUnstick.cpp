/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- ZHighRecomposeToStickUnstick.cpp - ZHigh High Level Optimizer ----===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZHigh dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

namespace {
/// Use anonymous namespace to avoid duplication symbol `populateWithGenerated`
/// among multiple tablegen-based definitions.

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Transform/ZHigh/ONNXZHighRecomposeToStickUnstick.inc"
} // namespace

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// ZHigh layout propagation Pass
//===----------------------------------------------------------------------===//

struct ZHighRecomposeToStickUnstickPass
    : public PassWrapper<ZHighRecomposeToStickUnstickPass,
          OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHighRecomposeToStickUnstickPass)

  StringRef getArgument() const override {
    return "zhigh-recompose-to-stick-unstick";
  }

  StringRef getDescription() const override {
    return "Compose LayerTransform and zhigh.F32ToDLF16 (zhigh.DLF16ToF32) "
           "back to ZHighStickOp (ZHighUnstickOp)";
  }

  void runOnOperation() override {
    Operation *function = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // Get patterns from tablegen.
    populateWithGenerated(patterns);

    // Get canonicalization rules for some important operations.
    ZHighDLF16ToF32Op::getCanonicalizationPatterns(patterns, &getContext());
    ZHighF32ToDLF16Op::getCanonicalizationPatterns(patterns, &getContext());
    ONNXLayoutTransformOp::getCanonicalizationPatterns(patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};

std::unique_ptr<Pass> createZHighRecomposeToStickUnstickPass() {
  return std::make_unique<ZHighRecomposeToStickUnstickPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
