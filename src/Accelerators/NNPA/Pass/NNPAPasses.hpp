/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- NNPAPasses.hpp - NNPA Passes Definition ------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for NNPA in
// addition to the passes used by ONNX MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NNPA_PASSES_H
#define ONNX_MLIR_NNPA_PASSES_H

#include "mlir/Pass/Pass.h"

#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"

namespace onnx_mlir {

// Add pass for device placement.
std::unique_ptr<mlir::Pass> createDevicePlacementPass();
std::unique_ptr<mlir::Pass> createDevicePlacementPass(
    std::string loadConfigFile, std::string saveConfigFile,
    NNPAPlacementHeuristic placementHeuristic);

/// Add pass for lowering ONNX ops to ZHigh ops.
std::unique_ptr<mlir::Pass> createONNXToZHighPass();
void configureONNXToZHighLoweringPass(bool reportOnNNPAUnsupportedOps,
    bool isDynQuant, bool quantIsActivationSym, bool quantIsWeightSym,
    llvm::ArrayRef<std::string> quantOpTypes);

/// Add pass for rewriting ONNX ops for ZHigh.
std::unique_ptr<mlir::Pass> createRewriteONNXForZHighPass();

/// Add pass for re-construct ONNX ops from ZHigh ops.
std::unique_ptr<mlir::Pass> createZHighToONNXPass();

/// Pass for folding std.alloc.
std::unique_ptr<mlir::Pass> createFoldStdAllocPass();

namespace zhigh {

/// Pass for layout propagation at ZHighIR.
std::unique_ptr<mlir::Pass> createZHighLayoutPropagationPass();

/// Pass for constant propagation at ZHighIR.
std::unique_ptr<mlir::Pass> createZHighConstPropagationPass();

/// Pass for scrubbing constants at ZHighIR.
std::unique_ptr<mlir::Pass> createZHighScrubDisposablePass(
    bool closeAfter = true);

/// Pass for decomposing stick/unstick at ZHighIR.
std::unique_ptr<mlir::Pass> createZHighDecomposeStickUnstickPass();

/// Pass for recomposing ops back to stick/unstick at ZHighIR.
std::unique_ptr<mlir::Pass> createZHighRecomposeToStickUnstickPass();
} // end namespace zhigh

namespace zlow {

/// Add pass for rewriting ZLow ops.
std::unique_ptr<mlir::Pass> createZLowRewritePass();

/// Add pass for rewriting ZLow ops.
std::unique_ptr<mlir::Pass> createZLowStickExpansionPass(
    bool enableParallel = false);

/// Add pass for rewriting ZLow ops.
std::unique_ptr<mlir::Pass> createZLowDummyOpForMultiDerefPass();

} // namespace zlow
} // namespace onnx_mlir
#endif
