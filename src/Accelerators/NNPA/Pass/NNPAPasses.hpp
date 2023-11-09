/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- NNPAPasses.hpp - NNPA Passes Definition ------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for NNPA in
// addition to the passes used by ONNX MLIR.
//
//===----------------------------------------------------------------------===//

#pragma once

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
std::unique_ptr<mlir::Pass> createONNXToZHighPass();

/// Add pass for rewriting ONNX ops for ZHigh.
std::unique_ptr<mlir::Pass> createRewriteONNXForZHighPass();
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

/// Pass for clipping values to dlfloat before stickification at ZHighIR.
std::unique_ptr<mlir::Pass> createZHighClipToDLFloatPass();

/// Pass for decomposing stick/unstick at ZHighIR.
std::unique_ptr<mlir::Pass> createZHighDecomposeStickUnstickPass();

} // end namespace zhigh

namespace zlow {

/// Convert some zlow operations to affine.
std::unique_ptr<mlir::Pass> createConvertZLowToAffinePass();

// Add pass for fusing some zlow ops.
std::unique_ptr<mlir::Pass> createZLowFusionPass();

/// Add pass for sinking zlow.attach_layout and zlow.detach operations into
/// affine-for loops.
std::unique_ptr<mlir::Pass> createZLowLoopSinkingPass();

/// Add pass for rewriting ZLow ops.
std::unique_ptr<mlir::Pass> createZLowRewritePass();

/// Add pass for replacing zlow.unstick and zlow.stick by inserting dlf16
/// conversion directly into affine.for loops.
std::unique_ptr<mlir::Pass> createZLowInsertDLF16ConversionPass();

/// Add pass for rewriting ZLow ops.
std::unique_ptr<mlir::Pass> createZLowDummyOpForMultiDerefPass();

} // namespace zlow
} // namespace onnx_mlir
