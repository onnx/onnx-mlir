/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- DLCPasses.hpp - DLC++ Passes Definition ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for DLC++ in
// addition to the passes used by ONNX MLIR.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace onnx_mlir {

/// Add pass for lowering ONNX ops to ZHigh ops.
std::unique_ptr<mlir::Pass> createONNXToZHighPass();
std::unique_ptr<mlir::Pass> createONNXToZHighPass(
    mlir::ArrayRef<std::string> execNodesOnCpu);

/// Add pass for rewriting ONNX ops for ZHigh.
std::unique_ptr<mlir::Pass> createRewriteONNXForZHighPass();
std::unique_ptr<mlir::Pass> createRewriteONNXForZHighPass(
    mlir::ArrayRef<std::string> execNodesOnCpu);
namespace zhigh {

/// Add pass for lowering ZHigh ops to ZLow ops.
std::unique_ptr<mlir::Pass> createZHighToZLowPass();

/// Add pass for lowering ZHigh ops to ZLow ops.
std::unique_ptr<mlir::Pass> createZHighToZLowPass(int optLevel);

/// Add pass for lowering ZHigh ops to ZLow ops.
std::unique_ptr<mlir::Pass> createZHighToZLowPass(
    bool emitDealloc, bool enableTiling);

/// Pass for layout propagation at ZHighIR.
std::unique_ptr<mlir::Pass> createZHighLayoutPropagationPass();

/// Pass for constant propagation at ZHighIR.
std::unique_ptr<mlir::Pass> createZHighConstPropagationPass();

} // end namespace zhigh
} // end namespace onnx_mlir

namespace mlir {

/// Add pass for rewriting ZLow ops.
std::unique_ptr<mlir::Pass> createZLowRewritePass();

/// Add pass for lowering ZLow ops to LLVM.
std::unique_ptr<mlir::Pass> createZLowToLLVMPass();

/// Pass for rewriting KRNL dialect operations.
std::unique_ptr<mlir::Pass> createKrnlEmptyOptimizationPass();

/// Pass for folding std.alloc.
std::unique_ptr<mlir::Pass> createFoldStdAllocPass();

/// Pass for instrument the ZHigh ops
std::unique_ptr<mlir::Pass> createInstrumentZHighPass();

} // end namespace mlir
