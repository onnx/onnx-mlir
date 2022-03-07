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

namespace mlir {
class Pass;

/// In ONNX, add pass for lowering Tensor types to MemRef types.
std::unique_ptr<Pass> createONNXTensorToMemRefPass();

/// Add pass for lowering ONNX ops to ZHigh ops.
std::unique_ptr<Pass> createONNXToZHighPass();
std::unique_ptr<Pass> createONNXToZHighPass(
    mlir::ArrayRef<std::string> execNodesOnCpu);

/// Add pass for rewriting ONNX ops for ZHigh.
std::unique_ptr<Pass> createRewriteONNXForZHighPass();
std::unique_ptr<Pass> createRewriteONNXForZHighPass(
    mlir::ArrayRef<std::string> execNodesOnCpu);

/// Add pass for lowering ZHigh ops to ZLow ops.
std::unique_ptr<Pass> createZHighToZLowPass();

/// Add pass for lowering ZHigh ops to ZLow ops.
std::unique_ptr<Pass> createZHighToZLowPass(int optLevel);

/// Add pass for lowering ZHigh ops to ZLow ops.
std::unique_ptr<Pass> createZHighToZLowPass(
    bool emitDealloc, bool enableTiling);

/// Add pass for lowering ZLow ops to LLVM.
std::unique_ptr<Pass> createZLowToLLVMPass();

/// Pass for rewriting KRNL dialect operations.
std::unique_ptr<Pass> createKrnlEmptyOptimizationPass();

/// Pass for folding std.alloc.
std::unique_ptr<Pass> createFoldStdAllocPass();

/// Pass for layout propagation at ZHighIR.
std::unique_ptr<Pass> createZHighLayoutPropagationPass();

/// Pass for constant propagation at ZHighIR.
std::unique_ptr<Pass> createZHighConstPropagationPass();

/// Pass for instrument the ZHigh ops
std::unique_ptr<Pass> createInstrumentZHighPass();

} // end namespace mlir
