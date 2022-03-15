/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Passes.hpp - ONNX-MLIR Passes Definition ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for ONNX-MLIR.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace mlir {
class Pass;

/// Pass for ONNX graph level optimization
std::unique_ptr<Pass> createONNXOpTransformPass();
std::unique_ptr<Pass> createONNXOpTransformPass(int threshold);

/// Pass for rewriting inside frontend dialect.
std::unique_ptr<Pass> createDecomposeONNXToONNXPass();

/// pass for ONNX to Leaky relu
std::unique_ptr<Pass> createONNXToAtenConstantOpTransformPass();
std::unique_ptr<Pass> createONNXToAtenLeakyReluOpTransformPass();
std::unique_ptr<Pass> createONNXToAtenMaxPool2dOpTransformPass();

/// Pass for ONNX to Aten conv2d operation
std::unique_ptr<Pass> createONNXToAtenConv2DOpTransformPass();
std::unique_ptr<Pass> createONNXToAtenConstantPadNdOpTransformPass();

std::unique_ptr<Pass> createShapeInferencePass(
    bool analyzeAllFunctions = false);

std::unique_ptr<Pass> createConstPropONNXToONNXPass();

/// Pass for eliding the values of constant operations.
std::unique_ptr<Pass> createElideConstantValuePass();

/// Pass for enabling a memory pool for MemRefs.
std::unique_ptr<Pass> createKrnlEnableMemoryPoolPass();

/// Pass for enabling a memory pool for MemRefs.
std::unique_ptr<Pass> createKrnlBundleMemoryPoolsPass();

/// Pass for optimizing memory pools.
std::unique_ptr<Pass> createKrnlOptimizeMemoryPoolsPass();

/// Pass for instrument the Onnx ops
std::unique_ptr<Pass> createInstrumentONNXPass();

/// Pass for verifying Onnx ops before lowering to Krnl
std::unique_ptr<Pass> createONNXPreKrnlVerifyPass();

/// Add pass for lowering to Krnl IR.
std::unique_ptr<Pass> createLowerToKrnlPass();
std::unique_ptr<Pass> createLowerToKrnlPass(int optLevel);
std::unique_ptr<Pass> createLowerToKrnlPass(
    bool emitDealloc, bool enableTiling);

/// Add pass for lowering to Torch IR.
std::unique_ptr<Pass> createLowerToTorchPass();
std::unique_ptr<Pass> createLowerToTorchPass(int optLevel);

/// Pass for lowering frontend dialects to Krnl IR dialect.
std::unique_ptr<Pass> createConvertKrnlToAffinePass();

/// Pass for lowering krnl.dim operations to standard dialect.
std::unique_ptr<Pass> createDisconnectKrnlDimFromAllocPass();

/// Pass for lowering krnl.shape operation.
std::unique_ptr<Pass> createLowerKrnlShapePass();

/// Pass for eliding the values of global Krnl operations.
std::unique_ptr<Pass> createElideConstGlobalValuePass();

/// Pass for lowering Krnl dialect to LLVM dialect.
std::unique_ptr<Pass> createConvertKrnlToLLVMPass();

} // end namespace mlir
