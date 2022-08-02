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
}

namespace onnx_mlir {

/// Pass for ONNX graph level optimization
std::unique_ptr<mlir::Pass> createONNXOpTransformPass();
std::unique_ptr<mlir::Pass> createONNXOpTransformPass(
    int threshold, bool report);

/// Pass for rewriting inside frontend dialect.
std::unique_ptr<mlir::Pass> createDecomposeONNXToONNXPass();

std::unique_ptr<mlir::Pass> createShapeInferencePass(
    bool analyzeAllFunctions = false);

std::unique_ptr<mlir::Pass> createConstPropONNXToONNXPass();

/// Pass for eliding the values of constant operations.
std::unique_ptr<mlir::Pass> createElideConstantValuePass();

/// Pass for instrument the Onnx ops
std::unique_ptr<mlir::Pass> createInstrumentONNXPass();
std::unique_ptr<mlir::Pass> createInstrumentONNXPass(
    llvm::StringRef ops, int actions);

/// Passes for instrumenting the ONNX ops to print their operand type
/// signatures at runtime.
std::unique_ptr<mlir::Pass> createInstrumentONNXSignaturePass();

/// Pass for verifying Onnx ops before lowering to Krnl
std::unique_ptr<mlir::Pass> createONNXPreKrnlVerifyPass();

/// Add pass for lowering to Krnl IR.
std::unique_ptr<mlir::Pass> createLowerToKrnlPass();
std::unique_ptr<mlir::Pass> createLowerToKrnlPass(int optLevel);
std::unique_ptr<mlir::Pass> createLowerToKrnlPass(
    bool emitDealloc, bool enableTiling);

/// Add pass for lowering to Mhlo IR.
std::unique_ptr<mlir::Pass> createLowerToMhloPass();

/// Pass for lowering krnl.dim operations to standard dialect.
std::unique_ptr<mlir::Pass> createDisconnectKrnlDimFromAllocPass();

/// Pass for lowering krnl.shape operation.
std::unique_ptr<mlir::Pass> createLowerKrnlShapePass();

/// Pass for eliding the values of global Krnl operations.
std::unique_ptr<mlir::Pass> createElideConstGlobalValuePass();

namespace krnl {

/// Pass for lowering frontend dialects to Krnl IR dialect.
std::unique_ptr<mlir::Pass> createConvertKrnlToAffinePass();

/// Pass for enabling a memory pool for MemRefs.
std::unique_ptr<mlir::Pass> createKrnlEnableMemoryPoolPass();

/// Pass for enabling a memory pool for MemRefs.
std::unique_ptr<mlir::Pass> createKrnlBundleMemoryPoolsPass();

/// Pass for optimizing memory pools.
std::unique_ptr<mlir::Pass> createKrnlOptimizeMemoryPoolsPass();

/// Pass for lowering Seq in Krnl dialect.
std::unique_ptr<mlir::Pass> createConvertSeqToMemrefPass();

/// Pass for lowering krnl.region operation.
std::unique_ptr<mlir::Pass> createLowerKrnlRegionPass();

/// Pass for lowering Krnl dialect to LLVM dialect.
std::unique_ptr<mlir::Pass> createConvertKrnlToLLVMPass();
std::unique_ptr<mlir::Pass> createConvertKrnlToLLVMPass(
    bool verifyInputTensors);

} // namespace krnl

/// Pass for lowering Onnx ops to TOSA dialect
std::unique_ptr<mlir::Pass> createConvertONNXToTOSAPass();

} // namespace onnx_mlir
