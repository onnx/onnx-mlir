/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Passes.hpp - ONNX-MLIR Passes Definition ------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for ONNX-MLIR.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class MLIRContext;
class Pass;
} // namespace mlir

namespace onnx_mlir {

/// Pass for removing DisposableElementsAttr attributes.
std::unique_ptr<mlir::Pass> createScrubDisposablePass(bool closeAfter = true);

/// Pass for ONNX graph level optimization
std::unique_ptr<mlir::Pass> createONNXOpTransformPass();
std::unique_ptr<mlir::Pass> createONNXOpTransformPass(int threshold,
    bool report, bool targetCPU, bool enableSimdDataLayoutOpt,
    bool enableConvOptPass, bool enableRecomposeOptPass);

/// Pass for rewriting inside frontend dialect.
std::unique_ptr<mlir::Pass> createDecomposeONNXToONNXPass(
    const std::string &target = "");
std::unique_ptr<mlir::Pass> createRecomposeONNXToONNXPass(
    const std::string &target = "");

std::unique_ptr<mlir::Pass> createConvOptONNXToONNXPass(
    bool enableSimdDataLayoutOpt = false);

std::unique_ptr<mlir::Pass> createShapeInferencePass();

// To configure ConstPropONNXToONNXPass at program start.
void configureConstPropONNXToONNXPass(bool roundFPToInt, int expansionBound,
    llvm::ArrayRef<std::string> disabledPatterns, bool constantPropIsDisabled);

std::unique_ptr<mlir::Pass> createConstPropONNXToONNXPass();

/// Pass for instrument the ops in specific stage.
std::unique_ptr<mlir::Pass> createInstrumentPass();
std::unique_ptr<mlir::Pass> createInstrumentPass(
    const std::string &ops, unsigned actions);

/// Passes for instrumenting the ONNX ops to print their operand type
/// signatures at runtime.
std::unique_ptr<mlir::Pass> createInstrumentONNXSignaturePass();

/// Pass for simplifying shape-related ONNX operations.
std::unique_ptr<mlir::Pass> createSimplifyShapeRelatedOpsPass();

/// Pass for replacing ONNXReturnOp with func::ReturnOp.
std::unique_ptr<mlir::Pass> createStandardFuncReturnPass();

/// Pass that combines multiple ONNX dialect transformations,
/// including shape inference.
std::unique_ptr<mlir::Pass> createONNXHybridTransformPass(
    bool enableRecomposition);

/// Pass for analyzing unknown dimension in ONNX operations.
std::unique_ptr<mlir::Pass> createONNXDimAnalysisPass();

/// Pass for setting onnx_node_name attribute if absent.
std::unique_ptr<mlir::Pass> createSetONNXNodeNamePass();

/// Pass for verifying Onnx ops before lowering to Krnl
std::unique_ptr<mlir::Pass> createONNXPreKrnlVerifyPass();

/// Add pass for lowering to Krnl IR.
std::unique_ptr<mlir::Pass> createLowerToKrnlPass();
std::unique_ptr<mlir::Pass> createLowerToKrnlPass(bool enableTiling,
    bool enableSIMD, bool enableParallel, std::string opsForCall);
void configureOnnxToKrnlLoweringPass(bool reportOnParallel,
    bool parallelIsEnabled, bool reportOnSimd, bool simdIsEnabled);

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
/// Add pass for lowering to StableHlo IR.
std::unique_ptr<mlir::Pass> createLowerToStableHloPass();
#endif

/// Pass for lowering krnl.dim operations to standard dialect.
std::unique_ptr<mlir::Pass> createDisconnectKrnlDimFromAllocPass();

/// Pass for lowering krnl.shape operation.
std::unique_ptr<mlir::Pass> createLowerKrnlShapePass();

/// Pass for eliding the values of global Krnl operations.
std::unique_ptr<mlir::Pass> createElideConstGlobalValuePass();

namespace krnl {
/// Pass for lowering frontend dialects to Krnl IR dialect.
std::unique_ptr<mlir::Pass> createConvertKrnlToAffinePass();

/// Pass for lowering Seq in Krnl dialect.
std::unique_ptr<mlir::Pass> createConvertSeqToMemrefPass();

/// Pass for lowering krnl.region operation.
std::unique_ptr<mlir::Pass> createLowerKrnlRegionPass();

/// Pass for lowering Krnl dialect to LLVM dialect.
std::unique_ptr<mlir::Pass> createConvertKrnlToLLVMPass();
std::unique_ptr<mlir::Pass> createConvertKrnlToLLVMPass(bool verifyInputTensors,
    bool useOpaquePointer, bool useLRODATA, bool storeConstantsToFile,
    float constantsToFileSingleThreshold, float constantsToFileTotalThreshold,
    std::string outputNameNoExt, bool enableParallel);

} // namespace krnl

/// Pass for lowering Onnx ops to TOSA dialect
std::unique_ptr<mlir::Pass> createConvertONNXToTOSAPass();

} // namespace onnx_mlir
