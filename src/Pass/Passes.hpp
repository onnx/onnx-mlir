/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Passes.hpp - ONNX-MLIR Passes Definition ------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for ONNX-MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PASSES_H
#define ONNX_MLIR_PASSES_H

#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"
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

std::unique_ptr<mlir::Pass> createReplaceOpWithItsOperandPass(
    std::vector<std::string> nodeNameRegexList);

std::unique_ptr<mlir::Pass> createShapeInferencePass();

// To configure ConstPropONNXToONNXPass at program start.
void configureConstPropONNXToONNXPass(bool roundFPToInt, int expansionBound,
    llvm::ArrayRef<std::string> disabledPatterns, bool constantPropIsDisabled);

std::unique_ptr<mlir::Pass> createConstPropONNXToONNXPass();

/// Pass for instrument the ops in specific stage.
#define GEN_PASS_DECL_INSTRUMENTPASS
#include "src/Transform/Passes.h.inc"
// GEN_PASS_DEF method only adds default constructor only,
// we add custom constructor with multi positional argument explicitly
std::unique_ptr<mlir::Pass> createInstrumentPass(
    const std::string &ops, unsigned actions);
/// Pass for instrument cleanup.
#define GEN_PASS_DECL_INSTRUMENTCLEANUPPASS
#include "src/Transform/Passes.h.inc"

/// Passes for instrumenting the ONNX ops to print their operand type
/// signatures at runtime.
std::unique_ptr<mlir::Pass> createInstrumentONNXSignaturePass(
    const std::string opPattern, const std::string nodePattern);

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
    bool enableSIMD, bool enableParallel, bool enableFastMath,
    std::string opsForCall);
void configureOnnxToKrnlLoweringPass(bool reportOnParallel,
    bool parallelIsEnabled, std::string specificParallelOps, bool reportOnSimd,
    bool simdIsEnabled);
#define GEN_PASS_DECL_PROCESSSCFPARALLELPRIVATEPASS
#include "src/Transform/Passes.h.inc"
#define GEN_PASS_DECL_PROCESSKRNLPARALLELCLAUSEPASS
#include "src/Transform/Passes.h.inc"

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
/// Add pass for lowering to Stablehlo IR.
std::unique_ptr<mlir::Pass> createLowerToStablehloPass();
std::unique_ptr<mlir::Pass> createLowerToStablehloPass(bool enableUnroll);
#endif

/// Pass for eliding the values of global Krnl operations.
std::unique_ptr<mlir::Pass> createElideConstGlobalValuePass();

namespace krnl {
/// Pass for lowering frontend dialects to Krnl IR dialect.
std::unique_ptr<mlir::Pass> createConvertKrnlToAffinePass();
std::unique_ptr<mlir::Pass> createConvertKrnlToAffinePass(bool parallelEnabled);

/// Pass for lowering Seq in Krnl dialect.
std::unique_ptr<mlir::Pass> createConvertSeqToMemrefPass();

/// Pass for lowering krnl.region operation.
//To use explicit namespace onnx_mlir::krnl::,
//we add constructor explicitly in the krnl namespace instead of GEN_PASS_DECL method.
/*
#define GEN_PASS_DECL_LOWERKRNLREGIONPASS
#include "src/Transform/Passes.h.inc"
*/
std::unique_ptr<mlir::Pass> createLowerKrnlRegionPass();

/// Pass for lowering Krnl dialect to LLVM dialect.
std::unique_ptr<mlir::Pass> createConvertKrnlToLLVMPass();
std::unique_ptr<mlir::Pass> createConvertKrnlToLLVMPass(bool verifyInputTensors,
    bool useLRODATA, bool storeConstantsToFile,
    float constantsToFileSingleThreshold, float constantsToFileTotalThreshold,
    std::string outputNameNoExt, bool enableParallel);

} // namespace krnl

/// Pass for lowering Onnx ops to TOSA dialect
std::unique_ptr<mlir::Pass> createConvertONNXToTOSAPass();

/// Pass for lowering Onnx ops to Linalg dialect
std::unique_ptr<mlir::Pass> createConvertONNXToLinalg();

// Get the function that creates the pass for pass manager.
// This function has the name of createPassName(), created from Passes.td.
// For buffer omploop hoisting pass, it is crateBufferOMPLoopHoisting()
// This function is the only globally visible function for the pass, and
// is defined at the end of the pass implementation file.
#define GEN_PASS_DECL_BUFFEROMPLOOPHOISTINGPASS
#include "src/Transform/Passes.h.inc"

// The function registerTransformsPasses() is generated from Passes.td and used
// to register th pass for onnx-mlir-opt. Different Passes.td will generate the
// same name function. They have to be put into different name space to be
// distinguished.
#define GEN_PASS_REGISTRATION
#include "src/Transform/Passes.h.inc"

} // namespace onnx_mlir

/*
#define GEN_PASS_REGISTRATION
#include "src/Transform/Passes.h.inc"
*/

#endif
