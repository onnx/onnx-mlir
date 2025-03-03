/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- CompilerPasses.hpp -------------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// Functions for configuring and adding passes.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_COMPILER_PASSES_H
#define ONNX_MLIR_COMPILER_PASSES_H
#include "mlir/Pass/PassManager.h"

namespace onnx_mlir {
// Configures passes up front based on command line options.
void configurePasses();

void addONNXToMLIRPasses(mlir::PassManager &pm, bool targetCPU,
    bool donotScrubDisposableElementsAttr = false);
void addONNXToKrnlPasses(mlir::PassManager &pm, int optLevel, bool enableCSE,
    std::string ONNXOpsStatFilename);
void addKrnlToAffinePasses(mlir::PassManager &pm);
void addKrnlToLLVMPasses(
    mlir::OpPassManager &pm, std::string outputNameNoExt, bool enableCSE);
InputIRLevelType determineInputIRLevel(
    mlir::OwningOpRef<mlir::ModuleOp> &module);
void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::PassManager &pm,
    EmissionTargetType emissionTarget, std::string outputNameNoExt);
} // namespace onnx_mlir
#endif
