/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- CompilerPasses.hpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "mlir/Pass/PassManager.h"

namespace onnx_mlir {
void addONNXToMLIRPasses(mlir::PassManager &pm, int transformThreshold,
    bool transformReport, bool targetCPU);
void addONNXToKrnlPasses(mlir::PassManager &pm, int optLevel, bool enableCSE,
    bool enableInstrumentONNXSignature, std::string ONNXOpsStatFilename);
void addKrnlToAffinePasses(mlir::PassManager &pm);
void addKrnlToLLVMPasses(
    mlir::OpPassManager &pm, bool enableCSE, bool verifyInputTensors);
InputIRLevelType determineInputIRLevel(
    mlir::OwningOpRef<mlir::ModuleOp> &module);
void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::PassManager &pm,
    EmissionTargetType emissionTarget);
} // namespace onnx_mlir
