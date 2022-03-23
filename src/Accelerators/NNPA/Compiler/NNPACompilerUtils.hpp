/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- NNPACompilerUtils.hpp ----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/PassManager.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Support/OMOptions.hpp"

namespace onnx_mlir {

typedef enum {
  EmitZNONE,
  EmitZLowIR,
  EmitZHighIR,
} NNPAEmissionTargetType;

void addMemoryPooling(mlir::PassManager &pm);

void addONNXToZHighPasses(mlir::PassManager &pm);

void addZHighToZLowPasses(mlir::PassManager &pm);

void addAllToLLVMPasses(mlir::PassManager &pm);

void addPassesNNPA(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    NNPAEmissionTargetType nnpaEmissionTarget,
    mlir::ArrayRef<std::string> execNodesOnCpu);

int compileModuleNNPA(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputBaseName,
    onnx_mlir::EmissionTargetType emissionTarget,
    NNPAEmissionTargetType dlcEmissionTarget =
        NNPAEmissionTargetType::EmitZNONE,
    mlir::ArrayRef<std::string> execNodesOnCpu = mlir::ArrayRef<std::string>());

} // namespace onnx_mlir
