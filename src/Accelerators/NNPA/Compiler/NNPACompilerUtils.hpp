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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"

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
