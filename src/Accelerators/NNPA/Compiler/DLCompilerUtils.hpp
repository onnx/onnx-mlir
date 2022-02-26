/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- DLCompilerUtils.hpp
//------------------------===//
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

enum DLCEmissionTargetType {
  EmitZNONE,
  EmitZLowIR,
  EmitZHighIR,
};

void addMemoryPooling(mlir::PassManager &pm);

void addONNXToZHighPasses(mlir::PassManager &pm);

void addZHighToZLowPasses(mlir::PassManager &pm);

void addAllToLLVMPasses(mlir::PassManager &pm);

void addPassesDLC(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    DLCEmissionTargetType dlcEmissionTarget,
    mlir::ArrayRef<std::string> execNodesOnCpu);

int compileModuleDLC(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputBaseName,
    onnx_mlir::EmissionTargetType emissionTarget,
    DLCEmissionTargetType dlcEmissionTarget = EmitZNONE,
    mlir::ArrayRef<std::string> execNodesOnCpu = mlir::ArrayRef<std::string>());
