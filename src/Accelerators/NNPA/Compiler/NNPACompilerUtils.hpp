/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- NNPACompilerUtils.hpp ----------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NNPA_COMPILER_UTILS_H
#define ONNX_MLIR_NNPA_COMPILER_UTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"

namespace onnx_mlir {

void addMemoryPooling(mlir::PassManager &pm);

void addONNXToZHighPasses(mlir::PassManager &pm);

void addZHighToZLowPasses(mlir::PassManager &pm);

void normalizeMemRefsPasses(mlir::PassManager &pm);

void addAllToLLVMPasses(mlir::PassManager &pm);

void addPassesNNPA(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt);

void configurePassesNNPA();

} // namespace onnx_mlir
#endif
