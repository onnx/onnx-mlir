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
void addONNXToMLIRPasses(mlir::PassManager &pm);
void addONNXToKrnlPasses(mlir::PassManager &pm, int optLevel);
void addKrnlToAffinePasses(mlir::PassManager &pm);
void addKrnlToLLVMPasses(mlir::OpPassManager &pm);
} // namespace onnx_mlir
