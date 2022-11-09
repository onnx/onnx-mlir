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

// clang-format off
#define INSTRUMENTSTAGE_EUM_NNPA                                               \
  , nnpaAfterOnnxToOnnx,                                                       \
    nnpaAfterOnnxToZhigh,                                                      \
    nnpaAfterZhighToZlow                                                       \
// clang-format on

#define INSTRUMENTSTAGE_CL_ENUM_NNPA                                           \
  , clEnumVal(nnpaAfterOnnxToOnnx, "NNPA profiling for onnx ops."),            \
      clEnumVal(                                                               \
          nnpaAfterOnnxToZhigh, "NNPA profiling for onnx and zhigh ops."),     \
      clEnumVal(nnpaAfterZhighToZlow, "NNPA profiling for zlow ops.")

namespace onnx_mlir {

typedef enum {
  EmitZNONE,
  EmitZLowIR,
  EmitZHighIR,
} NNPAEmissionTargetType;

void addMemoryPooling(mlir::PassManager &pm);

void addONNXToZHighPasses(mlir::PassManager &pm);

void addZHighToZLowPasses(mlir::PassManager &pm);

void normalizeMemRefsPasses(mlir::PassManager &pm);

void addAllToLLVMPasses(mlir::PassManager &pm);

void addPassesNNPA(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget);

} // namespace onnx_mlir
