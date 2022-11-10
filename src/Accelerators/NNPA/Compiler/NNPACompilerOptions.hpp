/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ NNPACompilerOptions.hpp ---------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/CommandLine.h"

#if 0
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#endif

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

  extern llvm::cl::OptionCategory OnnxMlirOptions;
  extern llvm::cl::opt<onnx_mlir::NNPAEmissionTargetType> nnpaEmissionTarget;
  extern llvm::cl::list<std::string> execNodesOnCpu;


} // namespace onnx_mlir
