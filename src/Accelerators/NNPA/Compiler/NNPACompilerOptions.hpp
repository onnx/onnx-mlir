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

// clang-format off
#define INSTRUMENTSTAGE_EUM_NNPA                                               \
    ,                                                                          \
    ZHigh,                                                                     \
    ZLow                                                                       \
// clang-format on

#define INSTRUMENTSTAGE_CL_ENUM_NNPA                                           \
  clEnumVal(Onnx, "Profile for onnx ops. For NNPA, profile onnx ops before "   \
                  "lowering to zhigh."),                                       \
      clEnumVal(ZHigh, "NNPA profiling for onnx and zhigh ops."),              \
      clEnumVal(ZLow, "NNPA profiling for zlow ops.")

namespace onnx_mlir {
typedef enum {
  EmitZNONE,
  EmitZLowIR,
  EmitZHighIR,
} NNPAEmissionTargetType;

extern llvm::cl::OptionCategory OnnxMlirOptions;
extern llvm::cl::opt<onnx_mlir::NNPAEmissionTargetType> nnpaEmissionTarget;
extern llvm::cl::list<std::string> execNodesOnCpu;
extern llvm::cl::opt<bool> nnpaClipToDLFloatRange;

} // namespace onnx_mlir
