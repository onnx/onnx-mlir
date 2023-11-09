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

#define INSTRUMENTSTAGE_ENUM_NNPA                                               \
    ,                                                                          \
    ZHigh,                                                                     \
    ZLow

// clang-format on

#define INSTRUMENTSTAGE_CL_ENUM_NNPA                                           \
  clEnumVal(Onnx, "Profile for onnx ops. For NNPA, profile onnx ops before "   \
                  "lowering to zhigh."),                                       \
      clEnumVal(ZHigh, "NNPA profiling for onnx and zhigh ops."),              \
      clEnumVal(ZLow, "NNPA profiling for zlow ops.")

#define PROFILEIR_CL_ENUM_NNPA                                                 \
  , clEnumVal(ZHigh, "Profile operations in ZHighIR generated by "             \
                     "--EmitZHighIR.")

namespace onnx_mlir {
typedef enum {
  EmitZNONE,
  EmitZLowIR,
  EmitZHighIR,
} NNPAEmissionTargetType;

typedef enum {
  QualifyingOps,    /* Any ops that qualify for NNPA will go on NNPA. */
  FasterOps,        /* Only qualifying ops that are faster on NNPA */
  FasterOpsWSU,     /* FasterOps with With Stick and Unstick (WSU) cost.*/
  MuchFasterOpsWSU, /* FasterOpsWSU only if significantly faster. */
} NNPAPlacementHeuristic;

extern llvm::cl::OptionCategory OnnxMlirOptions;
extern llvm::cl::opt<onnx_mlir::NNPAEmissionTargetType> nnpaEmissionTarget;
extern llvm::cl::opt<bool> nnpaClipToDLFloatRange;
extern llvm::cl::opt<bool> nnpaEnableZHighToOnnx;
extern llvm::cl::opt<NNPAPlacementHeuristic> nnpaPlacementHeuristic;
extern llvm::cl::opt<bool> profileZHighIR;
extern llvm::cl::opt<std::string> nnpaLoadDevicePlacementFile;
extern llvm::cl::opt<std::string> nnpaSaveDevicePlacementFile;
extern llvm::cl::opt<std::string> nnpaParallelOpt;
extern llvm::cl::opt<bool> nnpaParallel;
extern llvm::cl::opt<int> nnpaParallelNdev;
extern llvm::cl::opt<int> nnpaParallelMinimumDimThreshold;

} // namespace onnx_mlir
