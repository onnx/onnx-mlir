/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Steer estimation to the right arch  ----------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// File where we generate code that converts calls to the ones of a specific
// architecture. For example,
// defines estimatedTimeForCPU_Log_3ds as a call to either
// arch14_estimatedTimeForCPU_Log_3ds or arch15_estimatedTimeForCPU_Log_3ds.
// =============================================================================

// Include machine specific model.
inline static double ms_ceiling(double n, double m) { return ceil(n / m) * m; }
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/PerfModelArch14.inc"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/PerfModelArch15.inc"

// 3 parameter estimate functions(e3, e2, e1)
#define ESTIMATE_TIME_FOR_DEV3(NAME, DEV)                                      \
  static double estimatedTimeFor##DEV##_##NAME(                                \
      double e3, double e2, double e1) {                                       \
    if (isLessEqualNNPALevel(NNPALevel::M14))                                  \
      return arch14_estimatedTimeFor##DEV##_##NAME(e3, e2, e1);                \
    if (isLessEqualNNPALevel(NNPALevel::M15))                                  \
      return arch15_estimatedTimeFor##DEV##_##NAME(e3, e2, e1);                \
    llvm_unreachable("add new NNPA architecture model here");                  \
  }

#define ESTIMATE_TIME_FOR3(NAME)                                               \
  ESTIMATE_TIME_FOR_DEV3(NAME, CPU)                                            \
  ESTIMATE_TIME_FOR_DEV3(NAME, NNPA)

#define MISSING_ESTIMATE3(NAME, ARCH)                                          \
  static double ARCH##_estimatedTimeForCPU_##NAME(                             \
      double e3, double e2, double e1) {                                       \
    return e3 * e2 * e1;                                                       \
  }                                                                            \
  static double ARCH##_estimatedTimeForNNPA_##NAME(                            \
      double e3, double e2, double e1) {                                       \
    return 100 * e3 * e2 * e1;                                                 \
  }

// 4 parameter estimate functions(e4, e3, e2, e1). Identical as above otherwise.
#define ESTIMATE_TIME_FOR_DEV4(NAME, DEV)                                      \
  static double estimatedTimeFor##DEV##_##NAME(                                \
      double e4, double e3, double e2, double e1) {                            \
    if (isLessEqualNNPALevel(NNPALevel::M14))                                  \
      return arch14_estimatedTimeFor##DEV##_##NAME(e4, e3, e2, e1);            \
    return arch15_estimatedTimeFor##DEV##_##NAME(e4, e3, e2, e1);              \
  }

#define ESTIMATE_TIME_FOR4(NAME)                                               \
  ESTIMATE_TIME_FOR_DEV4(NAME, CPU)                                            \
  ESTIMATE_TIME_FOR_DEV4(NAME, NNPA)

#define MISSING_ESTIMATE4(NAME, ARCH)                                          \
  static double ARCH##_estimatedTimeForCPU_##NAME(                             \
      double e4, double e3, double e2, double e1) {                            \
    return e4 * e3 * e2 * e1;                                                  \
  }                                                                            \
  static double ARCH##_estimatedTimeForNNPA_##NAME(                            \
      double e4, double e3, double e2, double e1) {                            \
    return 100 * e4 * e3 * e2 * e1;                                            \
  }

// Invoke macros. Stick and Unstick op are handled directly in PerfModel.cpp.
ESTIMATE_TIME_FOR3(Add_3ds)
ESTIMATE_TIME_FOR3(Div_3ds)
ESTIMATE_TIME_FOR3(Exp_3ds)
MISSING_ESTIMATE3(Gelu_3ds, arch14)
ESTIMATE_TIME_FOR3(Gelu_3ds)
MISSING_ESTIMATE3(LeakyRelu_3ds, arch14)
ESTIMATE_TIME_FOR3(LeakyRelu_3ds)
ESTIMATE_TIME_FOR3(Log_3ds)
ESTIMATE_TIME_FOR4(MatMul_3ds)
// Skipping the special handling of broadcast in matmul for now.
// MISSING_ESTIMATE4(MatMul_bcast23, march14)
// ESTIMATE_TIME_FOR4(MatMul_bcast23)
ESTIMATE_TIME_FOR3(Max_3ds)
ESTIMATE_TIME_FOR3(Min_3ds)
ESTIMATE_TIME_FOR3(Mul_3ds)
ESTIMATE_TIME_FOR3(Pow2_3ds)
ESTIMATE_TIME_FOR3(Pow3_3ds)
ESTIMATE_TIME_FOR3(Pow4_3ds)
ESTIMATE_TIME_FOR3(Pow8_3ds)
ESTIMATE_TIME_FOR3(ReduceMean_4d)
ESTIMATE_TIME_FOR3(Relu_3ds)
ESTIMATE_TIME_FOR3(Sigmoid_3ds)
ESTIMATE_TIME_FOR3(Softmax_3ds)
MISSING_ESTIMATE3(Sqrt_3ds, arch14)
ESTIMATE_TIME_FOR3(Sqrt_3ds)
ESTIMATE_TIME_FOR3(Sub_3ds)
ESTIMATE_TIME_FOR3(Tanh_3ds)
