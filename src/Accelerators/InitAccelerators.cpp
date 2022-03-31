/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ InitAccelerators.cpp ------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Initialization of accelerators.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {
namespace accel {

#ifdef HAS_ACCELERATORS
APPLY_TO_ACCELERATORS(DECLARE_ACCEL_INIT_FUNCTION)
#endif

bool initAccelerators() {
#ifdef HAS_ACCELERATORS
  APPLY_TO_ACCELERATORS(INVOKE_ACCEL_INIT_FUNCTION)
  return true;
#else
  return false;
#endif
}

} // namespace accel
} // namespace onnx_mlir
