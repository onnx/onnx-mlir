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

APPLY_TO_ACCELERATORS(DECLARE_ACCEL_INIT_FUNCTION)

void initAccelerators() { APPLY_TO_ACCELERATORS(INVOKE_ACCEL_INIT_FUNCTION) }

} // namespace accel
} // namespace onnx_mlir
