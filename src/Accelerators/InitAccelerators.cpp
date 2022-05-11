/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ InitAccelerators.cpp ------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Initialization of accelerators' compile time data structures.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {
namespace accel {

APPLY_TO_ACCELERATORS(DECLARE_ACCEL_INIT_FUNCTION)

/// Initialize accelerators of the given kinds.
void initAccelerators(llvm::ArrayRef<Accelerator::Kind> kinds) {
  APPLY_TO_ACCELERATORS(INVOKE_ACCEL_INIT_FUNCTION, kinds)
}

} // namespace accel
} // namespace onnx_mlir
