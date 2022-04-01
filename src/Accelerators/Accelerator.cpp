/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Accelerator.cpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Accelerator base class.
//
// To enable a new accelerator, add the header include, an extern of the
// subclass and pushback that subclass variable onto acceleratorTargets.
//===----------------------------------------------------------------------===//

#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {
namespace accel {

llvm::SmallVector<Accelerator *, 4> Accelerator::acceleratorTargets;

const llvm::SmallVectorImpl<Accelerator *> &Accelerator::getAccelerators() {
  return acceleratorTargets;
}

} // namespace accel
} // namespace onnx_mlir
