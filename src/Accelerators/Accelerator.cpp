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
//===----------------------------------------------------------------------===//

#include "src/Accelerators/Accelerator.hpp"
#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"

namespace onnx_mlir {
namespace accel {

llvm::SmallPtrSet<Accelerator *, 2> Accelerator::accelerators;

void Accelerator::create(Accelerator::Kind kind,
    mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::MLIRContext &context,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType emissionTarget) {
  Accelerator *accel = nullptr;
  switch (kind) {
  case Kind::NNPA:
    #ifdef __NNPA__
    accel = new nnpa::NNPAAccelerator();
    #endif
    break;
  }
  assert(accel && "should never be null");

  // Initialize the new accelerator and add it to the list of available ones.
  accel->prepare(module, context, pm, emissionTarget);
  accelerators.insert(accel);
}

Accelerator::~Accelerator() { accelerators.erase(this); }

const llvm::SmallPtrSetImpl<Accelerator *> &Accelerator::getAccelerators() {
  return accelerators;
}

} // namespace accel
} // namespace onnx_mlir
