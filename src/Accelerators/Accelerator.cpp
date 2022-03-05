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
//#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include <iostream>
#include <vector>

namespace onnx_mlir {
namespace accel {

extern NNPAAccelerator nnpaAccelerator;
std::vector<Accelerator *> *Accelerator::acceleratorTargets;

Accelerator::Accelerator() {
  if (acceleratorTargets == NULL)
    acceleratorTargets = new std::vector<Accelerator *>();
}

Accelerator::~Accelerator() {}

std::vector<Accelerator *> *Accelerator::getAcceleratorList() {
  return acceleratorTargets;
}

} // namespace accel
} // namespace onnx_mlir
