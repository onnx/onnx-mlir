/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMAccelerator.hpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// OMAccelerator base class
//
//===----------------------------------------------------------------------===//
#include "src/Accelerators/Accelerator.hpp"
#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include <iostream>
#include <vector>

namespace mlir {
extern NNPAAccelerator nnpaAccelerator;
std::vector<Accelerator *> *Accelerator::acceleratorTargets;

Accelerator::Accelerator() {
  std::cout << "creating Accelerator" << std::endl;
  if (acceleratorTargets == NULL) {
    std::cout << "initializing acceleratorTargets" << std::endl;
    acceleratorTargets = new std::vector<Accelerator *>();
    acceleratorTargets->push_back(&nnpaAccelerator);
  }
}

std::vector<Accelerator *> *Accelerator::getAcceleratorList() {
  std::cout << "getting accelerator targets" << std::endl;
  return acceleratorTargets;
}

} // namespace mlir