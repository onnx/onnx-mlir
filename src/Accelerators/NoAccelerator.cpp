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
#include <iostream>
#include <vector>

namespace mlir {
std::vector<Accelerator *> *Accelerator::acceleratorTargets;

Accelerator::Accelerator() {
  std::cout << "creating null Accelerator" << std::endl;
}

std::vector<Accelerator *> *Accelerator::getAcceleratorList() {
  std::cout << "getting accelerator targets" << std::endl;
  return acceleratorTargets;
}

} // namespace mlir