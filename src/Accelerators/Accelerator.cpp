/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Accelerator.cpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Accelerator base class
//
//===----------------------------------------------------------------------===//
#include "src/Accelerators/Accelerator.hpp"
#include <iostream>
#include <vector>

namespace mlir {
std::vector<Accelerator *> *Accelerator::acceleratorTargets;

Accelerator::Accelerator() {
  if (acceleratorTargets == NULL) {
    acceleratorTargets = new std::vector<Accelerator *>();
  }
}

std::vector<Accelerator *> *Accelerator::getAcceleratorList() {
  return acceleratorTargets;
}

} // namespace mlir
