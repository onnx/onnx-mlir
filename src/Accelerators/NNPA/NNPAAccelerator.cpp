/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- PrepareAccelerator.cpp
//-------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Add accelerator support for NNPA
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include "src/Support/OMOptions.hpp"
#include <iostream>

namespace mlir {

NNPAAccelerator::NNPAAccelerator() {
  std::cout << "initializing NNPA" << std::endl;
  if (!initialized) {
    initialized = true;
    // getAcceleratorList()->push_back(this);
  } // else
    // getAcceleratorList()->push_back(this);
};

void NNPAAccelerator::prepareAccelerator() {
  std::cout << "preparing accelerator " << acceleratorTarget << std::endl;
  if (acceleratorTarget.compare("NNPA") == 0) {
    std::cout << "Targeting NNPA accelerator" << std::endl;
  }
};

bool NNPAAccelerator::initialized = false;
NNPAAccelerator nnpaAccelerator;

} // namespace mlir
