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

#include "src/Accelerators/OMAccelerator.hpp"
#include "src/Support/OMOptions.hpp"
#include <iostream>

namespace mlir {
class OMnnpaAccelerator : public OMAccelerator {
private:
  static bool initialized;

public:
  OMnnpaAccelerator() {
    if (!initialized) {
      initialized = true;
      OMAcceleratorTargets.push_back(this);
    }
  };

  void prepareAccelerator() {
    if (acceleratorTarget == "NNPA") {
      std::cout << "Targeting NNPA accelerator";
    }
  };
};

bool OMnnpaAccelerator::initialized = false;
static OMAccelerator nnpaAccelerator;

} // namespace mlir