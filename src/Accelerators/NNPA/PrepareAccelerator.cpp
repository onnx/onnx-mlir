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
    std::cout << "initializing NNPA" << std::endl;
    if (!initialized) {
      initialized = true;
      this->getAcceleratorList().push_back(this);
    }
  };

  void prepareAccelerator() override {
    if (acceleratorTarget == "NNPA") {
      std::cout << "Targeting NNPA accelerator";
    }
  };
};

bool OMnnpaAccelerator::initialized = false;
OMnnpaAccelerator nnpaAccelerator();

std::vector<OMAccelerator *> OMAccelerator::acceleratorTargets;

std::vector<OMAccelerator *> OMAccelerator::getAcceleratorList() {
    //if (OMAcceleratorTargets == NULL)
      //OMAcceleratorTargets = new vector<OMAccelerator *>;
      
    return acceleratorTargets;  
  }
  } // namespace mlir