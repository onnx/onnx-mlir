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
#include <vector>
#include "src/Accelerators/OMAccelerator.hpp"
#include <iostream>

namespace mlir {
std::vector<OMAccelerator *> *OMAccelerator::acceleratorTargets;

OMAccelerator::OMAccelerator() {
    std::cout << "creating OMAccelerator" << std::endl;
    if (acceleratorTargets == NULL) {
        std::cout << "initializing acceleratorTargets" << std::endl;
        acceleratorTargets =  new std::vector<OMAccelerator *>();
    }
 }

std::vector<OMAccelerator *> *OMAccelerator::getAcceleratorList() {
    //if (OMAcceleratorTargets == NULL)
      //OMAcceleratorTargets = new vector<OMAccelerator *>();
    std::cout << "getting accelerator targets" << std::endl;  
    return acceleratorTargets;  
  }    


}