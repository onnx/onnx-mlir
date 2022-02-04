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

namespace mlir {
class OMAccelerator {
public:
  OMAccelerator();
  static std::vector<OMAccelerator *> *getAcceleratorList();
  virtual void prepareAccelerator() = 0;
private:  
  static std::vector<OMAccelerator *> *acceleratorTargets;
};
}