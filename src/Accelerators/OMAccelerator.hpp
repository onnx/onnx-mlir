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

class OMAccelerator {
public:
  virtual void prepareAccelerator();
};

std::vector<OMAccelerator *> OMAcceleratorTargets;
