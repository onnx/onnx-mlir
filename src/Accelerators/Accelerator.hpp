/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Accelerator.hpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// OMAccelerator base class
//
//===----------------------------------------------------------------------===//
#pragma once
#include <vector>

namespace mlir {
class Accelerator {
public:
  Accelerator();
  static std::vector<Accelerator *> *getAcceleratorList();
  virtual void prepareAccelerator() = 0;

private:
  static std::vector<Accelerator *> *acceleratorTargets;
};
} // namespace mlir