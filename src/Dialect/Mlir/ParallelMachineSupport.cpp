/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ParallelMachineSupport.cpp - Parallel cost model constants --------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================

#include "src/Dialect/Mlir/ParallelMachineSupport.hpp"
#include "src/Compiler/CompilerOptions.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dialect_builder"

namespace onnx_mlir {

// =============================================================================
// Handling of global parallel machine support pointer

/*static*/ ParallelMachineSupport
    *ParallelMachineSupport::globalParallelMachineSupport = nullptr;

/*static*/ void ParallelMachineSupport::setGlobalParallelMachineSupport(
    const std::string &triple, const std::string &arch,
    const std::string &cpu) {
  // Every target gets the same numbers for now; see the header for why. The
  // triple/arch/cpu are taken so that adding a per-target subclass later is a
  // change here and nowhere else.
  globalParallelMachineSupport = new GenericParallelMachineSupport();
  assert(globalParallelMachineSupport &&
         "failed to allocate parallel machine support");

  // Start from the target's own values, then let an explicitly given
  // command-line option win. A negative value means "not given": these
  // overrides are how a number stops being justified-by-argument and starts
  // being measured, per target, which is the only way the measurement means
  // anything.
  ParallelTuning tuning = globalParallelMachineSupport->computeTuning();
  if (collapseMinParTripCountFloor >= 0)
    tuning.minParTripCountFloor = collapseMinParTripCountFloor;
  if (collapseMinAmortWork >= 0)
    tuning.minAmortWork = collapseMinAmortWork;
  if (collapseMaxForkCount >= 0)
    tuning.maxForkCount = collapseMaxForkCount;
  globalParallelMachineSupport->tuning = tuning;

  LLVM_DEBUG(llvm::dbgs() << "use parallel tuning " << getArchName()
                          << " for triple \"" << triple << "\" arch \"" << arch
                          << "\" cpu \"" << cpu << "\": minParTripCountFloor "
                          << tuning.minParTripCountFloor << ", minAmortWork "
                          << tuning.minAmortWork << ", maxForkCount "
                          << tuning.maxForkCount << "\n");
}

/*static*/ void ParallelMachineSupport::clearGlobalParallelMachineSupport() {
  if (!globalParallelMachineSupport)
    return;
  delete globalParallelMachineSupport;
  globalParallelMachineSupport = nullptr;
}

} // namespace onnx_mlir
