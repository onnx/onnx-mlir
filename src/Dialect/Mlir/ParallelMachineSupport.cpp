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
  // Recognize targets from the -march/-mcpu the user gave, as
  // VectorMachineSupport does: IBM Z by its arch number, Apple silicon by an
  // apple-m* arch (what -march=native gives on a Mac). Not from the default
  // (host) triple, so a build without -march -- every lit test that does not
  // pin a target -- gets the generic numbers on any host. A Mac
  // cross-compiling with -march=z17 gets the z17 numbers, as it should.
  int64_t zArchNum = getZArchNum(arch, cpu);
  bool isAppleArm = zArchNum < 0 && arch.rfind("apple-m", 0) == 0;
  if (zArchNum == 15)
    globalParallelMachineSupport = new ZArch15ParallelMachineSupport();
  else if (isAppleArm)
    globalParallelMachineSupport = new AppleArmParallelMachineSupport();
  else
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
  if (parallelForkCost > 0)
    tuning.forkCost = parallelForkCost;
  if (parallelExpectedThreads > 0)
    tuning.expectedThreads = parallelExpectedThreads;
  assert(tuning.expectedThreads > 0 && "expected a positive thread count");
  globalParallelMachineSupport->tuning = tuning;

  LLVM_DEBUG(llvm::dbgs() << "use parallel tuning " << getArchName()
                          << " for triple \"" << triple << "\" arch \"" << arch
                          << "\" cpu \"" << cpu << "\": minParTripCountFloor "
                          << tuning.minParTripCountFloor << ", minAmortWork "
                          << tuning.minAmortWork << ", maxForkCount "
                          << tuning.maxForkCount << ", forkCost "
                          << tuning.forkCost << ", expectedThreads "
                          << tuning.expectedThreads << "\n");
}

/*static*/ void ParallelMachineSupport::clearGlobalParallelMachineSupport() {
  if (!globalParallelMachineSupport)
    return;
  delete globalParallelMachineSupport;
  globalParallelMachineSupport = nullptr;
}

} // namespace onnx_mlir
