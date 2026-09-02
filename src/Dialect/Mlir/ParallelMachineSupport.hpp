/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ParallelMachineSupport.hpp - Parallel cost model constants --------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Target-derived constants for the parallel-region cost model: how wide a fused
// loop must be to be worth a region, how much work must sit under it, and how
// much entering a region costs. Sibling of VectorMachineSupport, with the same
// lifecycle: set once from the driver before the pipeline runs, immutable
// thereafter, read from anywhere through static accessors.
//
// These are target constants and not pass state. They are derived from the
// command line, fixed before the first pass runs, and identical for every pass,
// which is why they live here rather than being threaded through pattern
// constructors -- exactly as the vector length already is. A transformation
// on/off switch is a different kind of thing and does not belong here; see
// enableCollapse, which is a pass parameter.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PARALLEL_MACHINE_H
#define ONNX_MLIR_PARALLEL_MACHINE_H

#include <cassert>
#include <cstdint>
#include <string>

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// The cost-model constants themselves.

struct ParallelTuning {
  // Raises every site's floor on the width worth a parallel region. The width
  // target is max(this, the site's own minTripCountForParallel), so a value of
  // 0 leaves every site exactly where it is today. This is where a
  // "4 x threads" rule goes once someone has a thread count to put in it.
  int64_t minParWidthFloor;
  // Smallest number of elements one fused iteration must cover for an index
  // recovery chain to stay amortized. Stops a collapsed group from growing
  // onto the innermost level, which preserves both the recovery hoist and the
  // innermost dimension for vectorization.
  int64_t minAmortWork;
  // Largest statically known fork count accepted without penalty, i.e. how
  // many times a region may be entered before its depth stops being free.
  int64_t maxForkCount;
  // Price of an unresolved (dynamic) fork count, in the same milli-units as
  // the index-recovery tiers, so absorbing a level and tolerating it can be
  // compared rather than decreed. The weakest of the four: a threshold with a
  // stated meaning, not a measured number, and the one most likely to differ
  // per target since it prices an OpenMP region entry.
  int64_t forkPenaltyMilli;
};

//===----------------------------------------------------------------------===//
// Generic parallel machine support class, to be refined per target as
// measurements become available.

class ParallelMachineSupport {
protected:
  ParallelMachineSupport() = default;
  virtual ~ParallelMachineSupport() = default;

public:
  // Must call setGlobalParallelMachineSupport once before using any call below.
  // Takes the triple as well as arch/cpu, unlike its vector counterpart:
  // region-entry cost depends on the OpenMP runtime, so z/OS and Linux on Z
  // differ at an identical -march.
  static void setGlobalParallelMachineSupport(const std::string &triple,
      const std::string &arch, const std::string &cpu);
  static void clearGlobalParallelMachineSupport();

  static std::string getArchName() { return pms()->computeArchName(); }

  // The tuning constants for the target, with any command-line override
  // already applied. Valid only after setGlobalParallelMachineSupport.
  static const ParallelTuning &getTuning() { return pms()->tuning; }

protected:
  // Virtual functions that do the actual work. Called by the "get" functions.
  virtual std::string computeArchName() = 0;
  virtual ParallelTuning computeTuning() = 0;

  // Filled by setGlobalParallelMachineSupport from computeTuning(), then
  // overridden per field by whichever command-line options were given.
  ParallelTuning tuning = {};

private:
  static ParallelMachineSupport *pms() {
    assert(
        globalParallelMachineSupport && "parallel machine support undefined");
    return globalParallelMachineSupport;
  }

  static ParallelMachineSupport *globalParallelMachineSupport; // Init to null.
};

//===----------------------------------------------------------------------===//
// One implementation for every target, deliberately.
//
// There is no measurement on any target to differentiate these values, so
// inventing per-target numbers would dress a guess up as data. The point of the
// hook is that differentiating them later is a subclass and not a refactor:
// add a subclass overriding computeTuning, and select it in
// setGlobalParallelMachineSupport.

class GenericParallelMachineSupport : public ParallelMachineSupport {
public:
  GenericParallelMachineSupport() = default;
  virtual ~GenericParallelMachineSupport() = default;

  std::string computeArchName() override { return "generic"; }
  ParallelTuning computeTuning() override {
    return {/*minParWidthFloor=*/0, /*minAmortWork=*/16, /*maxForkCount=*/2,
        /*forkPenaltyMilli=*/4000};
  }
};

} // namespace onnx_mlir
#endif
