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

// Every cost in the collapse model is a count of clock cycles. The counts are
// stated estimates, not measurements, and are deliberately coarse: only their
// ratios are ever used, so being off by a factor of two in the same direction
// everywhere changes nothing.
struct ParallelTuning {
  // Raises every site's floor on the trip count worth a parallel region. The
  // target is max(this, the site's own minTripCountForParallel), so a value of
  // 0 leaves every site exactly where it is today. This is where a
  // "4 x threads" rule goes once someone has a thread count to put in it.
  int64_t minParTripCountFloor;
  // Smallest amount of work one fused iteration must cover for an index
  // rematerialization chain to stay amortized, in KrnlParallelCost::bodyCost's
  // work units -- one scalar op or one copied element per unit -- since that is
  // what it is compared against.
  //
  // A *depth* guard, not a worth-it guard: it stops a collapsed group from
  // growing onto the innermost level, preserving both the hoist of the recovery
  // arithmetic and that level for vectorization. Raising it makes groups
  // shallower, which is not the same as making them cheaper -- at
  // LayoutTransform's fast path, raising it past the body's own cost drops the
  // group entirely and lands the region on the innermost level under an unknown
  // entry count, which is worse. Which *start* wins among candidate groups is
  // decided by the ordering in GroupCandidate, not here.
  int64_t minAmortWork;
  // Largest statically known fork count accepted without penalty, i.e. how
  // many times a region may be entered before its depth stops being free.
  int64_t maxForkCount;
  // There is deliberately no price for an unresolved (dynamic) fork count.
  // Earlier drafts carried one -- a flat ~4000 cycles for an OpenMP region
  // entry
  // -- summed into the candidate's cost beside the index-rematerialization
  // estimate. That was removed rather than retuned, because pricing it is not
  // possible with the information available: the cost of entering a region is
  // paid per entry, and the number of entries is exactly what is unknown when a
  // dynamic level sits above the group. See the "how a candidate is priced"
  // banner in KrnlParallelPlan.cpp. An unresolved fork count is now a gate in
  // the candidate ordering, which is what it always was in practice -- the flat
  // penalty was two orders of magnitude above the tiers it was summed with, so
  // it decided every comparison it appeared in. maxForkCount below is the knob
  // for the case where the entry count *is* known.
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
    return {
        /*minParTripCountFloor=*/0, /*minAmortWork=*/16, /*maxForkCount=*/2};
  }
};

} // namespace onnx_mlir
#endif
