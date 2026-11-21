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
  //
  // To investigate: a work-based test (trip count times body cost against
  // forkCost) would decide whether a region has enough work more directly, and
  // so would likely subsume this floor. Kept until measured on more models.
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
  //
  // Still needed with forkCost: z17 measurements (2026-11-20) put index
  // recovery at ~1 ns per fused iteration, ~5 work units, against a ~0.2 ns
  // copy body, so collapsing small bodies does cost time. A candidate for
  // per-target calibration.
  int64_t minAmortWork;
  // Largest statically known fork count accepted without penalty, i.e. how
  // many times a region may be entered before its depth stops being free.
  //
  // To investigate: this is the static counterpart of the per-entry test that
  // forkCost (below) drives for a dynamic prefix. A static prefix of K entries
  // is fine whenever each entry repays its fork/join, so the same test could
  // replace this count. Kept as is until that is measured.
  int64_t maxForkCount;
  // No total price is computed for an unresolved (dynamic) fork count. Earlier
  // drafts summed a flat ~4000 cycles per region entry into the candidate's
  // cost beside the index-rematerialization estimate; that was removed, since
  // the total depends on the number of entries, which is exactly what is
  // unknown when a dynamic level sits above the group, and the flat penalty
  // decided every comparison it appeared in. See the "how a candidate is
  // priced" banner in KrnlParallelPlan.cpp. forkCost below brings the region
  // entry cost back as a per-entry test instead, which needs no entry count.
  //
  // Cost of entering one parallel region (fork/join and the static schedule),
  // in bodyCost's work units, for a typical thread count. Not summed into a
  // candidate's cost, for the reason above: it prices *one entry*. A region
  // under a dynamic level is accepted only if the known work of one entry is at
  // least a few forkCost, so each entry repays itself however many entries
  // there are, which needs no guess at the unknown entry count. Must be > 0.
  int64_t forkCost;
  // Thread count the compile-time decisions assume (the real count is a
  // run-time setting). Used with forkCost, which is measured at that count, for
  // the work per thread of a region entry, and for the width a region needs to
  // keep every thread busy (balanced trip count, about 4 x this). It only
  // steers how far levels are collapsed and which level gets the region, never
  // whether the code is correct.
  int64_t expectedThreads;
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
// Per-target implementations, selected in setGlobalParallelMachineSupport.
//
// Only forkCost is measured per target (with
// amodels/micro-bench/fork_cost, which links onnx-mlir's OpenMP runtime); the
// collapse knobs keep the values they had for every target. A new target is a
// subclass overriding computeTuning, selected in
// setGlobalParallelMachineSupport.

// Unknown targets, and any build without -march. forkCost is the ~4000-cycle
// region entry of an earlier draft, a middle value between the measured
// targets. Regions are placed as before, except that none is placed under a
// dynamic loop for little work per entry.
class GenericParallelMachineSupport : public ParallelMachineSupport {
public:
  GenericParallelMachineSupport() = default;
  virtual ~GenericParallelMachineSupport() = default;

  std::string computeArchName() override { return "generic"; }
  ParallelTuning computeTuning() override {
    return {/*minParTripCountFloor=*/0, /*minAmortWork=*/16,
        /*maxForkCount=*/2, /*forkCost=*/4000, /*expectedThreads=*/4};
  }
};

// IBM z17 (arch15). Measured 2026-11-20 on Linux on Z, one thread per core,
// default wait policy: an empty region costs 86 / 530 / 925 ns at 1 / 4 / 8
// threads and a simple vectorized element op ~0.19 ns, so one work unit is
// ~0.2 ns. forkCost is the 4-thread value (530 ns). Going parallel breaks even
// at about forkCost * T / (T - 1) of work and halves the time at about
// 4 * forkCost on 4 threads.
class ZArch15ParallelMachineSupport : public ParallelMachineSupport {
public:
  ZArch15ParallelMachineSupport() = default;
  virtual ~ZArch15ParallelMachineSupport() = default;

  std::string computeArchName() override { return "z17"; }
  ParallelTuning computeTuning() override {
    return {/*minParTripCountFloor=*/0, /*minAmortWork=*/16,
        /*maxForkCount=*/2, /*forkCost=*/2700, /*expectedThreads=*/4};
  }
};

// Apple silicon Macs, selected by -march=apple-m* (what -march=native gives).
// Measured 2026-11-20 on an M4 Max, threads not pinned: an empty region costs
// ~8.3 us at 4 threads and a simple element op ~0.075 ns, so forkCost is
// ~110,000 work units. A development target: these mostly keep Mac runs from
// being dominated by fork/join.
class AppleArmParallelMachineSupport : public ParallelMachineSupport {
public:
  AppleArmParallelMachineSupport() = default;
  virtual ~AppleArmParallelMachineSupport() = default;

  std::string computeArchName() override { return "apple-arm"; }
  ParallelTuning computeTuning() override {
    return {/*minParTripCountFloor=*/0, /*minAmortWork=*/16,
        /*maxForkCount=*/2, /*forkCost=*/110000, /*expectedThreads=*/4};
  }
};

} // namespace onnx_mlir
#endif
