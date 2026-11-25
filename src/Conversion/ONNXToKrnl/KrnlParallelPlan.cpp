/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlParallelPlan.cpp - Parallel region decision procedure -----===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// The decision procedure behind every parallel region the ONNX-to-Krnl lowering
// emits, and the plan carrying its answer to the krnl.iterate that consumes it.
// See KrnlParallelPlan.hpp for what a call site declares.
//
//===----------------------------------------------------------------------===//
//
// THE ALGORITHM (decideKrnlParallel)
//
// For one loop nest, decide which level, or group of adjacent levels collapsed
// into one, gets the parallel region, or that there is none. Input: the
// iteration bounds, a search window [parFirst, parLast), an optional collapse
// window [parFirst, collapseLast), excluded levels, and the site's cost.
//
// A level is *parallelizable* if it lies in the search window and is not
// excluded, and *collapsible* if it also lies in the collapse window and has a
// literal-zero lower bound (what krnl.collapse requires).
//
// Sizes are only partly known at compile time: every width is a
// TripCountProduct, a literal part and a count of dynamic sizes, and a dynamic
// size is never counted toward a threshold. The numbers that drive the steps:
//
//   target        smallest width worth a region at all (the site's
//                 minTripCount, raised by minParTripCountFloor)
//   T             expected thread count (expectedThreads, -parallel=N)
//   balanced      the width keeps T threads busy: at least the target, and the
//                 slowest thread at most 25% above the average; one dynamic
//                 size is judged by the literal part (BS = 1), two or more
//                 count as balanced
//   forkCost      cost of entering one region, in work units (measured per
//                 machine)
//   maxForkCount  most entries for a region with a known entry count
//   w             work of one parallel iteration: inner trip counts x bodyCost
//
// The steps, one function each:
//
//   STEP 0   no collapse possible (decideNoCollapsePossible): the outermost
//            parallelizable level whose trip count is dynamic or at least the
//            site's minimum; under a dynamic loop (or always, if the site
//            opts in), it must repay its fork cost (no maxForkCount cap, so
//            these sites stay unchanged).
//   STEP 1   a balanced single level (findBalancedSingleLevel): a literal,
//            balanced level under a small static prefix.
//   STEP 2a  build candidate groups (buildCandidateGroups, growCandidateGroup):
//            from each start other than a trip count of 1, grow while not
//            balanced and the work below stays at least minAmortWork.
//   STEP 2b  for each candidate, compute (computeCandidateMetrics): verdict,
//            under a dynamic loop, work per thread per entry, recovery
//            overhead, dynamic sizes.
//   STEP 2c  fork overhead (forkOverheadIsBounded): drop a candidate whose
//            total fork/join cost is unbounded (see below).
//   STEP 2d  pick the best remaining candidate (pickBestCandidate,
//            isRankedHigher): verdict, not under a dynamic loop, lower recovery
//            overhead, more dynamic sizes, fewer levels, shallower start.
//   STEP 3   cutoff (couldMeetTarget): a best candidate with verdict NO means
//            no region.
//
// Fork overhead. Every entry into a region pays forkCost, and what must stay
// bounded is the total. With a known entry count (a static prefix), it is
// capped: at most maxForkCount entries, so at most maxForkCount x forkCost,
// even if one entry does little work. With an unknown one (a dynamic prefix),
// nothing caps the count, so each entry must repay its own cost: work per
// thread per entry at least forkCost (repaysForkCost). A site may opt in to
// that test with a static prefix too (KrnlParallelCost::mustRepayFork),
// on top of the maxForkCount cap.
//
// STEP 0 is chosen by the frame when the site makes no collapse claim or no two
// adjacent levels are collapsible; otherwise decideWithPossibleCollapse runs
// STEP 1 (one level suffices, no collapse needed), then 2a-2d, then 3.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/KrnlParallelPlan.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Mlir/ParallelMachineSupport.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Helpers.

// Products of large literal trip counts can overflow; saturate instead (a
// saturated value is above any threshold and drives a quotient to 0).
static int64_t saturatingMul(int64_t a, int64_t b) {
  assert(a >= 0 && b >= 0 && "expected non-negative trip counts");
  int64_t r;
  if (llvm::MulOverflow(a, b, r))
    return std::numeric_limits<int64_t>::max();
  return r;
}

namespace {
enum class AtLeast { NO = 0, MAYBE = 1, YES = 2 };

// A product of trip counts: the product of its literal factors, and the number
// of dynamic ones.
struct TripCountProduct {
  int64_t literalProduct = 1;
  int64_t dynSizeCount = 0;

  // Multiply in a literal factor. A negative one (a trip count with ub < lb) is
  // an empty loop: factor 0.
  TripCountProduct &mulLiteral(int64_t factor) {
    literalProduct =
        saturatingMul(literalProduct, std::max<int64_t>(0, factor));
    return *this;
  }
  // Multiply in a trip count: its literal value, or one more dynamic size.
  TripCountProduct &mul(IndexExpr tripCount) {
    if (tripCount.isLiteral())
      return mulLiteral(tripCount.getLiteral());
    ++dynSizeCount;
    return *this;
  }

  // Is the product at least `target`? YES on its literal part alone, MAYBE if
  // it has a dynamic factor, NO otherwise.
  AtLeast atLeast(int64_t target) const {
    if (literalProduct >= target)
      return AtLeast::YES;
    return dynSizeCount > 0 ? AtLeast::MAYBE : AtLeast::NO;
  }
};
} // namespace

// Does a width P keep T threads busy? The slowest thread runs ceil(P/T)
// iterations; it splits evenly when that is at most 25% above the average P/T.
// So 12 on 4 threads splits evenly and 5 does not.
static constexpr int64_t kMaxSlowestToAveragePercent = 125;

static bool splitsEvenly(int64_t width, int64_t threads) {
  if (width <= 0)
    return true; // Nothing to balance.
  assert(threads > 0 && "expected a positive thread count");
  int64_t slowest = llvm::divideCeil(width, threads);
  return saturatingMul(100, saturatingMul(slowest, threads)) <=
         saturatingMul(kMaxSlowestToAveragePercent, width);
}

// Cycles to recover one collapsed level's index from the collapsed loop's: 2
// for a power-of-two literal (shift and mask), 6 for another literal
// (multiply-high), 40 for a dynamic size (hardware divide). Rough estimates;
// only the ratios matter.
static int64_t indexRecoveryCostInCycles(IndexExpr tripCount) {
  if (!tripCount.isLiteral())
    return 40;
  return llvm::isPowerOf2_64(tripCount.getLiteral()) ? 2 : 6;
}

//===----------------------------------------------------------------------===//
// The decision, step by step (see the banner at the top of this file).

namespace {
// All info needed to make a decision: the nest's bounds, the windows and
// exclusions of the site, its cost, and the target's tuning, named as in
// decideKrnlParallel; plus the questions every step asks about them. The
// constructor clamps the windows to the nest (a collapse window of -1 stays -1:
// NO_COLLAPSE_CLAIM). Built once per decision and only read.
struct ParallelNestDesc {
  ParallelNestDesc(ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
      int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
      ArrayRef<int64_t> exclusiveDims, int64_t collapseLastExclusiveDim,
      KrnlParallelCost cost)
      : lbs(lbs), ubs(ubs),
        parFirstInclusiveDim(std::max<int64_t>(parFirstInclusiveDim, 0)),
        parLastExclusiveDim(std::min<int64_t>(parLastExclusiveDim, lbs.size())),
        collapseLastExclusiveDim(
            std::min<int64_t>(collapseLastExclusiveDim, lbs.size())),
        exclusiveDims(exclusiveDims), cost(cost),
        tune(ParallelMachineSupport::getTuning()) {
    assert(lbs.size() == ubs.size() && "expected identical ranks for lb/ub");
  }

  // Data needed for the decision.
  ArrayRef<IndexExpr> lbs, ubs;
  int64_t parFirstInclusiveDim, parLastExclusiveDim, collapseLastExclusiveDim;
  ArrayRef<int64_t> exclusiveDims;
  KrnlParallelCost cost;
  const ParallelTuning &tune;

  // The glossary of the algorithm: target and T (forkCost is tune.forkCost).
  int64_t target() const {
    return std::max(cost.minTripCount, tune.minParTripCountFloor);
  }
  int64_t threads() const { return tune.expectedThreads; }

  int64_t rank() const { return lbs.size(); }

  IndexExpr tripCount(int64_t level) const { return ubs[level] - lbs[level]; }
  // Width of levels [first, end): literal product and count of dynamic sizes.
  TripCountProduct width(int64_t first, int64_t end) const {
    TripCountProduct prod;
    for (int64_t i = std::max<int64_t>(first, 0),
                 e = std::min<int64_t>(end, rank());
         i < e; ++i)
      prod.mul(tripCount(i));
    return prod;
  }
  // Times a region starting at `first` is entered: the levels outside it.
  TripCountProduct entries(int64_t first) const { return width(0, first); }
  bool isUnderDynamicLoop(int64_t first) const {
    return entries(first).dynSizeCount > 0;
  }
  // Is a region starting at `first` entered more than maxForkCount times by a
  // static prefix? (A dynamic prefix is not counted here.)
  bool exceedsMaxForkCount(int64_t first) const {
    TripCountProduct prefix = entries(first);
    return prefix.dynSizeCount == 0 &&
           prefix.literalProduct > tune.maxForkCount;
  }
  // w: work of one parallel iteration of a region ending at `end` (inner trip
  // counts x bodyCost).
  TripCountProduct workPerIteration(int64_t end) const {
    return width(end, rank()).mulLiteral(cost.bodyCost);
  }

  // May the level carry a region by itself? In the search window and not
  // excluded. (The collapse window may be narrower: levels in between can be
  // parallel, just not collapsed.)
  bool canParallelize(int64_t level) const {
    return level >= parFirstInclusiveDim && level < parLastExclusiveDim &&
           !llvm::is_contained(exclusiveDims, level);
  }
  // May the level be collapsed with its neighbors? In the collapse window, not
  // excluded, and with a literal-0 lower bound: krnl.collapse takes each
  // level's upper bound as its trip count. (A krnl.block tile loop qualifies;
  // its bound is counted in tiles.)
  bool canCollapse(int64_t level) const {
    return level >= parFirstInclusiveDim && level < collapseLastExclusiveDim &&
           !llvm::is_contained(exclusiveDims, level) &&
           lbs[level].isLiteralAndIdenticalTo(0);
  }
  // Length of the longest run of consecutive collapsible levels.
  int64_t longestCollapsibleRun() const {
    int64_t longest = 0;
    for (int64_t i = parFirstInclusiveDim, run = 0;
         i < collapseLastExclusiveDim; ++i) {
      run = canCollapse(i) ? run + 1 : 0;
      longest = std::max(longest, run);
    }
    return longest;
  }
};

// A candidate group: levels [first, end).
struct CandidateGroup {
  int64_t first;
  int64_t end;
};

// STEP 2b's metrics, one field per metric STEP 2c and 2d use.
struct CandidateMetrics {
  CandidateGroup group;
  AtLeast verdict = AtLeast::NO;     // Width against the target.
  bool underDynamicLoop = false;     // Entered an unknown number of times.
  int64_t w = 0;                     // Known work of one parallel iteration.
  int64_t workPerThreadPerEntry = 0; // ceil(P/T) * w, known factors only.
  int64_t recoveryCycles = 0;        // Index recovery per collapsed iteration.
  int64_t recoveryPercentOfWork = 0; // recoveryCycles as a percent of w.
  int64_t dynSizeCount = 0;          // Dynamic sizes the group covers.
  int64_t numLevels() const { return group.end - group.first; }
};
} // namespace

// Heuristic: a width with two or more dynamic sizes is assumed large and
// balanced, as they are unlikely to all be small at once.
static bool assumedLargeWidth(TripCountProduct width) {
  return width.dynSizeCount >= 2;
}

// Verdict: is a width at least the target? MAYBE is promoted to YES for an
// assumedLargeWidth.
static AtLeast targetVerdict(
    const ParallelNestDesc &desc, TripCountProduct width) {
  if (assumedLargeWidth(width))
    return AtLeast::YES;
  return width.atLeast(desc.target());
}

// Balanced: at least the target and splitsEvenly on its literal part (with
// one dynamic size, that is the BS = 1 case), or an assumedLargeWidth.
static bool isBalanced(const ParallelNestDesc &desc, TripCountProduct width) {
  if (assumedLargeWidth(width))
    return true;
  return width.literalProduct >= desc.target() &&
         splitsEvenly(width.literalProduct, desc.threads());
}

// ceil(P/T) * w for a region over [first, end), known factors only.
static int64_t workPerThreadPerEntry(
    const ParallelNestDesc &desc, int64_t first, int64_t end) {
  int64_t threads = desc.threads();
  int64_t width = desc.width(first, end).literalProduct;
  int64_t iterationsPerThread = llvm::divideCeil(width, threads);
  return saturatingMul(
      iterationsPerThread, desc.workPerIteration(end).literalProduct);
}

// Does each entry into a region over [first, end) repay its fork/join F?
// ceil(P/T) * w >= F keeps every entry ~2x faster than sequential at 4 threads
// (without it, GridFM had one region per edge for 8 elements of work).
static bool repaysForkCost(
    const ParallelNestDesc &desc, int64_t first, int64_t end) {
  return workPerThreadPerEntry(desc, first, end) >= desc.tune.forkCost;
}

// The per-entry half of the fork-overhead rule, as this site requires it.
// Under a dynamic loop, each entry must repay its fork cost. With a static
// prefix, only if the site opts in (mustRepayFork), and then a dynamic
// size in or below the region gets the benefit of the doubt: it may be large,
// as in STEP 0's search, and treating it as 1 would drop every dynamic shape.
static bool meetsPerEntryForkRule(
    const ParallelNestDesc &desc, int64_t first, int64_t end) {
  if (desc.isUnderDynamicLoop(first))
    return repaysForkCost(desc, first, end);
  if (!desc.cost.mustRepayFork || !desc.tune.honorMustRepayFork)
    return true;
  return desc.width(first, desc.rank()).dynSizeCount > 0 ||
         repaysForkCost(desc, first, end);
}

// Is the total fork/join overhead of a region over [first, end) bounded? With
// a known entry count, by capping it at maxForkCount; with an unknown one, by
// having each entry repay its own fork cost; with mustRepayFork, both.
// (See "Fork overhead" at the top.)
static bool forkOverheadIsBounded(
    const ParallelNestDesc &desc, int64_t first, int64_t end) {
  return !desc.exceedsMaxForkCount(first) &&
         meetsPerEntryForkRule(desc, first, end);
}

// STEP 0's search: the outermost parallelizable level whose trip count is
// dynamic or at least minTripCount; NO_PAR_FOUND if none.
static int64_t findOutermostSuitableLevel(const ParallelNestDesc &desc) {
  // The site's own minimum, not target(): no minParTripCountFloor here, so
  // these sites keep their pre-collapse threshold.
  int64_t minSize = desc.cost.minTripCount;
  for (int64_t level = desc.parFirstInclusiveDim;
       level < desc.parLastExclusiveDim; ++level) {
    if (!desc.canParallelize(level))
      continue; // Excluded: skip this level, keep scanning.
    IndexExpr tripCount = desc.tripCount(level);
    if (!tripCount.isLiteral()) {
      LLVM_DEBUG(
          llvm::dbgs() << "Pick level " << level << " because ub is dyn\n");
      return level;
    }
    if (tripCount.getLiteral() >= minSize) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Pick level " << level << " as " << tripCount.getLiteral()
                 << " is at least " << minSize << "\n");
      return level;
    }
  }
  return NO_PAR_FOUND;
}

// STEP 0: no collapse possible. The plain single-level search, then only the
// dynamic half of the fork-overhead rule: under a dynamic loop, the level must
// repay its fork cost (e.g. for the level left when a site excludes the outer
// one). A static prefix is not capped at maxForkCount, so these sites stay as
// they were before collapse.
static KrnlParallelDecision decideNoCollapsePossible(
    const ParallelNestDesc &desc) {
  int64_t level = findOutermostSuitableLevel(desc);
  if (level == NO_PAR_FOUND)
    return {};
  if (!meetsPerEntryForkRule(desc, level, level + 1)) {
    LLVM_DEBUG(llvm::dbgs()
               << "STEP 0: reject level " << level
               << ": does not repay the fork cost"
               << (desc.isUnderDynamicLoop(level) ? " under a dynamic loop"
                                                  : "")
               << "\n");
    return {};
  }
  return KrnlParallelDecision(level);
}

// STEP 1: a balanced single level: parallelizable, with a literal trip count,
// under a static prefix and with its fork overhead bounded (at most
// maxForkCount entries; each repaying its fork cost if the site opts in).
// Dynamic sizes are left to STEP 2. Collapsibility is not needed: nothing is
// collapsed.
static std::optional<KrnlParallelDecision> findBalancedSingleLevel(
    const ParallelNestDesc &desc) {
  for (int64_t level = desc.parFirstInclusiveDim;
       level < desc.parLastExclusiveDim; ++level) {
    if (!desc.canParallelize(level) || !desc.tripCount(level).isLiteral() ||
        !isBalanced(desc, desc.width(level, level + 1)))
      continue;
    TripCountProduct entries = desc.entries(level);
    if (desc.isUnderDynamicLoop(level) ||
        !forkOverheadIsBounded(desc, level, level + 1))
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "STEP 1: level " << level << ", trip count "
               << desc.tripCount(level).getLiteral() << ", balanced on "
               << desc.threads() << " threads, entered "
               << entries.literalProduct << " times\n");
    return KrnlParallelDecision(level);
  }
  return std::nullopt;
}

// Could the work of one parallel iteration of a region ending at `end` be at
// least minAmortWork? (MAYBE counts: a dynamic size may make it so.)
static bool workBelowCouldAmortize(const ParallelNestDesc &desc, int64_t end) {
  return desc.workPerIteration(end).atLeast(desc.tune.minAmortWork) !=
         AtLeast::NO;
}

// STEP 2a, growth: from `start`, absorb inner levels of the run ending at
// `runEnd` while the group is not balanced and the work below it stays at least
// minAmortWork (keeps collapse off the vectorized innermost loop). Not tied to
// the recovery cost of the added level: growth only happens when the
// alternative is less width (at BS = 1, none), and pays off while recovery
// < (T - 1) x work, which minAmortWork already ensures.
static int64_t growCandidateGroup(
    const ParallelNestDesc &desc, int64_t start, int64_t runEnd) {
  int64_t end = start + 1;
  while (end < runEnd && !isBalanced(desc, desc.width(start, end)) &&
         workBelowCouldAmortize(desc, end + 1))
    ++end;
  return end;
}

// STEP 2a: build candidate groups, one per start level in each run of
// collapsible levels, skipping a start of trip count 1. (A start entered too
// often is dropped in STEP 2c.)
static SmallVector<CandidateGroup> buildCandidateGroups(
    const ParallelNestDesc &desc) {
  SmallVector<CandidateGroup> groups;
  int64_t runStart = desc.parFirstInclusiveDim;
  while (runStart < desc.collapseLastExclusiveDim) {
    if (!desc.canCollapse(runStart)) {
      ++runStart;
      continue;
    }
    int64_t runEnd = runStart;
    while (runEnd < desc.collapseLastExclusiveDim && desc.canCollapse(runEnd))
      ++runEnd;
    for (int64_t start = runStart; start < runEnd; ++start) {
      if (desc.tripCount(start).isLiteralAndIdenticalTo(1))
        continue;
      groups.push_back({start, growCandidateGroup(desc, start, runEnd)});
    }
    runStart = runEnd;
  }
  return groups;
}

// STEP 2b: the metrics of one candidate.
static CandidateMetrics computeCandidateMetrics(
    const ParallelNestDesc &desc, CandidateGroup group) {
  CandidateMetrics m;
  m.group = group;
  TripCountProduct width = desc.width(group.first, group.end);
  m.verdict = targetVerdict(desc, width);
  m.underDynamicLoop = desc.isUnderDynamicLoop(group.first);
  m.w = desc.workPerIteration(group.end).literalProduct;
  m.workPerThreadPerEntry = workPerThreadPerEntry(desc, group.first, group.end);
  // One index recovery per member below the head.
  for (int64_t level = group.first + 1; level < group.end; ++level)
    m.recoveryCycles += indexRecoveryCostInCycles(desc.tripCount(level));
  m.recoveryPercentOfWork =
      (100 * m.recoveryCycles) / std::max<int64_t>(1, m.w);
  m.dynSizeCount = width.dynSizeCount;
  return m;
}

// STEP 2d's order: is `a` strictly better than `b`? Lexicographic, the first
// key that differs decides.
static bool isRankedHigher(
    const CandidateMetrics &a, const CandidateMetrics &b) {
  // A width known to meet the target beats one that may, which beats one that
  // cannot.
  if (a.verdict != b.verdict)
    return a.verdict > b.verdict;
  // A region entered a known number of times beats one whose entry count is
  // dynamic.
  if (a.underDynamicLoop != b.underDynamicLoop)
    return !a.underDynamicLoop;
  // Less index recovery per unit of work.
  if (a.recoveryPercentOfWork != b.recoveryPercentOfWork)
    return a.recoveryPercentOfWork < b.recoveryPercentOfWork;
  // More dynamic sizes: stays wide when some turn out to be 1.
  if (a.dynSizeCount != b.dynSizeCount)
    return a.dynSizeCount > b.dynSizeCount;
  // Fewer levels: less to collapse.
  if (a.numLevels() != b.numLevels())
    return a.numLevels() < b.numLevels();
  // Shallower start: the region is entered fewer times.
  return a.group.first < b.group.first;
}

// STEP 2d: pick the best remaining candidate; the first of equals wins.
static std::optional<CandidateMetrics> pickBestCandidate(
    ArrayRef<CandidateMetrics> candidates) {
  std::optional<CandidateMetrics> best;
  for (const CandidateMetrics &c : candidates)
    if (!best || isRankedHigher(c, *best))
      best = c;
  return best;
}

// STEP 3: cutoff. A verdict of NO (small literal width) means the target
// cannot be met: no region.
static bool couldMeetTarget(const CandidateMetrics &best) {
  return best.verdict != AtLeast::NO;
}

#ifndef NDEBUG
static void printCandidateMetrics(
    const ParallelNestDesc &desc, const CandidateMetrics &m, bool kept) {
  static const char *verdictName[] = {"NO", "MAYBE", "YES"};
  // Which half of the fork-overhead rule dropped it.
  const char *dropped =
      desc.exceedsMaxForkCount(m.group.first)
          ? "  -> dropped: entered more than maxForkCount times"
          : "  -> dropped: does not repay the fork cost";
  llvm::dbgs() << "  candidate [" << m.group.first << ", " << m.group.end
               << "): verdict " << verdictName[static_cast<int>(m.verdict)]
               << (m.underDynamicLoop ? ", under dynamic loop" : "") << ", w "
               << m.w << ", work/thread/entry " << m.workPerThreadPerEntry
               << ", recovery " << m.recoveryCycles
               << " cyc = " << m.recoveryPercentOfWork
               << "% of w, dynamic sizes " << m.dynSizeCount
               << (kept ? "" : dropped) << "\n";
}
#endif

// The collapse-possible path: STEP 1, then STEP 2a-2d, then STEP 3.
static KrnlParallelDecision decideWithPossibleCollapse(
    const ParallelNestDesc &desc) {
  // STEP 1: a balanced single level; no need to collapse.
  if (std::optional<KrnlParallelDecision> level = findBalancedSingleLevel(desc))
    return *level;

  LLVM_DEBUG(llvm::dbgs() << "STEP 2: candidates\n");
  SmallVector<CandidateMetrics> candidates;
  // STEP 2a builds the groups; for each, 2b computes its metrics and 2c drops
  // it if its fork overhead is unbounded.
  for (CandidateGroup group : buildCandidateGroups(desc)) {
    CandidateMetrics m = computeCandidateMetrics(desc, group);
    bool kept = forkOverheadIsBounded(desc, group.first, group.end);
    LLVM_DEBUG(printCandidateMetrics(desc, m, kept));
    if (kept)
      candidates.push_back(m);
  }
  // STEP 2d: pick the best remaining candidate.
  std::optional<CandidateMetrics> best = pickBestCandidate(candidates);

  // STEP 3: cutoff.
  if (!best || !couldMeetTarget(*best)) {
    LLVM_DEBUG(llvm::dbgs() << "STEP 3: nothing worth a region\n");
    return {};
  }
  LLVM_DEBUG(llvm::dbgs() << "STEP 2d: pick [" << best->group.first << ", "
                          << best->group.end << ")\n");
  return KrnlParallelDecision(best->group.first, best->numLevels());
}

//===----------------------------------------------------------------------===//
// The frame.

KrnlParallelDecision decideKrnlParallel(ArrayRef<IndexExpr> lbs,
    ArrayRef<IndexExpr> ubs, int64_t parFirstInclusiveDim,
    int64_t parLastExclusiveDim, ArrayRef<int64_t> exclusiveDims,
    int64_t collapseLastExclusiveDim, KrnlParallelCost cost) {
  ParallelNestDesc desc(lbs, ubs, parFirstInclusiveDim, parLastExclusiveDim,
      exclusiveDims, collapseLastExclusiveDim, cost);
  // STEP 0 when no collapse is claimed or possible (collapse needs two adjacent
  // collapsible levels; also keeps such sites unchanged), else the
  // collapse-possible path.
  if (desc.collapseLastExclusiveDim == NO_COLLAPSE_CLAIM ||
      desc.longestCollapsibleRun() < 2)
    return decideNoCollapsePossible(desc);
  return decideWithPossibleCollapse(desc);
}

//===----------------------------------------------------------------------===//
// KrnlParallelPlan.

// Shared reporting, so the two entry points cannot drift apart.
static int64_t reportKrnlParallelDecision(KrnlParallelDecision decision,
    Operation *op, const std::string &msg, ArrayRef<IndexExpr> lbs,
    ArrayRef<IndexExpr> ubs) {
  if (!decision.hasParallel()) {
    // Trip count 0 means "no parallel loop" (-1 means "runtime only").
    onnxToKrnlParallelReport(
        op, false, -1, 0, "no par dim with enough work in " + msg);
    return NO_PAR_FOUND;
  }
  int64_t first = decision.firstDim;
  int64_t end = first + decision.numDims;
  if (!decision.isCollapse()) {
    onnxToKrnlParallelReport(op, true, first, lbs[first], ubs[first], msg);
    return first;
  }
  // Report the group's collapsed trip count: the product if all members are
  // literal, else -1 ("runtime only").
  assert(first >= 0 && end <= (int64_t)lbs.size() &&
         "collapsed group must lie inside the bounds it is reported over");
  TripCountProduct collapsedTripCount;
  for (int64_t level = first; level < end; ++level)
    collapsedTripCount.mul(ubs[level] - lbs[level]);
  // Name both ends of the group. No comma: the report line is comma-separated.
  onnxToKrnlParallelReport(op, /*successful*/ true, first,
      collapsedTripCount.dynSizeCount == 0 ? collapsedTripCount.literalProduct
                                           : -1,
      msg + " with " + std::to_string(decision.numDims) +
          " loops collapsed at levels " + std::to_string(first) + "-" +
          std::to_string(end - 1));
  return first;
}

KrnlParallelPlan::KrnlParallelPlan(ValueRange loopDef,
    int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
    int64_t collapseLastExclusiveDim, KrnlParallelCost cost,
    ArrayRef<int64_t> exclusiveDims)
    : optLoopDef(loopDef.begin(), loopDef.end()),
      parFirstInclusiveDim(parFirstInclusiveDim),
      parLastExclusiveDim(parLastExclusiveDim),
      collapseLastExclusiveDim(collapseLastExclusiveDim), cost(cost),
      exclusiveDims(exclusiveDims.begin(), exclusiveDims.end()) {
  // Collapsing substitutes into the loop refs, so it needs some.
  assert(
      (collapseLastExclusiveDim == NO_COLLAPSE_CLAIM || !optLoopDef.empty()) &&
      "a collapse-eligible plan needs loop refs to collapse");
}

KrnlParallelPlan::KrnlParallelPlan(ValueRange loopDef, bool enableCollapse,
    int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
    int64_t collapseLastExclusiveDim, KrnlParallelCost cost,
    ArrayRef<int64_t> exclusiveDims)
    : KrnlParallelPlan(loopDef, parFirstInclusiveDim, parLastExclusiveDim,
          // No flag or a rank-0 nest (nothing to collapse): a noCollapse plan.
          enableCollapse && !loopDef.empty() ? collapseLastExclusiveDim
                                             : NO_COLLAPSE_CLAIM,
          cost, exclusiveDims) {}

/*static*/ KrnlParallelPlan KrnlParallelPlan::noCollapse(ValueRange loopDef,
    int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
    KrnlParallelCost cost, ArrayRef<int64_t> exclusiveDims) {
  return KrnlParallelPlan(loopDef, parFirstInclusiveDim, parLastExclusiveDim,
      NO_COLLAPSE_CLAIM, cost, exclusiveDims);
}

/*static*/ KrnlParallelPlan KrnlParallelPlan::noLoopRefs(
    int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
    KrnlParallelCost cost, ArrayRef<int64_t> exclusiveDims) {
  KrnlParallelPlan plan(/*loopDef=*/{}, parFirstInclusiveDim,
      parLastExclusiveDim, NO_COLLAPSE_CLAIM, cost, exclusiveDims);
  // Lets tryCreateParallel tell this plan from a rank-0 one.
  plan.noLoopRefsByDesign = true;
  return plan;
}

KrnlParallelPlan::~KrnlParallelPlan() {
  // A collapse no krnl.iterate consumes would be silently dropped, leaving its
  // krnl.parallel on a ref with no loop. Plans that never collapsed may go
  // unread.
  assert((!collapsed || consumed) &&
         "a krnl.collapse was emitted into this plan but its "
         "optimizedLoopDef() was never passed to a krnl.iterate");
}

ValueRange KrnlParallelPlan::optimizedLoopDef() const {
  // May be empty for a rank-0 nest (e.g. rank-0 Gather or Expand, or a full
  // Reduction with keepdims=0).
  consumed = true;
  return ValueRange(optLoopDef);
}

Value KrnlParallelPlan::collapseIfNeeded(
    const KrnlBuilder &createKrnl, KrnlParallelDecision decision) {
  assert(decision.hasParallel() && "expected a decision to apply");
  // Sites may pass fewer refs than bounds; the decision must land inside them.
  assert(decision.firstDim >= 0 &&
         decision.firstDim + decision.numDims <= (int64_t)optLoopDef.size() &&
         "decision must land inside the loop refs it indexes");
  if (!decision.isCollapse())
    return optLoopDef[decision.firstDim];
  // Collapse the group and substitute the collapsed ref for its entries.
  ValueRange group(
      ArrayRef<Value>(optLoopDef).slice(decision.firstDim, decision.numDims));
  Value collapsedRef = createKrnl.collapse(group);
  optLoopDef.erase(optLoopDef.begin() + decision.firstDim,
      optLoopDef.begin() + decision.firstDim + decision.numDims);
  optLoopDef.insert(optLoopDef.begin() + decision.firstDim, collapsedRef);
  collapsed = true;
  return collapsedRef;
}

int64_t KrnlParallelPlan::findParallelDim(Operation *op, std::string msg,
    ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs) const {
  // Single level only (NO_COLLAPSE_CLAIM). Not clamped to the refs: a
  // noLoopRefs plan has none.
  KrnlParallelDecision decision =
      decideKrnlParallel(lbs, ubs, parFirstInclusiveDim, parLastExclusiveDim,
          exclusiveDims, NO_COLLAPSE_CLAIM, cost);
  assert(!decision.isCollapse() && "asking cannot yield a group");
  return reportKrnlParallelDecision(decision, op, msg, lbs, ubs);
}

int64_t KrnlParallelPlan::tryCreateParallel(const KrnlBuilder &createKrnl,
    Operation *op, std::string msg, ArrayRef<IndexExpr> lbs,
    ArrayRef<IndexExpr> ubs) {
  assert(!noLoopRefsByDesign &&
         "emitting needs loop refs; a site that only asks wants "
         "findParallelDim on a noLoopRefs plan");
  // A rank-0 nest: no parallelism, but still reported.
  if (optLoopDef.empty()) {
    onnxToKrnlParallelReport(op, /*successful*/ false, -1, /*trip count*/ 0,
        "rank-0 nest in " + msg);
    return NO_PAR_FOUND;
  }
  // Clamp the windows to the refs (NO_COLLAPSE_CLAIM stays as is).
  int64_t refCount = optLoopDef.size();
  KrnlParallelDecision decision = decideKrnlParallel(lbs, ubs,
      parFirstInclusiveDim, std::min(parLastExclusiveDim, refCount),
      exclusiveDims, std::min(collapseLastExclusiveDim, refCount), cost);
  if (decision.hasParallel())
    createKrnl.parallel(collapseIfNeeded(createKrnl, decision));
  return reportKrnlParallelDecision(decision, op, msg, lbs, ubs);
}

} // namespace onnx_mlir
