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
// excluded, and *fusable* if it also lies in the collapse window and has a
// literal-zero lower bound (what krnl.collapse requires).
//
// Sizes are only partly known at compile time: every width is a
// TripCountProduct, a literal part and a count of dynamic sizes, and a dynamic
// size is never counted toward a threshold. The numbers that drive the steps:
//
//   target     smallest width worth a region at all (the site's
//              minTripCountForParallel, raised by minParTripCountFloor)
//   T          expected thread count (expectedThreads, -parallel=N)
//   balanced   the width keeps T threads busy: at least the target, and the
//              slowest thread at most 25% above the average; one dynamic size
//              is judged by the literal part (BS = 1), two or more count as
//              balanced
//   forkCost   cost of entering one region, per target
//   w          work of one parallel iteration: inner trip counts x bodyCost
//
// The steps, one function each:
//
//   STEP 0   no collapse possible (decideNoCollapsePossible): the outermost
//            parallelizable level whose trip count is dynamic or at least the
//            site's minimum, then the fork test.
//   STEP 1   one level that's clearly right (findClearlyRightLevel): a literal,
//            balanced level under a small static prefix.
//   STEP 2a  build candidate groups (buildCandidateGroups, growCandidateGroup):
//            from each start, grow while not balanced and the work below stays
//            at least minAmortWork.
//   STEP 2b  for each candidate, compute (computeCandidateMetrics): verdict,
//            under a dynamic loop, work per thread per entry, recovery
//            overhead, dynamic sizes.
//   STEP 2c  fork test (passesForkTest): under a dynamic loop, the work per
//            thread per entry must be at least forkCost.
//   STEP 2d  rank the remaining candidates (rankCandidates, isRankedHigher):
//            verdict, not under a dynamic loop, lower recovery overhead, more
//            dynamic sizes, fewer levels, shallower start.
//   STEP 3   cutoff (passesCutoff): a best candidate with verdict NO means no
//            region.
//
// STEP 0 is chosen by the frame when the site makes no collapse claim or no two
// adjacent levels are fusable; otherwise decideWithCollapse runs STEP 1, then
// 2a-2d, then 3.
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

// STEP 0's search: the outermost level of [firstInclusiveDim,
// lastExclusiveDim), not excluded, whose trip count is dynamic or at least
// minSize. Sets parDim and returns true if found. The window must already be
// clamped (the frame does it).
static bool findOutermostSuitableParallelDim(ArrayRef<IndexExpr> lb,
    ArrayRef<IndexExpr> ub, int64_t firstInclusiveDim, int64_t lastExclusiveDim,
    ArrayRef<int64_t> exclusiveDims, int64_t &parDim, int64_t minSize) {
  assert(lb.size() == ub.size() && "expected identical ranks for lb/ub");
  assert(firstInclusiveDim >= 0 &&
         lastExclusiveDim <= static_cast<int64_t>(lb.size()) &&
         "expected a window already clamped to the bounds");
  for (int64_t i = firstInclusiveDim; i < lastExclusiveDim; ++i) {
    if (llvm::is_contained(exclusiveDims, i))
      continue; // Excluded: skip this level, keep scanning.
    IndexExpr tripCount = ub[i] - lb[i];
    if (!tripCount.isLiteral()) {
      LLVM_DEBUG(llvm::dbgs() << "Pick dim " << i << " because ub is dyn\n");
      parDim = i;
      return true;
    }
    if (tripCount.getLiteral() >= minSize) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Pick dim " << i << " as " << tripCount.getLiteral()
                 << " greater than " << minSize << "\n");
      parDim = i;
      return true;
    }
  }
  return false;
}

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

// A negative literal trip count (ub < lb) is an empty loop: factor 0.
static int64_t literalTripCountAsFactor(int64_t literal) {
  return std::max<int64_t>(0, literal);
}

namespace {
enum class AtLeast { NO = 0, MAYBE = 1, YES = 2 };

// A product of trip counts: the product of its literal factors, and the number
// of dynamic ones.
struct TripCountProduct {
  int64_t literalProduct = 1;
  int64_t dynDimCount = 0;

  // Is the product at least `target`? YES on its literal part alone, MAYBE if
  // it has a dynamic factor, NO otherwise.
  AtLeast atLeast(int64_t target) const {
    if (literalProduct >= target)
      return AtLeast::YES;
    return dynDimCount > 0 ? AtLeast::MAYBE : AtLeast::NO;
  }
};
} // namespace

// Does a width P keep T threads busy? The slowest thread runs ceil(P/T)
// iterations; balanced when that is at most 25% above the average P/T. So 12 on
// 4 threads is balanced and 5 is not.
static constexpr int64_t kMaxImbalancePercent = 125;

static bool isBalancedWidth(int64_t width, int64_t threads) {
  if (width <= 0)
    return true; // Nothing to balance.
  threads = std::max<int64_t>(1, threads);
  int64_t slowest = (width + threads - 1) / threads;
  return saturatingMul(100, saturatingMul(slowest, threads)) <=
         saturatingMul(kMaxImbalancePercent, width);
}

// Cycles to recover one collapsed level's index from the fused one: 2 for a
// power-of-two literal (shift and mask), 6 for another literal (multiply-high),
// 40 for a dynamic size (hardware divide). Rough estimates; only the ratios
// matter.
static int64_t cycleOverheadForIndexRematerialization(IndexExpr tripCount) {
  if (!tripCount.isLiteral())
    return 40;
  int64_t t = tripCount.getLiteral();
  bool isPowerOfTwo = t > 0 && (t & (t - 1)) == 0;
  return isPowerOfTwo ? 2 : 6;
}

//===----------------------------------------------------------------------===//
// The decision, step by step (see the banner at the top of this file).

namespace {
// Everything a decision is made from: the nest's bounds, the (clamped) windows
// and exclusions of the site, its cost, and the target's tuning, named as in
// decideKrnlParallel; plus the questions every step asks about them. Built once
// per decision and only read.
struct ParallelNestDesc {
  ParallelNestDesc(ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
      int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
      ArrayRef<int64_t> exclusiveDims, int64_t collapseLastExclusiveDim,
      KrnlParallelCost cost)
      : lbs(lbs), ubs(ubs), parFirstInclusiveDim(parFirstInclusiveDim),
        parLastExclusiveDim(parLastExclusiveDim),
        collapseLastExclusiveDim(collapseLastExclusiveDim),
        exclusiveDims(exclusiveDims), cost(cost),
        tune(ParallelMachineSupport::getTuning()) {}

  ArrayRef<IndexExpr> lbs, ubs;
  int64_t parFirstInclusiveDim, parLastExclusiveDim, collapseLastExclusiveDim;
  ArrayRef<int64_t> exclusiveDims;
  KrnlParallelCost cost;
  const ParallelTuning &tune;

  // The glossary of the algorithm: target and T (forkCost is tune.forkCost).
  int64_t target() const {
    return std::max(cost.minTripCountForParallel, tune.minParTripCountFloor);
  }
  int64_t threads() const { return std::max<int64_t>(1, tune.expectedThreads); }

  int64_t rank() const { return lbs.size(); }

  IndexExpr tripCount(int64_t level) const { return ubs[level] - lbs[level]; }
  // Width of levels [first, end): literal product and count of dynamic sizes.
  TripCountProduct width(int64_t first, int64_t end) const {
    TripCountProduct prod;
    for (int64_t i = std::max<int64_t>(first, 0),
                 e = std::min<int64_t>(end, rank());
         i < e; ++i) {
      IndexExpr t = tripCount(i);
      if (t.isLiteral())
        prod.literalProduct = saturatingMul(
            prod.literalProduct, literalTripCountAsFactor(t.getLiteral()));
      else
        ++prod.dynDimCount;
    }
    return prod;
  }
  // Times a region starting at `first` is entered: the levels outside it.
  TripCountProduct entries(int64_t first) const { return width(0, first); }
  bool isUnderDynamicLoop(int64_t first) const {
    return entries(first).dynDimCount > 0;
  }
  // w: work of one parallel iteration of a region ending at `end` (inner trip
  // counts x bodyCost).
  TripCountProduct workPerIteration(int64_t end) const {
    TripCountProduct w = width(end, rank());
    w.literalProduct = saturatingMul(
        w.literalProduct, literalTripCountAsFactor(cost.bodyCost));
    return w;
  }

  // May the level carry a region by itself? In the search window and not
  // excluded. (The collapse window may be narrower: levels in between can be
  // parallel, just not fused.)
  bool canParallelize(int64_t level) const {
    return level >= parFirstInclusiveDim && level < parLastExclusiveDim &&
           !llvm::is_contained(exclusiveDims, level);
  }
  // May the level be fused with its neighbors? In the collapse window, not
  // excluded, and with a literal-0 lower bound: krnl.collapse takes each
  // level's upper bound as its trip count. (A krnl.block tile loop qualifies;
  // its bound is counted in tiles.)
  bool canFuse(int64_t level) const {
    return level >= parFirstInclusiveDim && level < collapseLastExclusiveDim &&
           !llvm::is_contained(exclusiveDims, level) &&
           lbs[level].isLiteral() && lbs[level].getLiteral() == 0;
  }
  // Length of the longest run of consecutive fusable levels.
  int64_t longestFusableRun() const {
    int64_t longest = 0;
    for (int64_t i = parFirstInclusiveDim, run = 0;
         i < collapseLastExclusiveDim; ++i) {
      run = canFuse(i) ? run + 1 : 0;
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

// STEP 2b's metrics, one field per column of the algorithm's candidate table.
struct CandidateMetrics {
  CandidateGroup group;
  AtLeast verdict = AtLeast::NO;     // Width against the target.
  bool underDynamicLoop = false;     // Entered an unknown number of times.
  int64_t w = 0;                     // Known work of one parallel iteration.
  int64_t workPerThreadPerEntry = 0; // ceil(P/T) * w, known factors only.
  int64_t recoveryCycles = 0;        // Index recovery per fused iteration.
  int64_t recoveryOverhead = 0;      // recoveryCycles per 100 units of w.
  int64_t dynamicSizes = 0;          // Dynamic sizes the group covers.
  int64_t numLevels() const { return group.end - group.first; }
  int64_t firstLevel() const { return group.first; }
};
} // namespace

// Verdict: is a width at least the target? Two or more dynamic sizes are
// promoted to YES: unlikely to all be small at once.
static AtLeast meetsTarget(
    const ParallelNestDesc &desc, TripCountProduct width) {
  AtLeast verdict = width.atLeast(desc.target());
  if (verdict == AtLeast::MAYBE && width.dynDimCount >= 2)
    return AtLeast::YES;
  return verdict;
}

// Balanced: at least the target and isBalancedWidth on its literal part (with
// one dynamic size, that is the BS = 1 case). Two or more dynamic sizes count
// as balanced, as in the verdict.
static bool isBalanced(const ParallelNestDesc &desc, TripCountProduct width) {
  if (width.dynDimCount >= 2)
    return true;
  return width.literalProduct >= desc.target() &&
         isBalancedWidth(width.literalProduct, desc.threads());
}

// ceil(P/T) * w for a region over [first, end), known factors only.
static int64_t workPerThreadPerEntry(
    const ParallelNestDesc &desc, int64_t first, int64_t end) {
  int64_t threads = desc.threads();
  int64_t width = desc.width(first, end).literalProduct;
  int64_t iterationsPerThread = (width + threads - 1) / threads;
  return saturatingMul(
      iterationsPerThread, desc.workPerIteration(end).literalProduct);
}

// Fork test: a region under a dynamic loop is entered an unknown number of
// times, so each entry must repay its fork/join F: ceil(P/T) * w >= F keeps
// every entry ~2x faster than sequential at 4 threads (without it, GridFM had
// one region per edge for 8 elements of work).
static bool passesForkTest(const ParallelNestDesc &desc, bool underDynamicLoop,
    int64_t perThreadWork) {
  return !underDynamicLoop || perThreadWork >= desc.tune.forkCost;
}

// STEP 0: no collapse possible. The plain single-level search, then the fork
// test (e.g. for the level left when a site excludes the outer one).
static KrnlParallelDecision decideNoCollapsePossible(
    const ParallelNestDesc &desc) {
  KrnlParallelDecision decision;
  int64_t level = NO_PAR_FOUND;
  if (!findOutermostSuitableParallelDim(desc.lbs, desc.ubs,
          desc.parFirstInclusiveDim, desc.parLastExclusiveDim,
          desc.exclusiveDims, level, desc.cost.minTripCountForParallel))
    return decision;
  if (!passesForkTest(desc, desc.isUnderDynamicLoop(level),
          workPerThreadPerEntry(desc, level, level + 1))) {
    LLVM_DEBUG(llvm::dbgs() << "STEP 0: reject level " << level
                            << ": fails the fork test under a dynamic loop\n");
    return decision;
  }
  decision.firstDim = level;
  decision.numDims = 1;
  return decision;
}

// STEP 1: one level that's clearly right: a parallelizable level with a
// balanced literal trip count, entered at most maxForkCount times. Dynamic
// sizes are left to STEP 2. Fusability is not needed: nothing is collapsed.
static std::optional<KrnlParallelDecision> findClearlyRightLevel(
    const ParallelNestDesc &desc) {
  for (int64_t level = desc.parFirstInclusiveDim;
       level < desc.parLastExclusiveDim; ++level) {
    if (!desc.canParallelize(level) || !desc.tripCount(level).isLiteral() ||
        !isBalanced(desc, desc.width(level, level + 1)))
      continue;
    TripCountProduct entries = desc.entries(level);
    if (entries.dynDimCount > 0 ||
        entries.literalProduct > desc.tune.maxForkCount)
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "STEP 1: level " << level << ", trip count "
               << desc.tripCount(level).getLiteral() << ", balanced on "
               << desc.threads() << " threads, entered "
               << entries.literalProduct << " times\n");
    KrnlParallelDecision decision;
    decision.firstDim = level;
    decision.numDims = 1;
    return decision;
  }
  return std::nullopt;
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
         desc.workPerIteration(end + 1).atLeast(desc.tune.minAmortWork) !=
             AtLeast::NO)
    ++end;
  return end;
}

// STEP 2a: build candidate groups, one per start level in each run of fusable
// levels, skipping a start of trip count 1 or one entered more than
// maxForkCount times by a static prefix.
static SmallVector<CandidateGroup> buildCandidateGroups(
    const ParallelNestDesc &desc) {
  SmallVector<CandidateGroup> groups;
  int64_t runStart = desc.parFirstInclusiveDim;
  while (runStart < desc.collapseLastExclusiveDim) {
    if (!desc.canFuse(runStart)) {
      ++runStart;
      continue;
    }
    int64_t runEnd = runStart;
    while (runEnd < desc.collapseLastExclusiveDim && desc.canFuse(runEnd))
      ++runEnd;
    for (int64_t start = runStart; start < runEnd; ++start) {
      IndexExpr trip = desc.tripCount(start);
      if (trip.isLiteral() && trip.getLiteral() == 1)
        continue;
      TripCountProduct entries = desc.entries(start);
      if (entries.dynDimCount == 0 &&
          entries.literalProduct > desc.tune.maxForkCount)
        continue;
      groups.push_back({start, growCandidateGroup(desc, start, runEnd)});
    }
    runStart = runEnd;
  }
  return groups;
}

// STEP 2b: for each candidate, compute its metrics.
static CandidateMetrics computeCandidateMetrics(
    const ParallelNestDesc &desc, CandidateGroup group) {
  CandidateMetrics m;
  m.group = group;
  TripCountProduct width = desc.width(group.first, group.end);
  m.verdict = meetsTarget(desc, width);
  m.underDynamicLoop = desc.isUnderDynamicLoop(group.first);
  m.w = desc.workPerIteration(group.end).literalProduct;
  m.workPerThreadPerEntry = workPerThreadPerEntry(desc, group.first, group.end);
  // One index recovery per member below the head.
  for (int64_t level = group.first + 1; level < group.end; ++level)
    m.recoveryCycles +=
        cycleOverheadForIndexRematerialization(desc.tripCount(level));
  m.recoveryOverhead = (100 * m.recoveryCycles) / std::max<int64_t>(1, m.w);
  m.dynamicSizes = width.dynDimCount;
  return m;
}

// STEP 2d's order, lexicographic:
//   1. verdict (higher first);
//   2. not under a dynamic loop;
//   3. lower recovery overhead;
//   4. more dynamic sizes (stays wide when some turn out to be 1);
//   5. fewer levels;
//   6. shallower start.
static bool isRankedHigher(
    const CandidateMetrics &a, const CandidateMetrics &b) {
  if (a.verdict != b.verdict)
    return a.verdict > b.verdict;
  if (a.underDynamicLoop != b.underDynamicLoop)
    return !a.underDynamicLoop;
  if (a.recoveryOverhead != b.recoveryOverhead)
    return a.recoveryOverhead < b.recoveryOverhead;
  if (a.dynamicSizes != b.dynamicSizes)
    return a.dynamicSizes > b.dynamicSizes;
  if (a.numLevels() != b.numLevels())
    return a.numLevels() < b.numLevels();
  return a.firstLevel() < b.firstLevel();
}

// STEP 2d: rank the remaining candidates; the first of equals wins.
static std::optional<CandidateMetrics> rankCandidates(
    ArrayRef<CandidateMetrics> candidates) {
  std::optional<CandidateMetrics> best;
  for (const CandidateMetrics &c : candidates)
    if (!best || isRankedHigher(c, *best))
      best = c;
  return best;
}

// STEP 3: cutoff. A verdict of NO (small literal width) means no region.
static bool passesCutoff(const CandidateMetrics &best) {
  return best.verdict != AtLeast::NO;
}

#ifndef NDEBUG
static void printCandidateMetrics(const CandidateMetrics &m, bool kept) {
  static const char *verdictName[] = {"NO", "MAYBE", "YES"};
  llvm::dbgs() << "  candidate [" << m.group.first << ", " << m.group.end
               << "): verdict " << verdictName[static_cast<int>(m.verdict)]
               << (m.underDynamicLoop ? ", under dynamic loop" : "") << ", w "
               << m.w << ", work/thread/entry " << m.workPerThreadPerEntry
               << ", recovery " << m.recoveryCycles
               << " cyc = " << m.recoveryOverhead << " per 100, dynamic sizes "
               << m.dynamicSizes << (kept ? "" : "  -> dropped by fork test")
               << "\n";
}
#endif

// The collapse path: STEP 1, then STEP 2a-2d, then STEP 3.
static KrnlParallelDecision decideWithCollapse(const ParallelNestDesc &desc) {
  if (std::optional<KrnlParallelDecision> level = findClearlyRightLevel(desc))
    return *level;

  LLVM_DEBUG(llvm::dbgs() << "STEP 2: candidates\n");
  SmallVector<CandidateMetrics> candidates;
  for (CandidateGroup group : buildCandidateGroups(desc)) {    // STEP 2a
    CandidateMetrics m = computeCandidateMetrics(desc, group); // STEP 2b
    bool kept = passesForkTest(                                // STEP 2c
        desc, m.underDynamicLoop, m.workPerThreadPerEntry);
    LLVM_DEBUG(printCandidateMetrics(m, kept));
    if (kept)
      candidates.push_back(m);
  }
  std::optional<CandidateMetrics> best = rankCandidates(candidates); // 2d

  KrnlParallelDecision decision;
  if (!best || !passesCutoff(*best)) { // STEP 3
    LLVM_DEBUG(llvm::dbgs() << "STEP 3: nothing worth a region\n");
    return decision;
  }
  LLVM_DEBUG(llvm::dbgs() << "STEP 2d: pick [" << best->group.first << ", "
                          << best->group.end << ")\n");
  decision.firstDim = best->group.first;
  decision.numDims = best->numLevels();
  return decision;
}

//===----------------------------------------------------------------------===//
// The frame.

KrnlParallelDecision decideKrnlParallel(ArrayRef<IndexExpr> lbs,
    ArrayRef<IndexExpr> ubs, int64_t parFirstInclusiveDim,
    int64_t parLastExclusiveDim, ArrayRef<int64_t> exclusiveDims,
    int64_t collapseLastExclusiveDim, KrnlParallelCost cost) {
  assert(lbs.size() == ubs.size() && "expected identical ranks for lb/ub");
  int64_t rank = lbs.size();

  // Clamp the windows to the nest (a collapse window of -1 stays -1: no claim).
  parFirstInclusiveDim = std::max<int64_t>(parFirstInclusiveDim, 0);
  parLastExclusiveDim = std::min(parLastExclusiveDim, rank);
  collapseLastExclusiveDim = std::min(collapseLastExclusiveDim, rank);

  ParallelNestDesc desc(lbs, ubs, parFirstInclusiveDim, parLastExclusiveDim,
      exclusiveDims, collapseLastExclusiveDim, cost);
  // STEP 0 when no collapse is claimed or possible (collapse needs two adjacent
  // fusable levels; also keeps such sites unchanged), else the collapse path.
  if (collapseLastExclusiveDim < 0 || desc.longestFusableRun() < 2)
    return decideNoCollapsePossible(desc);
  return decideWithCollapse(desc);
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
  int64_t parId = decision.firstDim;
  if (!decision.isCollapse()) {
    onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId], msg);
    return parId;
  }
  // Report the group's fused trip count: the product if all members are
  // literal, else -1 ("runtime only").
  assert(parId >= 0 && parId + decision.numDims <= (int64_t)lbs.size() &&
         "collapsed group must lie inside the bounds it is reported over");
  int64_t fusedTripCount = 1;
  bool allLiteral = true;
  for (int64_t k = parId; k < parId + decision.numDims; ++k) {
    IndexExpr tripCount = ubs[k] - lbs[k];
    if (tripCount.isLiteral())
      fusedTripCount *= tripCount.getLiteral();
    else
      allLiteral = false;
  }
  // Name both ends of the group. No comma: the report line is comma-separated.
  onnxToKrnlParallelReport(op, /*successful*/ true, parId,
      allLiteral ? fusedTripCount : -1,
      msg + " with " + std::to_string(decision.numDims) +
          " loops collapsed at levels " + std::to_string(parId) + "-" +
          std::to_string(parId + decision.numDims - 1));
  return parId;
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
  assert((collapseLastExclusiveDim < 0 || !optLoopDef.empty()) &&
         "a collapse-eligible plan needs loop refs to fuse");
}

KrnlParallelPlan::KrnlParallelPlan(ValueRange loopDef, bool enableCollapse,
    int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
    int64_t collapseLastExclusiveDim, KrnlParallelCost cost,
    ArrayRef<int64_t> exclusiveDims)
    : KrnlParallelPlan(loopDef, parFirstInclusiveDim, parLastExclusiveDim,
          // No flag or a rank-0 nest (nothing to fuse): a noCollapse plan.
          enableCollapse && !loopDef.empty() ? collapseLastExclusiveDim : -1,
          cost, exclusiveDims) {}

/*static*/ KrnlParallelPlan KrnlParallelPlan::noCollapse(ValueRange loopDef,
    int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
    KrnlParallelCost cost, ArrayRef<int64_t> exclusiveDims) {
  return KrnlParallelPlan(loopDef, parFirstInclusiveDim, parLastExclusiveDim,
      /*collapse last excl*/ -1, cost, exclusiveDims);
}

/*static*/ KrnlParallelPlan KrnlParallelPlan::noLoopRefs(
    int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
    KrnlParallelCost cost, ArrayRef<int64_t> exclusiveDims) {
  KrnlParallelPlan plan(/*loopDef=*/{}, parFirstInclusiveDim,
      parLastExclusiveDim, /*collapse last excl*/ -1, cost, exclusiveDims);
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

Value KrnlParallelPlan::collapseAndSubstitute(
    const KrnlBuilder &createKrnl, KrnlParallelDecision decision) {
  assert(decision.hasParallel() && "expected a decision to apply");
  // Sites may pass fewer refs than bounds; the decision must land inside them.
  assert(decision.firstDim >= 0 &&
         decision.firstDim + decision.numDims <= (int64_t)optLoopDef.size() &&
         "decision must land inside the loop refs it indexes");
  if (!decision.isCollapse())
    return optLoopDef[decision.firstDim];
  // Fuse the group and substitute the fused ref for its entries.
  ValueRange group(
      ArrayRef<Value>(optLoopDef).slice(decision.firstDim, decision.numDims));
  Value fused = createKrnl.collapse(group);
  optLoopDef.erase(optLoopDef.begin() + decision.firstDim,
      optLoopDef.begin() + decision.firstDim + decision.numDims);
  optLoopDef.insert(optLoopDef.begin() + decision.firstDim, fused);
  collapsed = true;
  return fused;
}

int64_t KrnlParallelPlan::findParallelDim(Operation *op, std::string msg,
    ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs) const {
  // Single level only (collapse window -1). Not clamped to the refs: a
  // noLoopRefs plan has none.
  KrnlParallelDecision decision =
      decideKrnlParallel(lbs, ubs, parFirstInclusiveDim, parLastExclusiveDim,
          exclusiveDims, /*collapse last excl*/ -1, cost);
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
  // Clamp the windows to the refs (-1 stays -1).
  int64_t refCount = optLoopDef.size();
  KrnlParallelDecision decision = decideKrnlParallel(lbs, ubs,
      parFirstInclusiveDim, std::min(parLastExclusiveDim, refCount),
      exclusiveDims, std::min(collapseLastExclusiveDim, refCount), cost);
  if (decision.hasParallel())
    createKrnl.parallel(collapseAndSubstitute(createKrnl, decision));
  return reportKrnlParallelDecision(decision, op, msg, lbs, ubs);
}

} // namespace onnx_mlir
