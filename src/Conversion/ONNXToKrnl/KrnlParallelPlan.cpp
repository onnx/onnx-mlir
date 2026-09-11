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
// Input: the iteration bounds, a profitability window [parFirst, parLast), an
// optional collapse-safety claim [parFirst, collapseLast), excluded levels, and
// the site's cost. Output: no region, one level, or a group of adjacent levels
// to fuse.
//
// Two properties of a level, needed in different places. It is *parallelizable*
// if it lies in the search window and the site did not exclude it, and
// *fusable* if it additionally lies in the collapse claim and has a
// literal-zero lower bound, which is what krnl.collapse requires. Fusable
// implies parallelizable; the converse fails, and conflating the two costs
// regions.
//
// The frame (decideKrnlParallel) clamps the windows, finds the longest run of
// consecutive fusable levels, then either answers directly or defers to a
// policy:
//
//   STEP 0  No collapse claim, or no run of two fusable levels -- so no fusion
//           is possible. Take the plain single-level search
//           (findSuitableParallelDimension): outermost parallelizable level
//           whose trip count is dynamic or >= minTripCountForParallel.
//
// Otherwise a group is legal and possible, and the policy
// (decideCollapseByCostModel) runs:
//
//   STEP 1  A single *parallelizable* level, trip count *literally* >= the
//           target, under a statically small prefix. No fused space, no index
//           recovery -- so fusability is not required here.
//   STEP 2  Fuse a run of adjacent fusable levels. Enumerate one candidate per
//           (run, start) pair, grow each while its guaranteed trip count falls
//           short of the target and the recovery chain stays amortized, then
//           rank them (see GroupCandidate). Taken unless the winner's trip
//           count is a small literal with no unknown factor.
//   STEP 3  Nothing qualifies: emit no region.
//
// Two invariants live in the frame rather than in a policy, which is why the
// frame is not swappable: flag-off IR is bit-identical because STEP 0 is the
// frame's own code, and -enable-collapse cannot change a site that only asks
// because the frame never reaches a policy when collapseLast < 0. A second
// policy implements STEP 1/2/3 and nothing else.
//
// Trip counts are only partly known at compile time, and an unknown factor is
// never counted toward a target -- trusting one as "large enough" is the defect
// this exists to fix. So every quantity compared here is a TripCountProduct, a
// product of literal factors plus a count of unknown ones, and comparisons over
// it are three-valued (AtLeast).
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
#include <string>

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

// The single-level search, used by STEP 0.
// Return the outermost loop within [firstInclusiveDim, lastExclusiveDim) that
// is not excluded and for which (ub-lb) > minSize. Runtime dimensions are
// assumed to satisfy the size requirement by definition. If found one, it is
// parDim and the function returns true. Otherwise parDim is unchanged.
//
// A valid window is a precondition, not something re-established here: the
// frame clamps it before any caller reaches this.
static bool findSuitableParallelDimension(ArrayRef<IndexExpr> lb,
    ArrayRef<IndexExpr> ub, int64_t firstInclusiveDim, int64_t lastExclusiveDim,
    ArrayRef<int64_t> exclusiveDims, int64_t &parDim, int64_t minSize) {
  assert(lb.size() == ub.size() && "expected identical ranks for lb/ub");
  assert(firstInclusiveDim >= 0 &&
         lastExclusiveDim <= static_cast<int64_t>(lb.size()) &&
         "expected a window already clamped to the bounds");
  for (int64_t i = firstInclusiveDim; i < lastExclusiveDim; ++i) {
    // A level the site refuses to parallelize. Keep scanning: an exclusion
    // costs that level, not the whole region.
    if (llvm::is_contained(exclusiveDims, i))
      continue;
    IndexExpr tripCount = ub[i] - lb[i];
    if (!tripCount.isLiteral()) {
      // Got a dyn dim, assume will be large enough.
      LLVM_DEBUG(llvm::dbgs() << "Pick dim " << i << " because ub is dyn\n");
      parDim = i;
      return true;
    }
    if (tripCount.getLiteral() >= minSize) {
      // Got a literal dim with large enough trip count.
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
// Helpers, shared by the frame and the policy. Per LLVM's coding standard the
// file-local functions below are `static`, and the anonymous namespaces are
// kept down to the three types that need internal linkage and cannot be
// `static`.

// Two questions, deliberately separate, because carrying a region alone and
// being fused with a neighbour have different requirements.

// May level i carry a parallel region by itself? Inside the window the site
// declared and not one it excluded. Nothing about its bounds matters: a
// krnl.parallel on a single level is a label on that loop, and the level keeps
// its own bounds.
//
// Note the window passed here is the *profitability* one, parLastExclusiveDim.
// A site may declare a narrower collapse claim than search window, and levels
// between the two are still individually parallel -- that is what the search
// window means -- they merely cannot be fused.
static bool isSafeForParallel(int64_t i, int64_t parFirstInclusiveDim,
    int64_t parLastExclusiveDim, ArrayRef<int64_t> exclusiveDims) {
  return i >= parFirstInclusiveDim && i < parLastExclusiveDim &&
         !llvm::is_contained(exclusiveDims, i);
}

// May level i be *fused* with its neighbours? Everything above over the
// collapse claim, plus krnl.collapse's own requirement: it takes each collapsed
// dimension's trip count to be its *upper bound*, so the lower bound must be a
// literal 0. Anything else is rejected when the nest is lowered, which fails
// the compilation, so screening it here turns that into a narrower group or a
// fall back to STEP 0.
//
// No step check: krnl.iterate has no step, the lowering builds every affine.for
// with a step of 1, and a step can only appear via krnl.block, whose results
// KrnlCollapseOp's verifier already refuses as operands. A plan cannot produce
// a non-unit step, so there is nothing to screen.
static bool isSafeForCollapse(int64_t i, ArrayRef<IndexExpr> lbs,
    int64_t parFirstInclusiveDim, int64_t collapseLastExclusiveDim,
    ArrayRef<int64_t> exclusiveDims) {
  // Checked first: what follows indexes lbs with i.
  if (!isSafeForParallel(
          i, parFirstInclusiveDim, collapseLastExclusiveDim, exclusiveDims))
    return false;
  return lbs[i].isLiteral() && lbs[i].getLiteral() == 0;
}

// Nothing bounds the trip counts, so a nest of large literal extents can
// overflow a product. Saturating is not optional: signed overflow is UB, and
// even granting wrapping, a `known` of 0 makes the widest nest read as too
// narrow while a negative one sails through the maxForkCount rejection. A
// saturated value behaves correctly everywhere -- above any threshold it meets,
// and as a divisor it drives the quotient to 0.
static int64_t saturatingMul(int64_t a, int64_t b) {
  assert(a >= 0 && b >= 0 && "expected non-negative trip counts");
  // MulOverflow leaves `r` holding the wrapped product on overflow; only the
  // no-overflow branch may read it.
  int64_t r;
  if (llvm::MulOverflow(a, b, r))
    return std::numeric_limits<int64_t>::max();
  return r;
}

// A product of trip counts: `known` multiplies the literal factors, `dyn`
// counts the unknown ones.
namespace {
struct TripCountProduct {
  int64_t known = 1;
  int64_t dyn = 0;
};
} // namespace

// One trip count as a factor. A literal below zero is degenerate IR -- an upper
// bound under its lower bound -- and reads as an empty loop, a factor of 0.
static int64_t literalTripCountAsFactor(int64_t literal) {
  return std::max<int64_t>(0, literal);
}

// The product over levels [firstIncl, lastExcl), clamped to the bounds given.
static TripCountProduct tripCountProduct(ArrayRef<IndexExpr> lbs,
    ArrayRef<IndexExpr> ubs, int64_t firstIncl, int64_t lastExcl) {
  TripCountProduct prod;
  firstIncl = std::max<int64_t>(firstIncl, 0);
  lastExcl = std::min<int64_t>(lastExcl, lbs.size());
  for (int64_t i = firstIncl; i < lastExcl; ++i) {
    IndexExpr tripCount = ubs[i] - lbs[i];
    if (tripCount.isLiteral())
      prod.known = saturatingMul(
          prod.known, literalTripCountAsFactor(tripCount.getLiteral()));
    else
      ++prod.dyn;
  }
  return prod;
}

// Three-valued "is this product at least `target`?". MAYBE stays distinct from
// NO: a group whose *guaranteed* trip count falls short may still be the best
// option available.
namespace {
enum class AtLeast { NO = 0, MAYBE = 1, YES = 2 };
} // namespace

static AtLeast atLeast(TripCountProduct prod, int64_t target) {
  if (prod.known >= target)
    return AtLeast::YES;
  return prod.dyn > 0 ? AtLeast::MAYBE : AtLeast::NO;
}

// Cost of recovering one collapsed level's original index from the fused one,
// in cycles. Three tiers, set by what the level's extent lets the compiler emit
// for the divide:
//
//   power-of-two literal   a shift and a mask                        2 cycles
//   other literal          a multiply-high, a shift and a fixup      6 cycles
//   unknown                a hardware divide, not strength-reduced  40 cycles
//
// Estimates for a modern out-of-order core. Only the shape matters -- ALU
// cheap, constant divide a few times that, hardware divide an order above again
// -- so being off by a factor of two throughout changes no decision.
static int64_t overheadForIndexRematerializationInCycles(IndexExpr tripCount) {
  if (!tripCount.isLiteral())
    return 40;
  int64_t t = tripCount.getLiteral();
  bool isPowerOfTwo = t > 0 && (t & (t - 1)) == 0;
  return isPowerOfTwo ? 2 : 6;
}

//===----------------------------------------------------------------------===//
// One candidate group [firstDim, firstDim + numDims), and the tuple it is
// ranked by. Lexicographic rather than a weighted sum: the fields are not
// commensurable, and a guaranteed-adequate trip count beats a possible one at
// any price.
//
// A group carries two overheads, and they are separate fields rather than one
// summed cost because they are amortized over different things:
//
//  - Index recovery is paid per *fused iteration*, and what it is spread over
//  --
//    the sequential levels below the group, times bodyCost -- is computable. A
//    genuine ratio, held here as a percentage. This is the *price*.
//  - Entering the region is paid per *region entry*, and the entry count is
//    exactly what is unknown when it matters: the known part of that product is
//    identical whether a candidate absorbs the dynamic level above it or leaves
//    it outside. Not priceable, so it is a *gate*: dynamic entry count, yes/no.
//
// Reintroducing a price for the second needs the *number* of entries, not a
// per-entry cost; maxForkCount covers the case where that number is known.
namespace {
struct GroupCandidate {
  int64_t firstDim = NO_PAR_FOUND; // The group's outermost level.
  int64_t numDims = 0;             // Members, so 1 means no fusion at all.
  // Is the fused trip count at least the target?
  AtLeast tripCountVerdict = AtLeast::NO;
  // The gate: does a dynamic level remain above this group, making the number
  // of region entries unknown? False is better.
  bool hasDynamicForkCount = false;
  // The price: cycles of index recovery per 100 work units of body. Less is
  // better. Scaled by 100 because the bare quotient integer-divides to 0 for
  // any body above a few dozen work units.
  //
  // Not a share of runtime, though it coincides with one when a work unit costs
  // about a cycle -- true for the scalar bodies the tiers were calibrated on,
  // and optimistic by roughly the vector width for a bulk copy. See bodyCost.
  int64_t rematOverhead = 0;
  // Unknown trip counts among the fused levels; more is better.
  int64_t dynTripCounts = 0;
  bool isValid() const { return numDims > 0; }

  // Wider verdict, then a known entry count, then cheaper recovery, then more
  // dynamic members, then fewer members, then a shallower start.
  //
  // The fourth field looks backwards and is not: among candidates tied above, a
  // group covering *more* dynamic levels is the one that stays wide when those
  // levels turn out to be 1 at run time.
  //
  // Not operator<: the early-outs are a "nothing beats nothing" rule, not the
  // strict weak ordering an operator would advertise to std::sort.
  bool isBetterThan(const GroupCandidate &other) const {
    if (!isValid())
      return false; // An invalid candidate is never better than anything.
    if (!other.isValid())
      return true; // ...and anything valid is better than an invalid one.
    if (tripCountVerdict != other.tripCountVerdict)
      return tripCountVerdict > other.tripCountVerdict;
    if (hasDynamicForkCount != other.hasDynamicForkCount)
      return !hasDynamicForkCount;
    if (rematOverhead != other.rematOverhead)
      return rematOverhead < other.rematOverhead;
    if (dynTripCounts != other.dynTripCounts)
      return dynTripCounts > other.dynTripCounts;
    // Fewer members means a shorter recovery chain, the same thing twice.
    if (numDims != other.numDims)
      return numDims < other.numDims;
    return firstDim < other.firstDim;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// The policy: STEP 1, 2 and 3.
//
// Called only on the case the frame cannot answer: windows clamped, and at
// least one run of two adjacent safe levels, so a group is legal and possible.
// Sole reader of the target tuning, which is what leaves STEP 0 independent of
// whether a target has been set.
static KrnlParallelDecision decideCollapseByCostModel(ArrayRef<IndexExpr> lbs,
    ArrayRef<IndexExpr> ubs, int64_t parFirstInclusiveDim,
    int64_t parLastExclusiveDim, ArrayRef<int64_t> exclusiveDims,
    int64_t collapseLastExclusiveDim, KrnlParallelCost cost) {
  int64_t rank = lbs.size();
  KrnlParallelDecision decision;
  const ParallelTuning &tune = ParallelMachineSupport::getTuning();
  // Below are predicates / values that are used repeatedly in the loops below.
  // Capture them once, so the code is easier to read and maintain.
  // STEP 1 asks the first, STEP 2 the second. Keeping them apart is what stops
  // a level that merely cannot be fused from being dropped as a single-level
  // candidate too.
  auto canParallelize = [&](int64_t i) {
    return isSafeForParallel(
        i, parFirstInclusiveDim, parLastExclusiveDim, exclusiveDims);
  };
  auto canFuse = [&](int64_t i) {
    return isSafeForCollapse(
        i, lbs, parFirstInclusiveDim, collapseLastExclusiveDim, exclusiveDims);
  };
  // Smallest fused trip count worth a region: the site's own floor, raised by
  // the target's if it has an opinion. The target's defaults to 0, so a trip
  // count the plain search accepts still clears the bar.
  int64_t tripCountTarget =
      std::max(cost.minTripCountForParallel, tune.minParTripCountFloor);
  // The work one iteration of a group ending at `g` does, in bodyCost's work
  // units: the levels left sequential inside it, times the body. Denominator of
  // both things bodyCost feeds -- the growth guard and the recovery ratio.
  auto amortWork = [&](int64_t g) -> TripCountProduct {
    TripCountProduct prod = tripCountProduct(lbs, ubs, g, rank);
    prod.known =
        saturatingMul(prod.known, literalTripCountAsFactor(cost.bodyCost));
    return prod;
  };

  // STEP 1: a single level, so no fused space and no index recovery. Stricter
  // than STEP 0 in two ways: the trip count must be *literally* wide (trusting
  // a dynamic extent is the defect being fixed, and it breaks on a level
  // swinging between 1 and 4096 across inference phases), and the prefix above
  // it must be statically small (a wide level entered an unknown number of
  // times is not a good answer). Being stricter loses nothing because STEP 2
  // follows: what this refuses, STEP 2 either rescues as a group or returns as
  // a one-member group.
  //
  // Searched over the profitability window, filtered only by what makes a level
  // parallelizable. Deliberately *not* filtered by fusability: this step emits
  // no krnl.collapse, so a level that cannot be fused is still a perfectly good
  // single-level answer and must stay a candidate here.
  for (int64_t i = parFirstInclusiveDim; i < parLastExclusiveDim; ++i) {
    if (!canParallelize(i))
      continue;
    IndexExpr tripCount = ubs[i] - lbs[i];
    if (!tripCount.isLiteral() || tripCount.getLiteral() < tripCountTarget)
      continue;
    // Number of times this region is entered (level i).
    TripCountProduct forkCount = tripCountProduct(lbs, ubs, 0, i);
    // Called too many times, not an interesting solution.
    if (forkCount.dyn > 0 || forkCount.known > tune.maxForkCount)
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "Collapse STEP 1: single dim " << i
               << " with literal trip count " << tripCount.getLiteral()
               << " >= " << tripCountTarget << " under a static prefix of "
               << forkCount.known << "\n");
    decision.firstDim = i;
    decision.numDims = 1;
    return decision;
  }

  // STEP 2: fuse a run of adjacent safe levels, one candidate per (run, start).
  GroupCandidate best;
  int64_t runStart = parFirstInclusiveDim;
  while (runStart < collapseLastExclusiveDim) {
    if (!canFuse(runStart)) {
      ++runStart;
      continue;
    }
    int64_t runEnd = runStart;
    while (runEnd < collapseLastExclusiveDim && canFuse(runEnd))
      ++runEnd;
    // [runStart, runEnd) is a maximal run. Every level in it is a candidate
    // start, including levels below the run's head: starting deeper is what
    // turns a dynamic member into the group's *leading* member, where it costs
    // no divisor at all.
    for (int64_t d = runStart; d < runEnd; ++d) {
      IndexExpr firstTripCount = ubs[d] - lbs[d];
      // Never lead with a trip count of 1: it buys no width and costs a
      // recovery level, and the sequential wrapper it leaves behind is free.
      if (firstTripCount.isLiteral() && firstTripCount.getLiteral() == 1)
        continue;
      TripCountProduct forkCount = tripCountProduct(lbs, ubs, 0, d);
      // A statically large prefix enters the region too many times for its
      // depth to be free. An *unresolved* prefix is gated below instead, since
      // absorbing it into the group is the alternative and the two must be
      // compared.
      if (forkCount.dyn == 0 && forkCount.known > tune.maxForkCount)
        continue;
      // Grow while the guaranteed trip count falls short of the target, but
      // never past the point where the recovery chain is known to have lost its
      // amortization. That second guard keeps a group off the innermost level,
      // preserving both the hoist of the recovery arithmetic and the innermost
      // dimension for vectorization. Three-valued so an unknown inner level
      // reads as "unresolved" rather than "inadequate".
      //
      // Two consequences worth knowing. A [batch, 1, seq, seq'] nest cannot
      // reach the group {2,3}: amort(rank) is bodyCost alone, below
      // minAmortWork, which stops growth before the innermost level -- and that
      // group is the worst cell of the model anyway, a per-element dynamic
      // divide. minAmortWork is the knob that reopens it. Second, this guard is
      // the only thing between a site and a one-member group, so an
      // under-declared bodyCost collapses the candidate set to single levels
      // and can pick a deep level under an unknown entry count.
      int64_t g = d + 1;
      while (g < runEnd &&
             tripCountProduct(lbs, ubs, d, g).known < tripCountTarget &&
             atLeast(amortWork(g + 1), tune.minAmortWork) != AtLeast::NO)
        ++g;
      // One index computation per member below the group's head -- the head
      // needs no divisor -- against the work one fused iteration does.
      int64_t rematerializationInCycles = 0;
      for (int64_t k = d + 1; k < g; ++k)
        rematerializationInCycles +=
            overheadForIndexRematerializationInCycles(ubs[k] - lbs[k]);
      TripCountProduct work = tripCountProduct(lbs, ubs, d, g);
      TripCountProduct amortAtG = amortWork(g);
      GroupCandidate cand;
      cand.firstDim = d;
      cand.numDims = g - d;
      cand.tripCountVerdict = atLeast(work, tripCountTarget);
      cand.dynTripCounts = work.dyn;
      cand.hasDynamicForkCount = forkCount.dyn > 0;
      cand.rematOverhead = (100 * rematerializationInCycles) /
                           std::max<int64_t>(1, amortAtG.known);
      LLVM_DEBUG(llvm::dbgs()
                 << "Collapse STEP 2: candidate group [" << d << ", " << g
                 << ") trip count known " << work.known << " dyn " << work.dyn
                 << " verdict " << (int)cand.tripCountVerdict
                 << (cand.hasDynamicForkCount ? " dyn-fork" : " known-fork")
                 << " remat " << rematerializationInCycles << " cyc / "
                 << amortAtG.known << " work = " << cand.rematOverhead
                 << " per 100\n");
      if (cand.isBetterThan(best))
        best = cand;
    }
    runStart = runEnd;
  }
  // A small literal trip count with no unknown factor is not worth a region,
  // whatever it scored on the other fields: fall through to STEP 3.
  if (best.isValid() && best.tripCountVerdict != AtLeast::NO) {
    LLVM_DEBUG(llvm::dbgs() << "Collapse STEP 2: pick group [" << best.firstDim
                            << ", " << best.firstDim + best.numDims << ")\n");
    decision.firstDim = best.firstDim;
    decision.numDims = best.numDims;
    return decision;
  }

  // STEP 3: nothing qualifies, so emit no krnl.parallel.
  LLVM_DEBUG(llvm::dbgs() << "Collapse STEP 3: nothing worth a region\n");
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

  // Make every window valid: a site may legitimately declare one wider than the
  // nest it hands over. Clamping -1 leaves it -1, so "not collapse-eligible"
  // survives. Done here so every policy is handed windows it can index with.
  parFirstInclusiveDim = std::max<int64_t>(parFirstInclusiveDim, 0);
  parLastExclusiveDim = std::min(parLastExclusiveDim, rank);
  collapseLastExclusiveDim = std::min(collapseLastExclusiveDim, rank);

  // A group is a run of *consecutive* safe levels, so the longest such run says
  // whether collapse can produce anything here at all.
  int64_t longestSafeRun = 0;
  for (int64_t i = parFirstInclusiveDim, run = 0; i < collapseLastExclusiveDim;
       ++i) {
    // If safe, increment the run by one; otherwise, reset the run to zero.
    run = isSafeForCollapse(i, lbs, parFirstInclusiveDim,
              collapseLastExclusiveDim, exclusiveDims)
              ? run + 1
              : 0;
    longestSafeRun = std::max(longestSafeRun, run);
  }

  // STEP 0: the plain single-level search. Taken when the site makes no
  // collapse claim (which includes every site while the flag is off) and
  // equally when its claim cannot yield a fusion, since a policy could then
  // only return a one-member group. Quick-exiting keeps such a site
  // bit-identical rather than subject to STEP 1's stricter rules with no STEP 2
  // to make up the difference; LayoutTransform's second site, whose window is
  // one level, is this case.
  if (collapseLastExclusiveDim < 0 || longestSafeRun < 2) {
    KrnlParallelDecision decision;
    int64_t parId = NO_PAR_FOUND;
    if (findSuitableParallelDimension(lbs, ubs, parFirstInclusiveDim,
            parLastExclusiveDim, exclusiveDims, parId,
            cost.minTripCountForParallel)) {
      decision.firstDim = parId;
      decision.numDims = 1;
    }
    return decision;
  }

  // A group is legal and possible, so this is a matter of judgement: hand it to
  // the policy. A second policy would be selected here from a cl::opt in
  // OnnxMlirCommonOptions -- the one category onnx-mlir-opt does not strip, so
  // a lit test can pick one -- with the target proposing the default.
  return decideCollapseByCostModel(lbs, ubs, parFirstInclusiveDim,
      parLastExclusiveDim, exclusiveDims, collapseLastExclusiveDim, cost);
}

//===----------------------------------------------------------------------===//
// KrnlParallelPlan.

// Shared reporting, so the two entry points cannot drift apart.
static int64_t reportKrnlParallelDecision(KrnlParallelDecision decision,
    Operation *op, const std::string &msg, ArrayRef<IndexExpr> lbs,
    ArrayRef<IndexExpr> ubs) {
  if (!decision.hasParallel()) {
    // Trip count 0, the report's "no parallel loop" value, rather than -1: -1
    // is its "runtime only" sentinel and belongs to lines that did parallelize.
    onnxToKrnlParallelReport(
        op, false, -1, 0, "no par dim with enough work in " + msg);
    return NO_PAR_FOUND;
  }
  int64_t parId = decision.firstDim;
  if (!decision.isCollapse()) {
    onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId], msg);
    return parId;
  }
  // A group's region sits at parId but its trip count is the product of the
  // group's, so reporting the single level's own would understate it. Report
  // the product when every member is literal, and -1 (the report's "runtime
  // only" sentinel) as soon as one is not. Deliberately unlike the policy's
  // `known`, which multiplies the literal factors as a lower bound for
  // deciding; this describes what was emitted.
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
  // Name both ends of the group: the report's own loop-level column gives only
  // where the region sits, so a lone "at level 0" leaves the reader to infer
  // the rest of the group from the count. No comma in the comment: the report
  // line is comma-separated and impl::onnxToKrnlParallelReport asserts on one.
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
  // A group is a substitution into the loop list, so claiming collapse-safety
  // without offering refs is a contradiction rather than a no-op.
  assert((collapseLastExclusiveDim < 0 || !optLoopDef.empty()) &&
         "a collapse-eligible plan needs loop refs to fuse");
}

KrnlParallelPlan::KrnlParallelPlan(ValueRange loopDef, bool enableCollapse,
    int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
    int64_t collapseLastExclusiveDim, KrnlParallelCost cost,
    ArrayRef<int64_t> exclusiveDims)
    : KrnlParallelPlan(loopDef, parFirstInclusiveDim, parLastExclusiveDim,
          // Without the flag this is exactly a noCollapse plan.
          enableCollapse ? collapseLastExclusiveDim : -1, cost, exclusiveDims) {
}

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
  // Records *why* the ref list is empty, which is what lets tryCreateParallel
  // reject this kind of plan while tolerating a rank-0 one.
  plan.noLoopRefsByDesign = true;
  return plan;
}

KrnlParallelPlan::~KrnlParallelPlan() {
  // A collapse no krnl.iterate lists among its optimized loops is never
  // gathered, so it is silently dropped and its krnl.parallel names a ref with
  // no loop. Nothing downstream can see that. Plans that never collapsed may go
  // unread: sites that can never collapse assemble their optimized list
  // separately.
  assert((!collapsed || consumed) &&
         "a krnl.collapse was emitted into this plan but its "
         "optimizedLoopDef() was never passed to a krnl.iterate");
}

ValueRange KrnlParallelPlan::optimizedLoopDef() const {
  // No non-empty assert: a rank-0 output gives a rank-0 nest whose list is
  // legitimately empty, reading `krnl.iterate() with ()`. Gather reaches this
  // when outputRank == 0, as do rank-0 Expand and a full Reduction with
  // keepdims=0. tryCreateParallel reads the same emptiness as "no level to
  // parallelize", so those sites need no rank guard.
  consumed = true;
  return ValueRange(optLoopDef);
}

Value KrnlParallelPlan::collapseAndSubstitute(
    const KrnlBuilder &createKrnl, KrnlParallelDecision decision) {
  assert(decision.hasParallel() && "expected a decision to apply");
  // Size equality does not hold: several sites legitimately pass fewer loop
  // refs than bounds, parallelizing only an outer slice of the nest. What must
  // hold is that the decision lands inside the refs it is about to index.
  assert(decision.firstDim >= 0 &&
         decision.firstDim + decision.numDims <= (int64_t)optLoopDef.size() &&
         "decision must land inside the loop refs it indexes");
  if (!decision.isCollapse())
    return optLoopDef[decision.firstDim];
  // Fuse the group into one ref and substitute it for the group's entries, so
  // the following krnl.iterate consumes the fused loop.
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
  // Passes -1 for the collapse window, so single-level always -- see the
  // header.
  //
  // Not clamped to optLoopDef's size the way tryCreateParallel is: nothing here
  // indexes the refs, and a site that only asks may pass bounds for a loop it
  // never defined (a noLoopRefs plan has none), so clamping would narrow the
  // window to zero and answer "no parallelism" to every such site.
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
  // A rank-0 nest has no level to place a region on, so the answer is "no
  // parallelism" rather than an error. Reported rather than returned silently,
  // so the parallel report still accounts for every site it visits.
  if (optLoopDef.empty()) {
    onnxToKrnlParallelReport(op, /*successful*/ false, -1, /*trip count*/ 0,
        "rank-0 nest in " + msg);
    return NO_PAR_FOUND;
  }
  // Lower an over-wide window to what these refs support, as the search lowers
  // it to what the bounds support; without this the decision could name a level
  // outside optLoopDef. Clamping -1 leaves it -1.
  int64_t refCount = optLoopDef.size();
  KrnlParallelDecision decision = decideKrnlParallel(lbs, ubs,
      parFirstInclusiveDim, std::min(parLastExclusiveDim, refCount),
      exclusiveDims, std::min(collapseLastExclusiveDim, refCount), cost);
  if (decision.hasParallel())
    createKrnl.parallel(collapseAndSubstitute(createKrnl, decision));
  return reportKrnlParallelDecision(decision, op, msg, lbs, ubs);
}

} // namespace onnx_mlir
