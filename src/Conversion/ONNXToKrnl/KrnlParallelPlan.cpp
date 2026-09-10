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
// emits, and the plan that carries its answer to the krnl.iterate that consumes
// it. See KrnlParallelPlan.hpp for what a call site declares and why.
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

// Legacy search when not considering collapse.
// Return the outermost loop within [firstInclusiveDim, lastExclusiveDim) for
// which (ub-lb) > minSize. Runtime dimensions are assumed to satisfy the size
// requirement by definition. If found one, it is parDim and the function
// returns true. Otherwise parDim is unchanged.
static bool findSuitableParallelDimension(ArrayRef<IndexExpr> lb,
    ArrayRef<IndexExpr> ub, int64_t firstInclusiveDim, int64_t lastExclusiveDim,
    int64_t &parDim, int64_t minSize) {
  assert(lb.size() == ub.size() && "expected identical ranks for lb/ub");
  if (firstInclusiveDim < 0)
    firstInclusiveDim = 0;
  if (lastExclusiveDim > static_cast<int64_t>(lb.size()))
    lastExclusiveDim = lb.size();
  for (int64_t i = firstInclusiveDim; i < lastExclusiveDim; ++i) {
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
// The decision, in two parts: a fixed frame and a swappable policy.
//
// decideKrnlParallel below is the *frame*. It validates the windows, works out
// which levels are safe to fuse, and answers the two cases that are not a
// matter of judgement at all: a site that makes no collapse-safety claim, and
// one whose claim cannot yield a fusion. Both take STEP 0 -- today's
// single-level search, verbatim. Only past that does it consult a *policy*, and
// the one policy that exists is decideCollapseByCostModel further down.
//
// The split is where two of this design's provable claims live, which is why
// the frame is not itself swappable. Flag-off IR is bit-identical because STEP
// 0 is the frame's own code, and -enable-collapse cannot change a predicate
// site because the frame never reaches a policy when collapseLastExclusiveDim <
// 0. Neither claim is left to a policy to honour. A second policy therefore
// implements STEP 1/2/3 and nothing else, is free to keep its own helpers
// file-local as this one does, and gets a declaration in the header only if it
// lives in another file.
//
//===----------------------------------------------------------------------===//
// Helpers, shared by the frame and today's policy.
//
// Every quantity the policy compares is a *product of trip counts*: the trip
// count of a fused loop, the number of times a region is entered, the work one
// fused iteration covers. Some factors are unknown at compile time, and an
// unknown factor is never counted toward the trip count -- trusting one as
// "large enough" is the defect this whole change exists to fix -- so a product
// cannot be reduced to a single number, and the comparisons over it are
// three-valued rather than boolean.

namespace {

// The levels a site vouches for as individually parallel, and therefore as safe
// to fuse with one another. Shared: the frame needs it to know whether any
// fusion is possible at all, the policy to enumerate candidates.
bool isSafeDim(int64_t i, int64_t parFirstInclusiveDim,
    int64_t collapseLastExclusiveDim, ArrayRef<int64_t> exclusiveDims) {
  return i >= parFirstInclusiveDim && i < collapseLastExclusiveDim &&
         !llvm::is_contained(exclusiveDims, i);
}

// Products range over trip counts nothing has bounded, so a nest of large
// literal extents can overflow the product. Saturating keeps that case honest,
// and is not optional: signed overflow is undefined behaviour, and even
// granting two's-complement wrapping the result is 0 or negative rather than
// merely wrong. Those flip *gates* rather than reordering candidates -- a
// `known` of 0 makes the widest possible nest read as too narrow to
// parallelize, and a negative one sails through the `known > maxForkCount`
// rejection. A saturated value instead does the right thing at every use: it is
// above any threshold it is compared against, and as a divisor it drives the
// quotient to 0.
//
// The ceiling is the type's, not a chosen one: any ceiling well above the
// thresholds behaves identically, so picking a specific number would be
// arbitrary, and the type's max is the only value that isn't.
int64_t saturatingMul(int64_t a, int64_t b) {
  assert(a >= 0 && b >= 0 && "expected non-negative trip counts");
  // MulOverflow leaves `r` holding the wrapped product on overflow; only the
  // no-overflow branch may read it.
  int64_t r;
  if (llvm::MulOverflow(a, b, r))
    return std::numeric_limits<int64_t>::max();
  return r;
}

// A product of trip counts, with the literal factors and the unknown ones kept
// apart: `known` is the product of the literal factors alone and `dyn` counts
// the rest. Keeping them apart is the whole point -- see above.
struct TripCountProduct {
  int64_t known = 1;
  int64_t dyn = 0;
};

// One trip count as a factor. A literal below zero is degenerate IR
// -- an upper bound under its lower bound -- and is read as an empty loop, i.e.
// a factor of 0.
int64_t literalTripCountAsFactor(int64_t literal) {
  return std::max<int64_t>(0, literal);
}

// The product of the trip counts of levels [firstIncl, lastExcl), clamped to
// the bounds actually given.
TripCountProduct tripCountProduct(ArrayRef<IndexExpr> lbs,
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

// Three-valued "is this product at least `target`?". An unknown factor does not
// count toward the target, but it is not nothing either: a group whose
// *guaranteed* trip count falls short may still be the best option, so
// MAYBE has to stay distinguishable from NO rather than being folded into it.
enum class AtLeast { NO = 0, MAYBE = 1, YES = 2 };

AtLeast atLeast(TripCountProduct prod, int64_t target) {
  if (prod.known >= target)
    return AtLeast::YES;
  return prod.dyn > 0 ? AtLeast::MAYBE : AtLeast::NO;
}

// What it costs to rematerialize one collapsed level's original index from the
// fused one, in clock cycles. Three tiers, set by what the level's extent lets
// the compiler emit for the rematerialization divide:
//
//   power-of-two literal extent  a shift and a mask                  2 cycles
//   any other literal extent     a divide strength-reduced to a      6 cycles
//                                multiply-high, a shift and a fixup
//   unknown extent               a hardware divide that cannot be   40 cycles
//                                strength-reduced at all
//
// Stated estimates for a modern out-of-order core, not measurements: the ALU
// tier is two dependent single-cycle ops, the constant-divisor tier is
// dominated by the multiply's latency, and 64-bit integer division lands
// anywhere from ~20 to ~90 cycles depending on target and operand magnitude.
// What the model relies on is the shape -- ALU cheap, constant divide a few
// times that, hardware divide an order of magnitude above again -- so being off
// by a factor of two in the same direction throughout changes no decision.
int64_t overheadForIndexRematerializationInCycles(IndexExpr tripCount) {
  if (!tripCount.isLiteral())
    return 40;
  int64_t t = tripCount.getLiteral();
  bool isPowerOfTwo = t > 0 && (t & (t - 1)) == 0;
  return isPowerOfTwo ? 2 : 6;
}

// One candidate group [firstDim, firstDim + numDims), with the tuple it is
// ranked by. Ranked lexicographically rather than by a weighted sum because the
// fields are not commensurable: a group whose trip count is guaranteed adequate
// beats one that merely might be, at any price.
struct GroupCandidate {
  int64_t firstDim = NO_PAR_FOUND; // The group's outermost level.
  int64_t numDims = 0;             // Members, so 1 means no fusion at all.
  // Is the fused trip count at least the target?
  AtLeast tripCountVerdict = AtLeast::NO;
  // Amortized index-rematerialization chain + fork penalty, in clock cycles;
  // less is better. Integer division, so a chain spread over more elements than
  // it costs cycles reads as 0 -- see where this is computed.
  int64_t costInCycles = 0;
  // Unknown trip counts among the fused levels; more better.
  int64_t dynTripCounts = 0;
  bool isValid() const { return numDims > 0; }

  // The lexicographic order over the fields above: wider verdict first, then
  // cheaper, then more dynamic members, then fewer members, then a shallower
  // start.
  //
  // The third field is the one that looks backwards and is not: among
  // candidates of equal verdict and equal cost, a group covering *more* dynamic
  // levels is preferred, because it is the one that stays wide when those
  // levels turn out to be 1 at run time. That robustness across shapes is the
  // reason the fix is needed at all.
  //
  // Deliberately not operator<: the two early-outs below are a "nothing beats
  // nothing" rule rather than a comparison, so this is not the strict weak
  // ordering an operator would advertise to std::sort and friends.
  bool isBetterThan(const GroupCandidate &other) const {
    if (!isValid())
      return false; // An invalid candidate is never better than anything.
    if (!other.isValid())
      return true; // ...and anything valid is better than an invalid one.
    if (tripCountVerdict != other.tripCountVerdict)
      return tripCountVerdict > other.tripCountVerdict;
    if (costInCycles != other.costInCycles)
      return costInCycles < other.costInCycles;
    if (dynTripCounts != other.dynTripCounts)
      return dynTripCounts > other.dynTripCounts;
    // Fewer members means a shorter rematerialization chain, the same thing
    // twice.
    if (numDims != other.numDims)
      return numDims < other.numDims;
    return firstDim < other.firstDim;
  }
};

//===----------------------------------------------------------------------===//
// The policy: STEP 1, 2 and 3 of the cost model.
//
// Called by the frame, and only on the case the frame cannot answer by itself:
// the site has claimed [parFirstInclusiveDim, collapseLastExclusiveDim) is
// per-level parallel, every window has been clamped to the bounds, and at least
// one run of two adjacent safe levels exists -- so a group is both legal and
// possible here. A policy neither sees nor needs the cases the frame keeps.
//
// This is the only reader of the target tuning, which is what leaves the frame
// -- and so STEP 0, and so the flag-off path -- independent of whether a target
// has been set at all.
KrnlParallelDecision decideCollapseByCostModel(ArrayRef<IndexExpr> lbs,
    ArrayRef<IndexExpr> ubs, int64_t parFirstInclusiveDim,
    int64_t parLastExclusiveDim, ArrayRef<int64_t> exclusiveDims,
    int64_t collapseLastExclusiveDim, KrnlParallelCost cost) {
  int64_t rank = lbs.size();
  KrnlParallelDecision decision;
  const ParallelTuning &tune = ParallelMachineSupport::getTuning();
  // Is this iter safe for parallelization.
  auto isSafe = [&](int64_t i) {
    return isSafeDim(
        i, parFirstInclusiveDim, collapseLastExclusiveDim, exclusiveDims);
  };
  // One trip-count target, not two. The analysis had a per-target `minParTotal`
  // beside the per-site `minSize`, but they are the same quantity -- the
  // smallest fused trip count worth a region -- so they are merged: the site's
  // own floor, raised by the target's if the target has an opinion. That floor
  // defaults to 0, which is what keeps this change additive, since a trip count
  // today's code accepts then still clears the bar.
  int64_t tripCountTarget =
      std::max(cost.minTripCountForParallel, tune.minParTripCountFloor);
  // The elements one iteration of a group ending at `g` covers: the levels left
  // sequential inside it, times whatever bulk work the body does per iteration.
  TripCountProduct amortWork = [&](int64_t g) {
    TripCountProduct prod = tripCountProduct(lbs, ubs, g, rank);
    prod.known =
        saturatingMul(prod.known, literalTripCountAsFactor(cost.bodyElems));
    return prod;
  };

  // STEP 1: a single level, so no fused space and no index rematerialization.
  // Preferred over a group whenever it is genuinely as good, and stricter than
  // STEP 0 in exactly two ways, both deliberate:
  //
  //  - the trip count must be *literally* wide. Trusting a dynamic extent as
  //    trip count is the defect being fixed; it is what breaks on a level that
  //    swings between 1 and 4096 across inference phases.
  //  - the prefix above it must be statically small. A wide level entered an
  //    unknown number of times is not a good answer, and absorbing that prefix
  //    into a group costs a couple of hoisted ALU ops.
  //
  // Being stricter loses nothing precisely because STEP 2 follows: what STEP 1
  // now refuses, STEP 2 either rescues as a group or returns as the same single
  // level in the form of a one-member group.
  //
  // The search is over the safe levels within [parFirstInclusiveDim,
  // parLastExclusiveDim), i.e. the profitability window, not the wider
  // collapse-safety window. Every site migrated so far sets the two last bounds
  // equal, so the safety filter here is currently a no-op; it is applied anyway
  // so that a site declaring a narrower safety claim than its search window
  // gets the conservative reading rather than a surprise.
  for (int64_t i = parFirstInclusiveDim; i < parLastExclusiveDim; ++i) {
    if (!isSafe(i))
      continue;
    IndexExpr tripCount = ubs[i] - lbs[i];
    if (!tripCount.isLiteral() || tripCount.getLiteral() < tripCountTarget)
      continue;
    // Number of times this region is entered (level i).
    TripCountProduct forkCount = tripCountProduct(lbs, ubs, 0, i);
    // Called to many times, not an interesting solution.
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

  // STEP 2: fuse a run of adjacent safe levels into one region. One candidate
  // per (run, start) pair, ranked by the tuple above -- the alternatives are
  // not comparable on trip count alone.
  GroupCandidate best;
  int64_t runStart = parFirstInclusiveDim;
  while (runStart < collapseLastExclusiveDim) {
    if (!isSafe(runStart)) {
      ++runStart;
      continue;
    }
    int64_t runEnd = runStart;
    while (runEnd < collapseLastExclusiveDim && isSafe(runEnd))
      ++runEnd;
    // [runStart, runEnd) is a maximal run. Every level in it is a candidate
    // start, including levels below the run's own head: starting deeper is what
    // turns a dynamic member into the group's *leading* member, which is where
    // it costs no divisor at all.
    for (int64_t d = runStart; d < runEnd; ++d) {
      IndexExpr firstTripCount = ubs[d] - lbs[d];
      // Never lead with a trip count of 1: it buys no trip count and costs a
      // rematerialization level, and excluding it is free because the
      // sequential wrapper it leaves behind is free.
      if (firstTripCount.isLiteral() && firstTripCount.getLiteral() == 1)
        continue;
      TripCountProduct forkCount = tripCountProduct(lbs, ubs, 0, d);
      // A statically large prefix means the region is entered too many times
      // for its depth to be free. An *unresolved* prefix is priced below
      // instead of being refused: absorbing it into the group is the
      // alternative, and the two have to be compared rather than one of them
      // decreed.
      if (forkCount.dyn == 0 && forkCount.known > tune.maxForkCount)
        continue;
      // Grow while the *guaranteed* trip count falls short of the target, but
      // never past the point where the rematerialization chain is known to have
      // lost its amortization. That second guard is what keeps a group off the
      // innermost level, preserving both the hoist of that arithmetic and the
      // innermost dimension for vectorization. It is three-valued so that an
      // unknown inner level does not read as "inadequate" when it only means
      // "unresolved".
      //
      // A known consequence, recorded rather than worked around: for a
      // [batch, 1, seq, seq'] Slice the analysis' shape census predicts the
      // group {2, 3}, and this guard cannot produce it -- amort(rank) is
      // bodyElems, a literal 1, which is below minAmortWork and stops growth
      // before the innermost level. That group is also the worst cell of the
      // cost model, a per-element dynamic divide, so the guard is the more
      // defensible of the two positions and is what gets implemented here.
      // minAmortWork is the knob that reopens the question, and it wants a
      // measurement rather than another argument.
      int64_t g = d + 1;
      while (g < runEnd &&
             tripCountProduct(lbs, ubs, d, g).known < tripCountTarget &&
             atLeast(amortWork(g + 1), tune.minAmortWork) != AtLeast::NO)
        ++g;
      // One index computation per member below the group's head -- the head's
      // own index needs no divisor -- amortized over the work one fused
      // iteration covers, since charging it per fused iteration overstates it
      // by orders of magnitude.
      //
      // Integer division, deliberately. A chain spread over more elements than
      // it costs cycles amortizes to 0, and 0 is the honest answer: a fraction
      // of a cycle per element is not a difference this model can see, and
      // ranking two such candidates by it would be inventing precision the
      // estimates do not have. Candidates that tie here fall through to the
      // next field, which prefers the one that stays wide across shapes -- a
      // better reason to choose than a rounding artifact.
      int64_t rematerializationInCycles = 0;
      for (int64_t k = d + 1; k < g; ++k)
        rematerializationInCycles +=
            overheadForIndexRematerializationInCycles(ubs[k] - lbs[k]);
      TripCountProduct work = tripCountProduct(lbs, ubs, d, g);
      GroupCandidate cand;
      cand.firstDim = d;
      cand.numDims = g - d;
      cand.tripCountVerdict = atLeast(work, tripCountTarget);
      cand.dynTripCounts = work.dyn;
      cand.costInCycles =
          rematerializationInCycles / std::max<int64_t>(1, amortWork(g).known) +
          (forkCount.dyn > 0 ? tune.forkPenaltyCycles : 0);
      LLVM_DEBUG(llvm::dbgs()
                 << "Collapse STEP 2: candidate group [" << d << ", " << g
                 << ") trip count known " << work.known << " dyn " << work.dyn
                 << " verdict " << (int)cand.tripCountVerdict << " cost "
                 << cand.costInCycles << " cycles\n");
      if (cand.isBetterThan(best))
        best = cand;
    }
    runStart = runEnd;
  }
  // A trip count that is a small literal with no unknown factor is not worth a
  // region, whatever it scored on the other fields: fall through to STEP 3.
  if (best.isValid() && best.tripCountVerdict != AtLeast::NO) {
    LLVM_DEBUG(llvm::dbgs() << "Collapse STEP 2: pick group [" << best.firstDim
                            << ", " << best.firstDim + best.numDims << ")\n");
    decision.firstDim = best.firstDim;
    decision.numDims = best.numDims;
    return decision;
  }

  // STEP 3: nothing qualifies, so emit no krnl.parallel -- today's answer when
  // the search finds nothing, reached here through a different route.
  LLVM_DEBUG(llvm::dbgs() << "Collapse STEP 3: nothing worth a region\n");
  return decision;
}

} // namespace

//===----------------------------------------------------------------------===//
// The frame. See the section banner above for what it keeps and why.

KrnlParallelDecision decideKrnlParallel(ArrayRef<IndexExpr> lbs,
    ArrayRef<IndexExpr> ubs, int64_t parFirstInclusiveDim,
    int64_t parLastExclusiveDim, ArrayRef<int64_t> exclusiveDims,
    int64_t collapseLastExclusiveDim, KrnlParallelCost cost) {
  assert(lbs.size() == ubs.size() && "expected identical ranks for lb/ub");
  int64_t rank = lbs.size();

  // Make every window valid, exactly as findSuitableParallelDimension does for
  // its own: a site may legitimately declare a window wider than the nest it
  // hands over. Clamping -1 leaves it -1, so "not collapse-eligible" survives.
  // Done here rather than in a policy so that every policy is handed windows it
  // can index with.
  parFirstInclusiveDim = std::max<int64_t>(parFirstInclusiveDim, 0);
  parLastExclusiveDim = std::min(parLastExclusiveDim, rank);
  collapseLastExclusiveDim = std::min(collapseLastExclusiveDim, rank);

  // A group is a run of *consecutive* safe levels, so the longest such run is
  // what says whether collapse can produce anything here at all.
  int64_t longestSafeRun = 0;
  for (int64_t i = parFirstInclusiveDim, run = 0; i < collapseLastExclusiveDim;
       ++i) {
    run = isSafeDim(
              i, parFirstInclusiveDim, collapseLastExclusiveDim, exclusiveDims)
              ? run + 1
              : 0;
    longestSafeRun = std::max(longestSafeRun, run);
  }

  // STEP 0: a single level, today's rules exactly. Taken when the site makes no
  // collapse-safety claim (collapseLastExclusiveDim == -1, which includes every
  // site while the flag is off) and equally when it makes one that cannot yield
  // a fusion: with no run of two adjacent safe levels a policy could only ever
  // return a one-member group, which is the identity. Quick-exiting to today's
  // path there is not merely an optimization -- it is what keeps such a site
  // bit-identical rather than subject to STEP 1's stricter rules with no STEP 2
  // able to make up the difference. LayoutTransform's second site, whose window
  // is a single level by construction, is exactly this case.
  //
  // Note the exclusiveDims check deliberately reproduces an oddity of the
  // original: the search returns the first suitable level and the exclusion is
  // tested only afterwards, so an excluded level makes the whole site give up
  // rather than continuing the scan. Reproducing it exactly is what
  // bit-identity requires; a policy uses a properly filtered set of levels
  // instead.
  if (collapseLastExclusiveDim < 0 || longestSafeRun < 2) {
    KrnlParallelDecision decision;
    int64_t parId = NO_PAR_FOUND;
    if (findSuitableParallelDimension(lbs, ubs, parFirstInclusiveDim,
            parLastExclusiveDim, parId, cost.minTripCountForParallel)) {
      if (!llvm::is_contained(exclusiveDims, parId)) {
        decision.firstDim = parId;
        decision.numDims = 1;
      }
    }
    return decision;
  }

  // A group is both legal and possible here, so this is a matter of judgement:
  // hand it to the policy. A second policy is selected here, from a cl::opt in
  // OnnxMlirCommonOptions -- the one category onnx-mlir-opt does not strip, so
  // that a lit test can pick one -- with the target proposing a default the way
  // ParallelMachineSupport already proposes the tuning constants. Not written
  // yet: a switch over one value is noise, and this is the whole of the change
  // when the second one arrives.
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
    onnxToKrnlParallelReport(
        op, false, -1, -1, "no par dim with enough work in " + msg);
    return NO_PAR_FOUND;
  }
  int64_t parId = decision.firstDim;
  if (!decision.isCollapse()) {
    onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId], msg);
    return parId;
  }
  // A collapsed group needs both numbers restated. The region still sits at
  // parId, but its trip count is the product of the group's own, so the
  // single level's own trip count would understate it -- and understating it is
  // exactly what would hide the diagnosis this report exists to confirm.
  //
  // Report the product when every member is literal, and -1 (the report's
  // "runtime only" sentinel) as soon as one is not: a product with an unknown
  // factor is unknown, not partially known. Note this deliberately differs from
  // the heuristic's own `known()`, which multiplies the literal factors only --
  // that is a lower bound used for deciding, whereas this is a description of
  // what was emitted.
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
  // No comma in the comment: the report line is comma-separated and
  // impl::onnxToKrnlParallelReport asserts on one.
  onnxToKrnlParallelReport(op, /*successful*/ true, parId,
      allLiteral ? fusedTripCount : -1,
      msg + " with " + std::to_string(decision.numDims) +
          " loops collapsed at level " + std::to_string(parId));
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
  // without offering any refs is a contradiction rather than a no-op.
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
  // keep rejecting this kind of plan while tolerating a rank-0 one.
  plan.noLoopRefsByDesign = true;
  return plan;
}

KrnlParallelPlan::~KrnlParallelPlan() {
  // A collapse that no krnl.iterate lists among its optimized loops is never
  // gathered, so it is silently dropped and its krnl.parallel names a ref that
  // no longer has a loop. Nothing downstream can see this, so it is checked
  // here. Plans that never collapsed are free to go unread: the sites that can
  // never collapse (blocked refs, or a decision over a subset of the iterate's
  // loops) legitimately assemble their optimized list separately.
  assert((!collapsed || consumed) &&
         "a krnl.collapse was emitted into this plan but its "
         "optimizedLoopDef() was never passed to a krnl.iterate");
}

ValueRange KrnlParallelPlan::optimizedLoopDef() const {
  // Deliberately no non-empty assert here. A rank-0 output gives a rank-0 nest,
  // whose optimized-loop list is legitimately empty and whose krnl.iterate
  // reads `krnl.iterate() with ()`. Gather reaches this whenever outputRank ==
  // dataRank + indicesRank - 1 == 0, which
  // test/mlir/conversion/onnx_to_krnl/Tensor/Gather.mlir already covers; so do
  // rank-0 Expand and a full Reduction with keepdims=0. tryCreateParallel
  // treats the same emptiness as "no level to parallelize" and returns
  // NO_PAR_FOUND, so those sites need no rank guard before calling it.
  consumed = true;
  return ValueRange(optLoopDef);
}

Value KrnlParallelPlan::collapseAndSubstitute(
    const KrnlBuilder &createKrnl, KrnlParallelDecision decision) {
  assert(decision.hasParallel() && "expected a decision to apply");
  // The size equality that would be natural here does not hold: several sites
  // legitimately pass fewer loop refs than bounds (parallelizing only an outer
  // slice of the nest). What must hold is that the decision lands inside the
  // refs it is about to index.
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
  // Single-level always, even on a collapse-eligible plan: see the header for
  // why a group would be meaningless to the sites that only ask.
  //
  // Deliberately *not* clamped to optLoopDef's size the way tryCreateParallel
  // is. Nothing here indexes the refs, and a site that only asks may hand in
  // bounds for a loop it never defined -- a noLoopRefs plan has no refs at all
  // -- so clamping would narrow the window to zero and answer "no parallelism"
  // to every such site.
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
  // A rank-0 nest has no level a region could be placed on, so the answer is
  // "no parallelism" rather than an error, and no site needs a rank guard of
  // its own before calling. Reported rather than returned silently, so the
  // parallel report still accounts for every site it visits.
  if (optLoopDef.empty()) {
    onnxToKrnlParallelReport(
        op, /*successful*/ false, -1, -1, "rank-0 nest in " + msg);
    return NO_PAR_FOUND;
  }
  // Lower an over-wide window to what these refs can support, exactly as
  // findSuitableParallelDimension lowers it to what the bounds can support: a
  // site may legitimately declare a window wider than the slice of the nest it
  // handed us, and the two limits should forgive equally. Without this the
  // decision could name a level outside optLoopDef, which collapseAndSubstitute
  // can only assert on. Clamping -1 leaves it -1, so a non-collapse-eligible
  // plan stays non-collapse-eligible.
  int64_t refCount = optLoopDef.size();
  KrnlParallelDecision decision = decideKrnlParallel(lbs, ubs,
      parFirstInclusiveDim, std::min(parLastExclusiveDim, refCount),
      exclusiveDims, std::min(collapseLastExclusiveDim, refCount), cost);
  if (decision.hasParallel())
    createKrnl.parallel(collapseAndSubstitute(createKrnl, decision));
  return reportKrnlParallelDecision(decision, op, msg, lbs, ubs);
}

} // namespace onnx_mlir
