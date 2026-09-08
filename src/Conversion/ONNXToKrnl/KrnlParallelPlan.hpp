/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlParallelPlan.hpp - Parallel region decision procedure -----===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Deciding where a parallel region goes, and emitting it.
//
// Every ONNX-to-Krnl site that wants a parallel region asks the same two
// questions -- is one worth it here, and over which loop level -- so both the
// decision procedure and the two entry points onto it live here rather than in
// the general lowering support. A region may span several adjacent levels fused
// into one by krnl.collapse, which makes the answer a *group* of levels and the
// handoff to the following krnl.iterate a two-call protocol; KrnlParallelPlan
// is what carries that handoff, and what a site uses to declare which of its
// levels may be searched and which may be fused.
//
// Included from ONNXToKrnlCommon.hpp, so every lowering site sees this without
// an include of its own.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_KRNL_PARALLEL_PLAN_H
#define ONNX_MLIR_KRNL_PARALLEL_PLAN_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"

#include <string>

namespace onnx_mlir {

// "No loop level was found worth parallelizing." Returned by
// KrnlParallelPlan::findParallelDim and ::tryCreateParallel, and the value of
// KrnlParallelDecision::firstDim when there is no decision. A level is always a
// valid index into the bounds, so any negative value would do; the point of the
// name is that a call site reads the answer rather than the encoding.
//
// Deliberately not reused for the several other -1s in this area, which mean
// different things: `collapseLastExclusiveDim == -1` is "this site makes no
// collapse-safety claim", and the -1s handed to onnxToKrnlParallelReport /
// onnxToKrnlSimdReport are that report's own "unknown at compile time"
// sentinel.
static constexpr int64_t NO_PAR_FOUND = -1;

// Return the outermost loop within [firstDim, lastDim) for which (ub-lb) >=
// minSize. Runtime dimensions are assumed to satisfy the size requirement by
// definition. If found one, it is parDim and the function returns true.
bool findSuitableParallelDimension(mlir::ArrayRef<IndexExpr> lb,
    mlir::ArrayRef<IndexExpr> ub, int64_t firstInclusiveDim,
    int64_t lastExclusiveDim, int64_t &parDim, int64_t minSize = 4);

// The two numbers only the call site knows: the width worth a parallel region,
// and how much work one innermost iteration covers. Kept together because both
// feed the same width target, and kept apart from KrnlParallelPlan's window
// bounds because those say *which loop refs* while these say *how much work* --
// an orthogonal question with an orthogonal owner. Neither number indexes into
// the loop list, so neither has any reason to travel with it.
//
// This is the single home for both. In particular it is where the predicate and
// the emitter agree: a predicate has no plan to hang anything off, so putting
// either number on the plan would make the two entry points take their cost
// differently for no gain.
// Note for anyone changing the plan factories' parameter order: this and the
// `exclusiveDims` that follows it are both constructible from `{someInt}`, so a
// site that passes exclusions while omitting the cost binds its exclusion list
// to the cost and compiles clean. That happened once, at Concat, where a
// `/*excl dims*/ {axis}` argument silently became a
// `minTripCountForParallel` of `axis`. Keep `cost` ahead of `exclusiveDims`,
// and keep both labelled at every call site.
struct KrnlParallelCost {
  // This site's floor on the trip count worth a parallel region. Was the
  // `minSize` argument.
  int64_t minTripCountForParallel = 4;
  // Elements one iteration of the innermost surviving loop covers, e.g. a
  // memcpy length or a tile width. Cannot be inferred instead of declared:
  // when the decision runs the body has not been built yet, so this exists
  // only in the caller's hand.
  int64_t bodyElems = 1;
};

// What the decision procedure concluded: no parallelism, one level, or a group
// of adjacent levels to be fused into a single region.
struct KrnlParallelDecision {
  // NO_PAR_FOUND <=> !hasParallel(); otherwise the level, or a group's
  // outermost level.
  int64_t firstDim = NO_PAR_FOUND;
  // 0 none, 1 a single level (no collapse), >1 a collapsed group.
  int64_t numDims = 0;
  bool hasParallel() const { return numDims > 0; }
  bool isCollapse() const { return numDims > 1; }
};

// Shared decision core: pure, emits nothing, needs no builder.
//
// [parFirstInclusiveDim, parLastExclusiveDim) is the search window for a plain
// (non-collapsed) parallel level -- a profitability bound, not a safety claim.
// collapseLastExclusiveDim is the stronger, explicit claim that every level in
// [parFirstInclusiveDim, collapseLastExclusiveDim) is individually parallel and
// therefore safe to fuse; -1 means the site makes no such claim, and only the
// single-level search runs.
KrnlParallelDecision decideKrnlParallel(mlir::ArrayRef<IndexExpr> lbs,
    mlir::ArrayRef<IndexExpr> ubs, int64_t parFirstInclusiveDim,
    int64_t parLastExclusiveDim, mlir::ArrayRef<int64_t> exclusiveDims,
    int64_t collapseLastExclusiveDim, KrnlParallelCost cost);

// Everything a site knows about *which* loops a parallel decision may range
// over, plus the two entry points that act on it. Every site builds one,
// whether or not it can collapse and whether or not it emits, so the decision
// is always made through the same object.
//
// It also carries the optimized-loop list from the decision to the krnl.iterate
// that consumes it. krnl.parallel is a label on an existing ref, so tagging it
// is fire-and-forget; krnl.collapse *substitutes* refs, so the decision has to
// be applied to the list the following iterate consumes -- a two-call protocol
// where there was none. Making the plan the receiver keeps that handoff a
// single token, and makes it plain that tryCreateParallel mutates it.
class KrnlParallelPlan {
public:
  // Collapse-eligible when enableCollapse is set; degrades to exactly the
  // noCollapse state when it is not, so a call site reads the same either way.
  //
  // Everything a site declares about its own parallel decision arrives here, in
  // one statement: which loop refs, which levels may be searched, which may be
  // fused, which are excluded, and how much work there is. Only the iteration
  // bounds stay a call argument, because they are the actual space rather than
  // a declaration about it -- and because a site that only asks may hand in
  // bounds belonging to a loop it never defined. See decideKrnlParallel for
  // what the two windows mean.
  //
  // For all constructors (incl noCollapse and noLoopRefs), first dim will be
  // max-ed with 0 and last dim will be min-ed with the size of the lower/upper
  // bounds, so that the search window is always valid.
  KrnlParallelPlan(mlir::ValueRange loopDef, bool enableCollapse,
      int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
      int64_t collapseLastExclusiveDim, KrnlParallelCost cost = {},
      mlir::ArrayRef<int64_t> exclusiveDims = {});

  // *Never* collapse-eligible, whatever the flag says: blocked or permuted
  // refs, iterArgs, or a decision made over a subset of the iterate's loops. A
  // distinct statement from `enableCollapse == false`, and named so the set of
  // sites not yet migrated is greppable rather than inferred from the absence
  // of an argument. Still takes the search window and exclusions, since the
  // single-level search runs whether or not collapse is possible.
  //
  // The search window is spelled out here as it is in the constructor: which
  // levels a site may parallelize is a statement about that loop nest, and a
  // default would let a site inherit `[0, 2)` without ever having considered
  // whether its own nest deserves it.
  static KrnlParallelPlan noCollapse(mlir::ValueRange loopDef,
      int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
      KrnlParallelCost cost = {}, mlir::ArrayRef<int64_t> exclusiveDims = {});

  // For a site that asks the question but emits its own parallelism, through
  // forLoopIE/forLoopsIE's useParallel flag, and so has no krnl.iterate
  // optimized-loop list at all. There are no refs to fuse, which is why such a
  // site can never collapse -- a fact about the data here, not a convention.
  // Use with findParallelDim; tryCreateParallel has nothing to emit onto.
  // The search window is required here too, for the reason given above.
  static KrnlParallelPlan noLoopRefs(int64_t parFirstInclusiveDim,
      int64_t parLastExclusiveDim, KrnlParallelCost cost = {},
      mlir::ArrayRef<int64_t> exclusiveDims = {});

  // Asserts that a plan which emitted a krnl.collapse was actually consumed.
  // This closes the one hole the KrnlToAffine checks cannot see: a collapse
  // absent from every iterate's optimized-loop list is never gathered, so none
  // of them ever runs. Caught here, at the site that made the mistake.
  ~KrnlParallelPlan();

  //===--------------------------------------------------------------------===//
  // The two entry points.

  // Ask only: "is a parallel region here worth it, and over which level?"
  // Returns the chosen level, or NO_PAR_FOUND. Emits nothing, mutates nothing.
  //
  // Deliberately single-level even on a collapse-eligible plan. The sites that
  // ask rather than emit hand the answer to forLoopIE/forLoopsIE's useParallel,
  // which parallelizes one level and cannot express a fused iteration space,
  // and several of them size per-thread reduction buffers off it. A group here
  // would be justified by a width that never materializes.
  int64_t findParallelDim(mlir::Operation *op, std::string msg,
      mlir::ArrayRef<IndexExpr> lbs, mlir::ArrayRef<IndexExpr> ubs) const;

  // Decide and emit: a krnl.collapse when the decision is a group, then the
  // krnl.parallel. Returns the chosen level, or NO_PAR_FOUND having emitted
  // nothing.
  // Substitutes any fused ref into this plan's loop list, so the following
  // krnl.iterate must be handed optimizedLoopDef().
  //
  // A rank-0 nest yields NO_PAR_FOUND rather than an error: there is no level
  // to parallelize, which is an answer and not a mistake. Sites therefore need
  // no rank guard of their own around this call. Likewise a parLastExclusiveDim
  // wider than this plan's loop refs is lowered to what the refs support, the
  // same way the search lowers it to what the bounds support.
  int64_t tryCreateParallel(const onnx_mlir::KrnlBuilder &createKrnl,
      mlir::Operation *op, std::string msg, mlir::ArrayRef<IndexExpr> lbs,
      mlir::ArrayRef<IndexExpr> ubs);

  // The optimized-loop list, to be passed explicitly to whatever builds the
  // krnl.iterate. Reading it marks the plan consumed.
  mlir::ValueRange optimizedLoopDef() const;

  int64_t getNumLoopRefs() const { return optLoopDef.size(); }

private:
  // Apply a decision to the loop list and return the ref to parallelize: the
  // level itself for a single-level decision, or, for a group, the result of a
  // krnl.collapse that replaces the group's entries. A group of one is the
  // identity and emits nothing.
  mlir::Value collapseAndSubstitute(
      const onnx_mlir::KrnlBuilder &createKrnl, KrnlParallelDecision decision);

  KrnlParallelPlan(mlir::ValueRange loopDef, int64_t parFirstInclusiveDim,
      int64_t parLastExclusiveDim, int64_t collapseLastExclusiveDim,
      KrnlParallelCost cost, mlir::ArrayRef<int64_t> exclusiveDims);

  llvm::SmallVector<mlir::Value, 6> optLoopDef;
  int64_t parFirstInclusiveDim;
  int64_t parLastExclusiveDim;
  int64_t collapseLastExclusiveDim; // -1 when not collapse-eligible.
  KrnlParallelCost cost;
  llvm::SmallVector<int64_t, 2> exclusiveDims;
  bool collapsed = false;
  mutable bool consumed = false;
  // Set only by the noLoopRefs factory. Such a site has no krnl.iterate
  // optimized-loop list at all, so calling tryCreateParallel on it is a
  // site-authoring error worth catching. Deliberately not the same question as
  // "is optLoopDef empty": a rank-0 nest legitimately gives a plan built from
  // real refs an empty list, and that is an answer rather than a mistake.
  bool noLoopRefsByDesign = false;
};

} // namespace onnx_mlir
#endif
