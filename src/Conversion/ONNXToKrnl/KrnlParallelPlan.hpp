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
// Every ONNX-to-Krnl site that wants a region asks the same two questions -- is
// one worth it here, and over which loop level -- so the decision procedure and
// the two entry points onto it live here. A region may span several adjacent
// levels fused into one by krnl.collapse, which makes the answer a *group* of
// levels; KrnlParallelPlan carries that answer to the following krnl.iterate.
//
// Included from ONNXToKrnlCommon.hpp, so every lowering site sees it.
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

// "No loop level was found worth parallelizing." Returned by findParallelDim
// and tryCreateParallel, and the value of KrnlParallelDecision::firstDim when
// there is no decision.
//
// Not reused for the other -1s here, which mean different things:
// `collapseLastExclusiveDim == -1` is "this site makes no collapse-safety
// claim", and the -1s handed to onnxToKrnlParallelReport are that report's own
// "unknown at compile time" sentinel.
static constexpr int64_t NO_PAR_FOUND = -1;

// The two numbers only the call site knows. Kept apart from the plan's bounds:
// those say *which loop refs*, these say *how much work*. Neither indexes into
// the loop list.
//
// Hazard: this and the `exclusiveDims` parameter that follows it at every
// factory are both constructible from `{someInt}`, so a site passing exclusions
// while omitting the cost binds its exclusion list to the cost and compiles
// clean. Keep `cost` ahead of `exclusiveDims` and keep both labelled.
struct KrnlParallelCost {
  // This site's floor on the trip count worth a parallel region.
  int64_t minTripCountForParallel = 4;
  // Roughly how much work one iteration of this nest's innermost loop does, in
  // *work units*: one scalar operation, or one element copied, counts as 1. An
  // order of magnitude read off the body:
  //
  //      1   a copy: a load and a store
  //     10   a handful of arithmetic ops, an index computation, a compare
  //   1000   a transcendental, an interpolation kernel, a whole tile
  //
  // One digit is enough: it feeds only how far a group may grow and the
  // denominator of the index-recovery ratio, and both ask whether the body is
  // big enough to hide an integer divide.
  //
  // Work units and not cycles, deliberately. Cycles would be the ideal unit --
  // the recovery chain it is weighed against is measured in them -- but pricing
  // a bulk copy in cycles needs a bytes-per-cycle or vector-width constant that
  // no target here has measured, and guessing it *low* is the unsafe direction
  // (see below). So a copied element counts as 1 even though vectorization
  // makes it cheaper than a scalar op, which means the ratio in GroupCandidate
  // understates the true fraction of runtime for bulk-copy bodies, by roughly
  // the vector width. minAmortWork is calibrated in these same units.
  //
  // A body that loops or copies over N elements multiplies through: a
  // 128-element memcpy is ~128, an scf loop of N iterations around ten ops is
  // ~10N. Any site whose innermost krnl iteration hides a memcpy, an scf loop
  // or a SIMD span **must** set this -- under-declaring it stops a group from
  // growing, which changes the candidate set rather than merely narrowing the
  // winner, and can pick an answer worse than no collapse at all.
  //
  // Cannot be inferred: when the decision runs the body is not built yet.
  int64_t bodyCost = 1;
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

// Everything a site knows about *which* loops a parallel decision may range
// over, plus the two entry points that act on it. Every site builds one, so the
// decision is always made through the same object.
//
// It also carries the optimized-loop list from the decision to the krnl.iterate
// that consumes it: krnl.parallel is a label on an existing ref, but
// krnl.collapse *substitutes* refs, so a site that may collapse must hand
// optimizedLoopDef() to its iterate.
class KrnlParallelPlan {
public:
  // Collapse-eligible when enableCollapse is set; otherwise exactly a
  // noCollapse plan. Everything a site declares arrives here in one statement:
  // which loop refs, which levels may be searched, which may be fused, which
  // are excluded, and how much work there is. Only the iteration bounds stay a
  // call argument, since a site that merely asks may pass bounds for a loop it
  // never defined.
  //
  // For every constructor, first dim is max-ed with 0 and last dim min-ed with
  // the size of the bounds, so the search window is always valid.
  //
  // `cost` has no default here while the factories below keep theirs: bodyCost
  // is read only when a group is possible, so at a noCollapse or noLoopRefs
  // site it cannot be read, and at a collapse-eligible one its default can
  // produce an answer worse than no collapse (see KrnlParallelCost). Write it
  // with designated initializers, as the collapse sites do; that also makes a
  // stray `{someInt}` in this slot visible.
  KrnlParallelPlan(mlir::ValueRange loopDef, bool enableCollapse,
      int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
      int64_t collapseLastExclusiveDim, KrnlParallelCost cost,
      mlir::ArrayRef<int64_t> exclusiveDims = {});

  // *Never* collapse-eligible, whatever the flag says: blocked or permuted
  // refs, iterArgs, or a decision made over a subset of the iterate's loops.
  // Named so the set of sites not yet migrated is greppable. Still takes the
  // search window and exclusions, since the single-level search runs
  // regardless.
  //
  // The window has no default here or below: which levels a site may
  // parallelize is a statement about that nest, not something to inherit.
  static KrnlParallelPlan noCollapse(mlir::ValueRange loopDef,
      int64_t parFirstInclusiveDim, int64_t parLastExclusiveDim,
      KrnlParallelCost cost = {}, mlir::ArrayRef<int64_t> exclusiveDims = {});

  // For a site that asks the question but emits its own parallelism through
  // forLoopIE/forLoopsIE's useParallel flag, and so has no krnl.iterate
  // optimized-loop list at all. No refs to fuse, hence never collapse-eligible.
  // Use with findParallelDim; tryCreateParallel has nothing to emit onto.
  static KrnlParallelPlan noLoopRefs(int64_t parFirstInclusiveDim,
      int64_t parLastExclusiveDim, KrnlParallelCost cost = {},
      mlir::ArrayRef<int64_t> exclusiveDims = {});

  // Asserts that a plan which emitted a krnl.collapse was actually consumed. A
  // collapse absent from every iterate's optimized-loop list is never gathered,
  // so it never runs, and nothing downstream can see that.
  ~KrnlParallelPlan();

  //===--------------------------------------------------------------------===//
  // The two entry points.

  // Ask only: "is a region here worth it, and over which level?" Returns the
  // level or NO_PAR_FOUND. Emits nothing, mutates nothing.
  //
  // Single-level even on a collapse-eligible plan: the sites that ask hand the
  // answer to forLoopIE/forLoopsIE's useParallel, which parallelizes one level
  // and cannot express a fused iteration space, and several size per-thread
  // buffers off it.
  int64_t findParallelDim(mlir::Operation *op, std::string msg,
      mlir::ArrayRef<IndexExpr> lbs, mlir::ArrayRef<IndexExpr> ubs) const;

  // Decide and emit: a krnl.collapse when the decision is a group, then the
  // krnl.parallel. Returns the level, or NO_PAR_FOUND having emitted nothing.
  // Substitutes any fused ref into this plan's loop list, so the following
  // krnl.iterate must be handed optimizedLoopDef().
  //
  // Needs no rank guard at the call site: a rank-0 nest answers NO_PAR_FOUND,
  // and a parLastExclusiveDim wider than this plan's refs is lowered to what
  // the refs support.
  int64_t tryCreateParallel(const onnx_mlir::KrnlBuilder &createKrnl,
      mlir::Operation *op, std::string msg, mlir::ArrayRef<IndexExpr> lbs,
      mlir::ArrayRef<IndexExpr> ubs);

  // The optimized-loop list, to be passed to whatever builds the krnl.iterate.
  // Reading it marks the plan consumed.
  mlir::ValueRange optimizedLoopDef() const;

  int64_t getNumLoopRefs() const { return optLoopDef.size(); }

private:
  // Apply a decision to the loop list and return the ref to parallelize: the
  // level itself for a single-level decision, or the result of a krnl.collapse
  // replacing the group's entries. A group of one is the identity, emitting
  // nothing.
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
  // Set only by the noLoopRefs factory, so that tryCreateParallel can reject
  // such a plan while still tolerating a rank-0 one, whose empty optLoopDef is
  // an answer rather than a mistake.
  bool noLoopRefsByDesign = false;
};

} // namespace onnx_mlir
#endif
