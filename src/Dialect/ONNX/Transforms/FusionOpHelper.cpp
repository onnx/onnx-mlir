/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ FusionOpHelper.cpp - ONNXFusedOp builder base ----------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================

#include "src/Dialect/ONNX/Transforms/FusionOpHelper.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "op-fusion"

using namespace mlir;

namespace onnx_mlir {

namespace {

// -- Part B: absorbing shape-metadata ops into the fused body --------------
//
// In addition to constants, two more op shapes are safe to clone into the
// body rather than thread through as external inputs -- and only these:
// an onnx.Dim of a chain-produced value, and an i64 ("shape descriptor")
// onnx.Concat that depends on such a Dim. Both are Pure and operate on
// tiny integer tensors, so re-deriving them inside the body is always
// cheap -- unlike absorbing arbitrary computation, which this deliberately
// does not do (the fused body is not always literally re-executed; only
// the unFuse() inlining fallback truly runs it, so anything beyond
// shape/constant ops would risk silently duplicating real compute).
//
// This is intentionally a single-hop check, not a recursive closure walk:
// a Dim's own operand must itself be chain-produced (no deeper), and a
// Concat's operands are checked one level down for such a Dim (no deeper
// than that). Anything requiring more hops than this falls through to
// being an ordinary external input, at which point
// computeInputsAndInsertionPoint()'s general feasibility check is the
// backstop that declines the fusion outright if that input turns out to
// be downstream of a chain-produced value.

bool isDimOfChainProduced(Operation *op, const DenseSet<Value> &chainProduced) {
  auto dimOp = dyn_cast<ONNXDimOp>(op);
  return dimOp && chainProduced.contains(dimOp.getData());
}

// Bounds how much a Concat absorption can ever duplicate. i64 element type
// alone isn't enough: a Concat can be i64-typed and depend on a chain Dim
// while one of its OTHER operands is large real data (e.g. an i64 index
// tensor), not shape metadata -- absorbing that would clone the Concat's
// data movement, not a cheap scalar recompute. A genuine shape descriptor
// (Expand/Reshape's "shape" operand) is bounded by tensor rank -- never
// more than a handful of elements -- so this is a real discriminator, not
// an arbitrary cutoff. Static-shape check first: an unranked/dynamic result
// can't be bounded at all, so it's conservatively rejected. Shared by
// isShapeConcatDependentOnChain() and isAbsorbedPlumbing() so creation and
// retrieval can never disagree on what counts as "small enough."
constexpr int64_t kMaxAbsorbableShapeElements = 8;

bool isSmallI64ShapeTensor(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !shapedType.getElementType().isInteger(64))
    return false;
  return shapedType.hasStaticShape() &&
         shapedType.getNumElements() <= kMaxAbsorbableShapeElements;
}

bool isShapeConcatDependentOnChain(
    Operation *op, const DenseSet<Value> &chainProduced) {
  auto concatOp = dyn_cast<ONNXConcatOp>(op);
  if (!concatOp)
    return false;
  if (!isSmallI64ShapeTensor(concatOp.getConcatResult().getType()))
    return false;
  for (Value operand : concatOp.getInputs()) {
    Operation *operandDef = operand.getDefiningOp();
    if (operandDef && isDimOfChainProduced(operandDef, chainProduced))
      return true;
  }
  return false;
}

// True for anything createFusedOp()/computeInputsAndInsertionPoint() clone
// into the body rather than expose as an external input: constants (as
// before), plus the two shape-metadata shapes above.
bool isAbsorbable(Operation *op, const DenseSet<Value> &chainProduced) {
  return op->hasTrait<mlir::OpTrait::ConstantLike>() ||
         mlir::isa<ONNXNoneOp, ONNXConstantOp>(op) ||
         isDimOfChainProduced(op, chainProduced) ||
         isShapeConcatDependentOnChain(op, chainProduced);
}

// Retrieval-side counterpart of isAbsorbable(): recognizes the same
// body-implementation-detail ops when walking a rebuilt fused body, but
// without needing chainProduced -- once rebuilt, ANY onnx.Dim or small
// i64-shape onnx.Concat is unambiguously absorbed plumbing, never a
// semantic chain member, for every FusionOpKindHelper subclass today
// (their anchor/chain ops are always f32/f16-typed compute ops, never a
// bare Dim or a small-integer-shape Concat). Kept separate from
// isAbsorbable() rather than reconstructing chainProduced from a rebuilt
// body -- but shares isSmallI64ShapeTensor() with
// isShapeConcatDependentOnChain() so the two sides can never disagree on
// what counts as absorbable: creation never clones a large i64 Concat into
// the body, so retrieval never needs to (and must not) treat one as
// plumbing either.
bool isAbsorbedPlumbing(Operation *op) {
  if (op->hasTrait<mlir::OpTrait::ConstantLike>() ||
      mlir::isa<ONNXNoneOp, ONNXConstantOp, ONNXDimOp>(op))
    return true;
  if (auto concatOp = dyn_cast<ONNXConcatOp>(op))
    return isSmallI64ShapeTensor(concatOp.getConcatResult().getType());
  return false;
}

} // namespace

//===----------------------------------------------------------------------===//
// FusionOpKindHelper — non-virtual method implementations
//
// Defined in the same order as declared in the header: the two fusion-pass
// (creation) methods first, in call order, then the two lowering-pass
// methods, in call order, then the shared unFuse() fallback -- then the
// protected/private helpers those public methods are built from.
//===----------------------------------------------------------------------===//

bool FusionOpKindHelper::computeInputsAndInsertionPoint() {
  assert(!ops.empty() &&
         "computeInputsAndInsertionPoint() called with empty ops list");
  fusedInputs.clear();
  insertionAnchor = nullptr;
  insertAfterAnchor = false;

  // Build the set of values produced by the chain ops themselves; these
  // are visible inside the body via the clone mapping and never external.
  DenseSet<Value> chainProduced;
  DenseSet<Operation *> chainOps(ops.begin(), ops.end());
  for (Operation *op : ops)
    for (Value result : op->getResults())
      chainProduced.insert(result);

  // Pre-scan: collect ALL external values the chain ops need. Pure query --
  // no IR mutation -- mirrored by ensureInBody's cloning walk in
  // createFusedOp(), which trusts this list rather than recomputing it.
  //   - isAbsorbable() ops (constants, plus a Dim/shape-Concat depending on
  //     a chain-produced value -- see the Part B comment above) will be
  //     cloned inside the body — they do NOT become block arguments.
  //   - Everything else (non-constant tensors, e.g. dynamically-computed
  //     reshape shape vectors) becomes an additional FusedOp input.
  DenseSet<Value> inputSet;
  std::function<void(Value)> collectExternals = [&](Value v) {
    if (inputSet.contains(v) || chainProduced.contains(v))
      return;
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      // Block argument (e.g. function parameter) — thread through as an input.
      inputSet.insert(v);
      fusedInputs.push_back(v);
      return;
    }
    if (isAbsorbable(defOp, chainProduced)) {
      // Recursively collect the absorbed op's own inputs (e.g. a
      // constant's initializer, or a Dim's data operand) so that they are
      // also cloned inside the body rather than threaded through as
      // inputs. A Dim's data operand, when chain-produced, is skipped by
      // the chainProduced check at the top of this lambda -- it is visible
      // inside the body via the clone mapping already.
      for (Value operand : defOp->getOperands())
        collectExternals(operand);
    } else {
      inputSet.insert(v);
      fusedInputs.push_back(v);
    }
  };
  for (Operation *op : ops)
    for (Value operand : op->getOperands())
      collectExternals(operand);

  // Latest-positioned external input with a defining op in this block.
  // Block arguments (no defining op) dominate everything, so they impose no
  // ordering constraint and are skipped here.
  Operation *latestInputDef = nullptr;
  for (Value v : fusedInputs) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      continue;
    if (!latestInputDef || latestInputDef->isBeforeInBlock(defOp))
      latestInputDef = defOp;
  }

  // Earliest-positioned use of any finalResults value whose owner is not
  // itself part of the chain (i.e. a use that survives past replaceAndErase).
  // ops.back() is the default insertion point (see below); this loop exists
  // to detect when that default is unsafe -- when some non-chain use of a
  // yielded value sits before ops.back()'s original position.
  Operation *earliestUse = nullptr;
  for (Value v : finalResults) {
    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      if (chainOps.contains(owner))
        continue;
      if (!earliestUse || owner->isBeforeInBlock(earliestUse))
        earliestUse = owner;
    }
  }

  // Default: preserve today's placement (immediately before ops.back()) --
  // always safe with respect to inputs (every input dominates whichever
  // chain member consumes it, and every chain member precedes-or-is
  // ops.back()). Only unsafe if some outside use sits before ops.back().
  if (!earliestUse || ops.back()->isBeforeInBlock(earliestUse)) {
    insertionAnchor = ops.back();
    insertAfterAnchor = false;
    return true;
  }

  // ops.back() is unsafe: some outside use of a yielded value is positioned
  // before it. Need the latest point that is still after every input.
  if (!latestInputDef) {
    // No input imposes a lower bound at all, so the earliest point in the
    // chain's own original span is always safe: earliestUse is a genuine use
    // of some chain-produced value, which (in valid pre-existing IR) must
    // already be positioned after whichever chain member produced it, which
    // is at-or-after ops.front().
    insertionAnchor = ops.front();
    insertAfterAnchor = false;
    return true;
  }
  if (latestInputDef->isBeforeInBlock(earliestUse)) {
    insertionAnchor = latestInputDef;
    insertAfterAnchor = true;
    return true;
  }

  // No valid single insertion point exists: some external input this chain
  // needs is only available after some outside use of a yielded value has
  // already occurred. That input is, transitively, downstream of a
  // chain-produced value -- fusing here would make the FusedOp's own input
  // depend on its own (not-yet-defined) output. Decline to fuse.
  // TODO: I believe that we could consider moving ops; ok for now.
  LLVM_DEBUG(
      llvm::dbgs() << "FusionOpKindHelper: no valid insertion point -- an "
                      "external input is only ready after an outside use of a "
                      "yielded chain value; declining to fuse\n");
  return false;
}

ONNXFusedOp FusionOpKindHelper::fuse(PatternRewriter &rewriter, Location loc) {
  assert(!ops.empty() && "fuse() called with empty ops list");
  assert(insertionAnchor &&
         "computeInputsAndInsertionPoint() must be called (and must return "
         "true) before fuse()");
  if (insertAfterAnchor)
    rewriter.setInsertionPointAfter(insertionAnchor);
  else
    rewriter.setInsertionPoint(insertionAnchor);
  ONNXFusedOp fusedOp = create(rewriter, loc);
  replaceAndErase(rewriter, fusedOp);
  return fusedOp;
}

void FusionOpKindHelper::retrieveOpsAndOutputValues(ONNXFusedOp fusedOp) {
  ops.clear();
  finalResults.clear();
  Block &body = fusedOp.getBody().front();
  for (Operation &op : body) {
    if (isa<ONNXYieldOp>(&op)) {
      for (Value v : op.getOperands())
        finalResults.push_back(v);
    } else if (!isAbsorbedPlumbing(&op)) {
      // Constants, Dims, and shape-Concats are body implementation details
      // (cloned from the outer IR by createFusedOp() -- see isAbsorbable()
      // near the top of this file). Exclude them so that ops[] always
      // contains exactly the semantic chain ops — the same set that
      // detectIfBeneficial() collected — making verify() reliable at all
      // times. isAbsorbedPlumbing() must stay in sync with what
      // createFusedOp() actually clones (isAbsorbable()), or an absorbed op
      // would leak into ops[] here and desync every subclass's positional
      // indexing in verify().
      ops.push_back(&op);
    }
  }
}

bool FusionOpKindHelper::verifyAndRetrieveAttrs(ONNXFusedOp fusedOp) {
  if (!retrieveAttrs(fusedOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "FusionOpKindHelper: retrieveAttrs failed for kind '"
               << fusedOp.getKind() << "'\n");
    return false;
  }
  if (!verify()) {
    LLVM_DEBUG(llvm::dbgs() << "FusionOpKindHelper: verify failed for kind '"
                            << fusedOp.getKind() << "'\n");
    return false;
  }
  return true;
}

LogicalResult FusionOpKindHelper::unFuse(
    PatternRewriter &rewriter, ONNXFusedOp fusedOp) {
  LLVM_DEBUG(llvm::dbgs() << "FusionOpKindHelper::unFuse: inlining "
                          << "onnx.Fused (kind='" << fusedOp.getKind()
                          << "') — no dedicated lowering or verify failed\n");
  Block &body = fusedOp.getBody().front();
  auto yieldOp = cast<ONNXYieldOp>(body.getTerminator());
  // Snapshot yield operands before they move during inlining.
  SmallVector<Value> results(yieldOp.getOperands());
  // Inline the body just before the FusedOp.  Pass the original
  // (pre-conversion) FusedOp inputs so that block-argument types match;
  // the rewriter then converts the newly exposed ops in the same pass.
  rewriter.inlineBlockBefore(&body, fusedOp, fusedOp.getInputs());
  rewriter.eraseOp(yieldOp);
  rewriter.replaceOp(fusedOp, results);
  return success();
}

bool FusionOpKindHelper::isInsideFusedOp(Operation *op) {
  return mlir::isa<ONNXFusedOp>(op->getParentOp());
}

ONNXFusedOp FusionOpKindHelper::create(
    PatternRewriter &rewriter, Location loc) {
  ONNXFusedOp fusedOp = createFusedOp(rewriter, loc, getKind());
  embedAttrs(fusedOp);
  return fusedOp;
}

ONNXFusedOp FusionOpKindHelper::createFusedOp(
    PatternRewriter &rewriter, Location loc, StringRef kind) {
  // Build the set of values produced by the chain ops themselves; these
  // are visible inside the body via the clone mapping and never external.
  DenseSet<Value> chainProduced;
  for (Operation *op : ops)
    for (Value result : op->getResults())
      chainProduced.insert(result);

  // fusedInputs was already computed (and used to determine the insertion
  // point) by computeInputsAndInsertionPoint() -- reuse it rather than
  // recomputing, so the inputs actually threaded through are guaranteed
  // identical to the ones the insertion-point feasibility check reasoned
  // about.

  // Build FusedOp with the complete input list.
  SmallVector<Type, 4> outputTypes;
  for (Value v : finalResults)
    outputTypes.push_back(v.getType());
  auto fusedOp = ONNXFusedOp::create(
      rewriter, loc, outputTypes, rewriter.getStringAttr(kind), fusedInputs);

  // Build the isolated body: one block argument per fusedInput.
  SmallVector<Type, 4> argTypes;
  SmallVector<Location> argLocs;
  for (Value v : fusedInputs) {
    argTypes.push_back(v.getType());
    argLocs.push_back(v.getLoc());
  }
  Block *body = rewriter.createBlock(&fusedOp.getBody(), {}, argTypes, argLocs);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);

  // Map every fusedInput to its corresponding block argument.
  IRMapping mapping;
  for (auto [v, arg] : llvm::zip(fusedInputs, body->getArguments()))
    mapping.map(v, arg);

  // Recursively clone absorbable ops (constants, plus a Dim/shape-Concat
  // depending on a chain-produced value -- see the Part B comment near the
  // top of this file) inside the body on demand. A Dim's data operand, when
  // chain-produced, is skipped by the chainProduced check above -- it is
  // already visible inside the body via the clone mapping, populated as
  // each real chain op is cloned by the loop below.
  std::function<void(Value)> ensureInBody = [&](Value v) {
    if (mapping.contains(v) || chainProduced.contains(v))
      return;
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return;
    assert(isAbsorbable(defOp, chainProduced) &&
           "non-absorbable external value not collected in pre-scan");
    for (Value operand : defOp->getOperands())
      ensureInBody(operand);
    rewriter.clone(*defOp, mapping);
  };

  for (Operation *op : ops) {
    for (Value operand : op->getOperands())
      ensureInBody(operand);
    rewriter.clone(*op, mapping);
  }

  // Yield the mapped results of the last chain ops.
  SmallVector<Value> yieldVals;
  for (Value v : finalResults)
    yieldVals.push_back(mapping.lookup(v));
  ONNXYieldOp::create(rewriter, loc, ValueRange(yieldVals));

  return fusedOp;
}

void FusionOpKindHelper::replaceAndErase(
    PatternRewriter &rewriter, ONNXFusedOp fusedOp) {
  DenseMap<Value, unsigned> outputMap;
  for (auto [idx, v] : llvm::enumerate(finalResults))
    outputMap[v] = idx;

  for (int i = (int)ops.size() - 1; i >= 0; --i) {
    auto it = outputMap.find(ops[i]->getResult(0));
    if (it != outputMap.end())
      rewriter.replaceOp(ops[i], fusedOp.getOutputs()[it->second]);
    else
      rewriter.eraseOp(ops[i]);
  }
}

} // namespace onnx_mlir
