/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ONNXFusionOpHelper.cpp - ONNXFusedOp CPU kinds ---------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================

#include "src/Dialect/ONNX/Transforms/ONNXFusionOpHelper.hpp"

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

// Elementwise.hpp is re-included below (with X-macros defined) to reuse its
// canonical list of elementwise op types. That trick only re-expands
// Elementwise.hpp's own repeatable list section; everything it and
// ONNXToKrnlCommon.hpp pull in at namespace scope (mlir/Pass/Pass.h, etc.)
// must already be fully included -- with real header guards tripped -- at
// this file's top level first, or those namespace-scope declarations would
// be re-emitted from inside a function body below and fail to compile.
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "op-fusion"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Per-half allow-list: every ONNX op with the standard elementwise Krnl
// lowering (i.e. an emitScalarOpFor<T>, used identically for scalar and SIMD
// operands -- see Elementwise.hpp's own STEP 1/2 comments), reusing the same
// canonical op-type list `FusionOpStickUnstick.cpp`'s
// `canOpFuseWithStickUnstick` pulls in via this file's X-macro trick, rather
// than hand-picking a subset. ELEMENTWISE_UNARY ops take exactly one operand
// (the half itself); ELEMENTWISE_BINARY/_VARIADIC ops, when actually applied
// with 2 operands here, take the half plus exactly one external operand.
bool isUnaryAllowedOpType(Operation *op) {
#define ELEMENTWISE_UNARY(_OP_TYPE)                                            \
  if (mlir::isa<_OP_TYPE>(op))                                                 \
    return true;
#include "src/Conversion/ONNXToKrnl/Math/Elementwise.hpp"
  return false;
}
bool isBinaryAllowedOpType(Operation *op) {
#define ELEMENTWISE_BINARY(_OP_TYPE)                                           \
  if (mlir::isa<_OP_TYPE>(op))                                                 \
    return true;
#define ELEMENTWISE_VARIADIC(_OP_TYPE)                                         \
  if (mlir::isa<_OP_TYPE>(op))                                                 \
    return true;
#include "src/Conversion/ONNXToKrnl/Math/Elementwise.hpp"
  return false;
}
bool isAllowedOpType(Operation *op) {
  return isUnaryAllowedOpType(op) || isBinaryAllowedOpType(op);
}

// One resolved branch feeding the Concat: the Slice it ultimately traces
// back to, the optional per-half op (null => plain copy of the slice), and
// (when the op is binary) the op's extra, non-chain operand.
struct Branch {
  ONNXSliceOp sliceOp = nullptr;
  Operation *opNode = nullptr;
  Value externalOperand = nullptr;
};

// Resolve one Concat operand into a Branch: either directly a single-use
// Slice result, or a single-use allow-listed op (unary, or binary with
// exactly one operand equal to a single-use Slice result -- the other
// becomes the external operand).
bool resolveBranch(Value val, Branch &branch) {
  if (auto sliceOp = val.getDefiningOp<ONNXSliceOp>()) {
    if (!val.hasOneUse())
      return false;
    branch.sliceOp = sliceOp;
    return true;
  }
  Operation *opNode = val.getDefiningOp();
  if (!opNode || !val.hasOneUse() || !isAllowedOpType(opNode))
    return false;
  if (isUnaryAllowedOpType(opNode)) {
    // Guards against an op like ONNXClipOp, which the ELEMENTWISE_UNARY
    // list still counts as unary (single required data operand) even
    // though it structurally always carries 3 operands (data plus two
    // NoneType-or-real min/max operands) -- getOperand(0) alone would
    // silently ignore any real min/max bound. Simplest safe v1 behavior:
    // decline the match entirely rather than mis-handle the extra operands.
    if (opNode->getNumOperands() != 1)
      return false;
    auto sliceOp = opNode->getOperand(0).getDefiningOp<ONNXSliceOp>();
    if (!sliceOp || !opNode->getOperand(0).hasOneUse())
      return false;
    branch.sliceOp = sliceOp;
    branch.opNode = opNode;
    return true;
  }
  // Binary: exactly one operand must be a single-use Slice result; the
  // other becomes the external operand. Requires exactly 2 operands --
  // several ELEMENTWISE_VARIADIC ops (Add, Sum, Max, ...) structurally
  // accept more; declines rather than silently ignoring the rest.
  if (opNode->getNumOperands() != 2)
    return false;
  Value lhs = opNode->getOperand(0), rhs = opNode->getOperand(1);
  auto lhsSlice = lhs.getDefiningOp<ONNXSliceOp>();
  auto rhsSlice = rhs.getDefiningOp<ONNXSliceOp>();
  bool lhsOk = lhsSlice && lhs.hasOneUse();
  bool rhsOk = rhsSlice && rhs.hasOneUse();
  if (lhsOk == rhsOk) // neither, or both -- ambiguous/unsupported
    return false;
  branch.sliceOp = lhsOk ? lhsSlice : rhsSlice;
  branch.opNode = opNode;
  branch.externalOperand = lhsOk ? rhs : lhs;
  return true;
}

// Validate a branch's optional op: its external operand (if any) must be
// exactly the same shape as the branch's own slice output (no broadcast),
// and must not directly be the other branch's slice result (no cross-half
// dependency, direct-use check only per v1 scope).
bool checkBranchOp(
    const Branch &b, ONNXSliceOp otherSlice, const DimAnalysis *dimAnalysis) {
  if (!b.opNode || !b.externalOperand)
    return true;
  if (b.externalOperand == otherSlice.getResult())
    return false; // direct cross-half dependency
  ONNXSliceOp sliceOp = b.sliceOp;
  if (!isa<ShapedType>(b.externalOperand.getType()) ||
      !isa<ShapedType>(sliceOp.getResult().getType()))
    return false;
  // Comparing the two static ArrayRef<int64_t> shapes directly would treat
  // any two dynamic dims as equal, since both are encoded as the same
  // ShapedType::kDynamic sentinel regardless of whether they are actually
  // the same dimension at runtime -- e.g. the external operand's dim could
  // turn out to be 1, which broadcasts against the slice's dim, but this v1
  // fusion has no broadcast support. DimAnalysis::sameShape() instead checks
  // dynamic dims for provable equality (same dynamic-dim set), so a pair
  // that can't be proven equal is conservatively declined rather than
  // assumed to match.
  return dimAnalysis->sameShape(b.externalOperand, sliceOp.getResult());
}

} // namespace

bool SplitOpGatherFusionHelper::detectIfBeneficial(
    const DimAnalysis *dimAnalysis, ONNXConcatOp startOp) {
  auto returnFailure = [](llvm::StringRef msg) -> bool {
    LLVM_DEBUG(llvm::dbgs()
               << "  detectIfBeneficial simd-split-op-gather: " << msg << "\n");
    return false;
  };

  ops.clear();
  finalResults.clear();
  axis = -1;
  splitPoint = -1;
  hasOpForSplitLow = false;
  hasOpForSplitHigh = false;
  outputOffsetForSplitLow = -1;
  outputOffsetForSplitHigh = -1;

  if (isInsideFusedOp(startOp))
    return returnFailure("already inside a fused op body");

  // ---- Concat: exactly two inputs, axis is the innermost dim -------------
  auto concatInputs = startOp.getInputs();
  if (concatInputs.size() != 2)
    return returnFailure("concat: must have exactly two inputs");

  Value concatOut = startOp.getConcatResult();
  if (!hasShapeAndRank(concatOut))
    return returnFailure("concat: output has no shape/rank");
  int64_t rank = cast<ShapedType>(concatOut.getType()).getRank();
  int64_t A = startOp.getAxis();
  if (A < 0)
    A += rank;
  if (A < 0 || A >= rank)
    return returnFailure("concat: axis out of range after normalization");
  if (A != rank - 1)
    return returnFailure("concat: only innermost-axis split supported (v1)");

  // ---- Resolve both branches -----------------------------------------------
  Branch branch0, branch1;
  if (!resolveBranch(concatInputs[0], branch0))
    return returnFailure(
        "operand 0: not a (optionally-computed) single-use Slice result");
  if (!resolveBranch(concatInputs[1], branch1))
    return returnFailure(
        "operand 1: not a (optionally-computed) single-use Slice result");

  ONNXSliceOp slice0 = branch0.sliceOp, slice1 = branch1.sliceOp;
  if (slice0.getData() != slice1.getData())
    return returnFailure("the two slices must read the same source tensor");
  Value data = slice0.getData();
  if (!hasShapeAndRank(data))
    return returnFailure("source tensor has no shape/rank");
  int64_t dataRank = cast<ShapedType>(data.getType()).getRank();
  if (dataRank != rank)
    return returnFailure("source tensor rank does not match concat rank");

  // Each slice's `axes` operand must name exactly the split axis -- avoids
  // separately verifying that every other dim is a full-range identity
  // slice (ONNX Slice leaves un-listed axes untouched by construction).
  for (ONNXSliceOp s : {slice0, slice1}) {
    auto axesAttr = getElementAttributeFromONNXValue(s.getAxes());
    if (!axesAttr || axesAttr.getNumElements() != 1)
      return returnFailure("slice: axes must be a single-element constant");
    int64_t sliceAxis = *axesAttr.getValues<int64_t>().begin();
    if (sliceAxis < 0)
      sliceAxis += dataRank;
    if (sliceAxis != A)
      return returnFailure("slice: axes must name exactly the split axis");
  }

  // D must be static: not just so Part 2's lowering has a compile-time
  // length for VL/unroll decisions, but also so ONNX's "end >= INT32_MAX
  // means slice to the end of the axis" sentinel (see below) resolves to a
  // *literal* by construction, letting us check it below with a plain
  // integer comparison instead of re-deriving that fold ourselves. A
  // dynamic D could in principle still support the sentinel (`end`'s raw
  // constant would need reading directly, independent of D), but that's a
  // v1.1 relaxation, not implemented here.
  int64_t D = cast<ShapedType>(data.getType()).getShape()[A];
  if (D == ShapedType::kDynamic)
    return returnFailure("source tensor's split-axis dim must be static");

  // ---- Clamped/normalized start/end/step for the split axis, on each slice.
  ONNXSliceOpShapeHelper shapeHelper0(
      slice0.getOperation(), slice0.getOperation()->getOperands());
  if (failed(shapeHelper0.computeShape()))
    return returnFailure("slice 0: failed to compute shape");
  ONNXSliceOpShapeHelper shapeHelper1(
      slice1.getOperation(), slice1.getOperation()->getOperands());
  if (failed(shapeHelper1.computeShape()))
    return returnFailure("slice 1: failed to compute shape");

  IndexExpr start0 = shapeHelper0.starts[A], end0 = shapeHelper0.ends[A],
            step0 = shapeHelper0.steps[A];
  IndexExpr start1 = shapeHelper1.starts[A], end1 = shapeHelper1.ends[A],
            step1 = shapeHelper1.steps[A];
  if (!start0.isLiteral() || !end0.isLiteral() || !step0.isLiteral() ||
      !start1.isLiteral() || !end1.isLiteral() || !step1.isLiteral())
    return returnFailure("slice bounds must be compile-time literals");
  if (step0.getLiteral() != 1 || step1.getLiteral() != 1)
    return returnFailure("slices must be dense (step == 1)");

  // ---- Determine (low, high) ordering and contiguity/coverage -------------
  Branch *lowBranch, *highBranch;
  int64_t lowEnd, highStart, highEnd;
  if (start0.getLiteral() == 0) {
    lowBranch = &branch0;
    highBranch = &branch1;
    lowEnd = end0.getLiteral();
    highStart = start1.getLiteral();
    highEnd = end1.getLiteral();
  } else if (start1.getLiteral() == 0) {
    lowBranch = &branch1;
    highBranch = &branch0;
    lowEnd = end1.getLiteral();
    highStart = start0.getLiteral();
    highEnd = end0.getLiteral();
  } else {
    return returnFailure("neither slice starts at 0");
  }
  if (lowEnd != highStart)
    return returnFailure("slices are not contiguous (gap or overlap)");
  // `highEnd` is already the CLAMPED end (shapeHelper.ends[A], not the raw
  // `end` operand) -- so a raw end of e.g. INT64_MAX ("slice to the end of
  // the axis", per ONNX semantics) is not a separate case to handle here:
  // ONNXSliceOpShapeHelper::computeShape() itself folds any raw end literal
  // >= INT32_MAX to exactly D (see Slice.cpp's `endPos.selectOrSelf(endInput
  // >= posInf, dimInput)`), so `highEnd` is already D by the time we read it
  // whether the exporter wrote D directly or used that sentinel. Verified
  // against the actual Granite-4 IR, whose second Slice uses exactly this
  // INT64_MAX sentinel.
  if (highEnd != D)
    return returnFailure("high slice does not reach the end of the axis");
  splitPoint = lowEnd;

  // ---- Validate each branch's optional op ----------------------------------
  if (!checkBranchOp(*lowBranch, highBranch->sliceOp, dimAnalysis))
    return returnFailure("low half op: invalid external operand");
  if (!checkBranchOp(*highBranch, lowBranch->sliceOp, dimAnalysis))
    return returnFailure("high half op: invalid external operand");

  // ---- Output placement, from Concat's actual operand order ---------------
  int64_t lowLen = splitPoint;
  int64_t highLen = D - splitPoint;
  if (lowBranch == &branch0) {
    outputOffsetForSplitLow = 0;
    outputOffsetForSplitHigh = lowLen;
  } else {
    outputOffsetForSplitHigh = 0;
    outputOffsetForSplitLow = highLen;
  }

  axis = A;
  hasOpForSplitLow = lowBranch->opNode != nullptr;
  hasOpForSplitHigh = highBranch->opNode != nullptr;

  // ---- Fixed, deterministic op order regardless of Concat's operand order.
  ops.push_back(lowBranch->sliceOp.getOperation());
  if (hasOpForSplitLow)
    ops.push_back(lowBranch->opNode);
  ops.push_back(highBranch->sliceOp.getOperation());
  if (hasOpForSplitHigh)
    ops.push_back(highBranch->opNode);
  ops.push_back(startOp.getOperation());
  finalResults.push_back(concatOut);

  LLVM_DEBUG(llvm::dbgs() << "  simd-split-op-gather: successful\n");
  return true;
}

void SplitOpGatherFusionHelper::embedAttrs(ONNXFusedOp fusedOp) const {
  Builder b(fusedOp->getContext());
  fusedOp->setAttr("axis", b.getI64IntegerAttr(axis));
  fusedOp->setAttr("splitPoint", b.getI64IntegerAttr(splitPoint));
  fusedOp->setAttr("hasOpForSplitLow", b.getBoolAttr(hasOpForSplitLow));
  fusedOp->setAttr("hasOpForSplitHigh", b.getBoolAttr(hasOpForSplitHigh));
  fusedOp->setAttr(
      "outputOffsetForSplitLow", b.getI64IntegerAttr(outputOffsetForSplitLow));
  fusedOp->setAttr("outputOffsetForSplitHigh",
      b.getI64IntegerAttr(outputOffsetForSplitHigh));
}

bool SplitOpGatherFusionHelper::retrieveAttrs(ONNXFusedOp fusedOp) {
  auto getI64 = [&](llvm::StringRef name, int64_t &out) -> bool {
    auto attr = fusedOp->getAttrOfType<IntegerAttr>(name);
    if (!attr)
      return false;
    out = attr.getInt();
    return true;
  };
  auto getBool = [&](llvm::StringRef name, bool &out) -> bool {
    auto attr = fusedOp->getAttrOfType<BoolAttr>(name);
    if (!attr)
      return false;
    out = attr.getValue();
    return true;
  };
  if (!getI64("axis", axis))
    return false;
  if (!getI64("splitPoint", splitPoint))
    return false;
  if (!getBool("hasOpForSplitLow", hasOpForSplitLow))
    return false;
  if (!getBool("hasOpForSplitHigh", hasOpForSplitHigh))
    return false;
  if (!getI64("outputOffsetForSplitLow", outputOffsetForSplitLow))
    return false;
  if (!getI64("outputOffsetForSplitHigh", outputOffsetForSplitHigh))
    return false;
  return true;
}

bool SplitOpGatherFusionHelper::verify() const {
  size_t expected =
      3 + (hasOpForSplitLow ? 1 : 0) + (hasOpForSplitHigh ? 1 : 0);
  if (ops.size() != expected)
    return false;
  size_t idx = 0;
  if (!isa<ONNXSliceOp>(ops[idx++]))
    return false;
  if (hasOpForSplitLow && !isAllowedOpType(ops[idx++]))
    return false;
  if (!isa<ONNXSliceOp>(ops[idx++]))
    return false;
  if (hasOpForSplitHigh && !isAllowedOpType(ops[idx++]))
    return false;
  auto concatOp = dyn_cast<ONNXConcatOp>(ops[idx++]);
  if (!concatOp)
    return false;
  int64_t rank =
      cast<ShapedType>(concatOp.getConcatResult().getType()).getRank();
  int64_t A = concatOp.getAxis();
  if (A < 0)
    A += rank;
  return A == axis;
}

ONNXSliceOp SplitOpGatherFusionHelper::getSliceLowOp() const {
  return cast<ONNXSliceOp>(ops[0]);
}

Operation *SplitOpGatherFusionHelper::getOpLowNode() const {
  return hasOpForSplitLow ? ops[1] : nullptr;
}

ONNXSliceOp SplitOpGatherFusionHelper::getSliceHighOp() const {
  return cast<ONNXSliceOp>(ops[hasOpForSplitLow ? 2 : 1]);
}

Operation *SplitOpGatherFusionHelper::getOpHighNode() const {
  size_t idx = (hasOpForSplitLow ? 2 : 1) + 1;
  return hasOpForSplitHigh ? ops[idx] : nullptr;
}

} // namespace onnx_mlir
