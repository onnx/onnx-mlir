/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZHighFusionOpHelper.cpp - ZHigh Fusion Helper Functions -----===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ZHighFusionOpHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "op-fusion"

using namespace mlir;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Static helpers — implementation details shared by all subclasses.
// Not members of the class hierarchy; may be reused by future subclasses.
//===----------------------------------------------------------------------===//

/// Return true if \p val has a static innermost dimension that is a multiple
/// of \p mod.
static bool hasStaticInnermostDimMod(Value val, int64_t mod) {
  if (!hasShapeAndRank(val))
    return false;
  auto type = cast<ShapedType>(val.getType());
  auto shape = type.getShape();
  int64_t rank = type.getRank();
  if (rank == 0 || shape[rank - 1] == ShapedType::kDynamic)
    return false;
  return mod <= 1 || shape[rank - 1] % mod == 0;
}

/// Return the unique user of \p val that is of type \p T, or null if there
/// isn't exactly one such user.  Unlike singleUserOfOpType, \p val is allowed
/// to have other, non-T-typed uses as well (e.g. a value that also escapes
/// to a function result); callers that need to know about those other uses
/// should check val.hasOneUse() themselves after a successful match.
template <typename T>
static T uniqueUserOfOpType(Value val) {
  T found = nullptr;
  for (Operation *user : val.getUsers()) {
    if (auto typed = dyn_cast<T>(user)) {
      if (found)
        return nullptr; // more than one T-typed user -- ambiguous
      found = typed;
    }
  }
  return found;
}

/// Return true if \p perm keeps the last dimension in place.
static bool transposeKeepsLastDim(ArrayAttr perm) {
  int64_t rank = static_cast<int64_t>(perm.size());
  return ArrayAttrIntVal(perm, rank - 1) == rank - 1;
}

/// Try to interpret \p reshape as a split (outRank == inRank + 1).
/// Mirrors PatternsForExtendedLayoutTransform::locateReshapeSplit exactly.
/// On success fills \p axis and \p factor and returns true.
static bool detectSplitReshape(ONNXReshapeOp reshape, int64_t &axis,
    int64_t &factor, const DimAnalysis *dimAnalysis) {
  assert(dimAnalysis && "detectSplitReshape requires a non-null DimAnalysis");
  auto returnFailure = [](llvm::StringRef msg) -> bool {
    LLVM_DEBUG(llvm::dbgs() << "  detectSplitReshape failed: " << msg << "\n");
    return false;
  };

  Value inputVal = reshape.getData();
  Value reshapedVal = reshape.getReshaped();
  int64_t inputRank = cast<ShapedType>(inputVal.getType()).getRank();
  int64_t reshapedRank = cast<ShapedType>(reshapedVal.getType()).getRank();
  if (reshapedRank != inputRank + 1)
    return returnFailure("split one dim (ranks)");

  // Walk dimensions in parallel; find the single axis where the split occurs.
  axis = -1;
  int64_t din = 0, dout = 0;
  for (; din < inputRank; ++din, ++dout) {
    if (dout >= reshapedRank)
      return returnFailure("split one dim (out of dout)");
    if (dimAnalysis->sameDim(inputVal, din, reshapedVal, dout))
      continue;
    // Found a difference — this must be the only split axis.
    if (axis != -1)
      return returnFailure("split one dim (second split)");
    axis = din;
    ++dout; // skip the extra output dim introduced by the split
  }
  if (din != inputRank || dout != inputRank + 1)
    return returnFailure("split one dim (end condition)");

  // The second split component is at outShape[axis + 1].
  factor = cast<ShapedType>(reshapedVal.getType()).getShape()[axis + 1];
  // Factor must be a static constant so the lowering can emit LitIE(factor).
  if (factor == ShapedType::kDynamic)
    return returnFailure("split one dim (const in 2nd place)");
  // When splitting the last (innermost) dimension, the factor must be a
  // multiple of 64 to remain compatible with NNPA stick alignment.
  if (axis == inputRank - 1 && factor % 64 != 0)
    return returnFailure("split last dim supports only 0 mod 64 static shape");
  return true;
}

/// Try to interpret \p reshape as a merge (outRank == inRank - 1).
/// Mirrors PatternsForExtendedLayoutTransform::locateReshapeMerge exactly.
/// On success fills \p axis and returns true.
static bool detectMergeReshape(
    ONNXReshapeOp reshape, int64_t &axis, const DimAnalysis *dimAnalysis) {
  assert(dimAnalysis && "detectMergeReshape requires a non-null DimAnalysis");
  auto returnFailure = [](llvm::StringRef msg) -> bool {
    LLVM_DEBUG(llvm::dbgs() << "  detectMergeReshape failed: " << msg << "\n");
    return false;
  };

  Value inputVal = reshape.getData();
  Value reshapedVal = reshape.getReshaped();
  int64_t inputRank = cast<ShapedType>(inputVal.getType()).getRank();
  int64_t reshapedRank = cast<ShapedType>(reshapedVal.getType()).getRank();
  if (reshapedRank != inputRank - 1)
    return returnFailure("merge two dims (ranks)");

  // Walk dimensions in parallel; find the single axis where the merge occurs.
  axis = -1;
  int64_t din = 0, dout = 0;
  for (; dout < reshapedRank; ++dout, ++din) {
    if (din >= inputRank)
      return returnFailure("merge one dim (out of din)");
    if (dimAnalysis->sameDim(inputVal, din, reshapedVal, dout))
      continue;
    // Found a difference — this must be the only merge axis.
    if (axis != -1)
      return returnFailure("merge one dim (second merge)");
    axis = din;
    ++din; // skip the extra input dim consumed by the merge
  }
  if (din != reshapedRank + 1 || dout != reshapedRank)
    return returnFailure("merge one dim (end condition)");
  return true;
}

//===----------------------------------------------------------------------===//
// ExtLayoutTransformFusionHelper — virtual method implementations
//===----------------------------------------------------------------------===//

bool ExtLayoutTransformFusionHelper::detectIfBeneficial(
    const DimAnalysis *dimAnalysis, ONNXLayoutTransformOp startOp) {
  auto returnFailure = [](llvm::StringRef msg) -> bool {
    LLVM_DEBUG(llvm::dbgs() << "  detectIfBeneficial ext-layout-trans failed: "
                            << msg << "\n");
    return false;
  };

  // Reset all fields.
  ops.clear();
  finalResults.clear();
  reshapeSplitAxis = -1;
  reshapeSplitFactor = 1;
  reshapeMergeAxis = -1;
  transposePattern = std::nullopt;
  dlf16ToF32 = false;
  finalLayout = std::nullopt;

  LLVM_DEBUG({
    llvm::dbgs() << "Attempt to fuse op\n  ";
    startOp.dump();
  });

  if (isInsideFusedOp(startOp))
    return returnFailure("already inside a fused op body");

  // ---- Step 1: validate and record the initial layout transform --------
  // Must be a ZTensor -> CPU conversion (no target layout means CPU).
  if (startOp.getTargetLayout().has_value())
    return returnFailure("has no target layout");
  Value inputData = startOp.getData();
  if (!isZTensor(inputData.getType()))
    return returnFailure("has no zTensor input type");
  if (!supportedLayoutForCompilerGeneratedStickUnstick(
          inputData, /*nhwc=*/false))
    return returnFailure("zTensor layout not supported");
  if (getExtendedLayoutTransformInnerTile(inputData.getType()) == 0)
    return returnFailure("zTensor inner dim is not 0 mod 64, or 32");

  ops.push_back(startOp.getOperation());
  Value current = startOp.getOutput();

  // ---- Step 2: optional split reshape ----------------------------------
  bool reshapeMayBeMerge = false;
  if (auto splitReshape = singleUserOfOpType<ONNXReshapeOp>(current)) {
    if (detectSplitReshape(
            splitReshape, reshapeSplitAxis, reshapeSplitFactor, dimAnalysis)) {
      ops.push_back(splitReshape.getOperation());
      current = splitReshape.getReshaped();
    } else {
      reshapeMayBeMerge = true; // might be a merge — don't advance yet
    }
  }

  // ---- Step 3: optional transpose (only when no pending merge) ----------
  if (!reshapeMayBeMerge) {
    if (auto transpose = singleUserOfOpType<ONNXTransposeOp>(current)) {
      auto perm = transpose.getPerm();
      if (!perm.has_value())
        return returnFailure("default perm unsupported");
      if (!transposeKeepsLastDim(perm.value()))
        return returnFailure("perm last dim");
      transposePattern = perm;
      ops.push_back(transpose.getOperation());
      current = transpose.getTransposed();
    }
  }

  // ---- Step 4: optional merge reshape ----------------------------------
  if (auto mergeReshape = singleUserOfOpType<ONNXReshapeOp>(current)) {
    if (detectMergeReshape(mergeReshape, reshapeMergeAxis, dimAnalysis)) {
      ops.push_back(mergeReshape.getOperation());
      current = mergeReshape.getReshaped();
    }
    // If detectMergeReshape fails here we just stop — no merge found.
  }

  // ---- Step 5: optional final layout transform or DLF16->F32 -----------
  if (auto finalLT = singleUserOfOpType<ONNXLayoutTransformOp>(current)) {
    auto layoutAttr = finalLT.getTargetLayout();
    if (!layoutAttr.has_value())
      return returnFailure("second LT must target a zTensor layout");
    if (!supportedLayoutForCompilerGeneratedStickUnstick(
            finalLT.getOutput(), /*nhwc=*/false))
      return returnFailure("unsupported target zTensor type");
    OpBuilder b(finalLT);
    finalLayout =
        getZTensorLayoutAttr(b, cast<ZTensorEncodingAttr>(layoutAttr.value()));
    ops.push_back(finalLT.getOperation());
    current = finalLT.getOutput();
  } else if (auto dlf = singleUserOfOpType<ZHighDLF16ToF32Op>(current)) {
    dlf16ToF32 = true;
    ops.push_back(dlf.getOperation());
    current = dlf.getOut();
  }

  finalResults.push_back(current);

  // ---- Step 6: beneficial check ----------------------------------------
  // Require at least: a transpose, OR a reshape together with a final LT/dlf16.
  bool hasTranspose = transposePattern.has_value();
  bool hasReshape = reshapeSplitAxis != -1 || reshapeMergeAxis != -1;
  bool hasFinalConv = finalLayout.has_value() || dlf16ToF32;
  if (!hasTranspose && !(hasReshape && hasFinalConv))
    return returnFailure("successful but NOT beneficial");

  LLVM_DEBUG(llvm::dbgs() << "  successful and beneficial\n");
  return true;
}

void ExtLayoutTransformFusionHelper::embedAttrs(ONNXFusedOp fusedOp) const {
  Builder b(fusedOp->getContext());
  fusedOp->setAttr("reshapeSplitAxis", b.getI64IntegerAttr(reshapeSplitAxis));
  fusedOp->setAttr(
      "reshapeSplitFactor", b.getI64IntegerAttr(reshapeSplitFactor));
  fusedOp->setAttr("reshapeMergeAxis", b.getI64IntegerAttr(reshapeMergeAxis));
  fusedOp->setAttr("dlf16ToF32", b.getBoolAttr(dlf16ToF32));
  if (transposePattern.has_value())
    fusedOp->setAttr("transposePattern", *transposePattern);
  if (finalLayout.has_value())
    fusedOp->setAttr("finalLayout", *finalLayout);
}

bool ExtLayoutTransformFusionHelper::retrieveAttrs(ONNXFusedOp fusedOp) {
  auto getI64 = [&](StringRef name, int64_t &out) -> bool {
    auto attr = fusedOp->getAttrOfType<IntegerAttr>(name);
    if (!attr)
      return false;
    out = attr.getInt();
    return true;
  };
  if (!getI64("reshapeSplitAxis", reshapeSplitAxis))
    return false;
  if (!getI64("reshapeSplitFactor", reshapeSplitFactor))
    return false;
  if (!getI64("reshapeMergeAxis", reshapeMergeAxis))
    return false;
  auto dlf = fusedOp->getAttrOfType<BoolAttr>("dlf16ToF32");
  if (!dlf)
    return false;
  dlf16ToF32 = dlf.getValue();
  // Optional attrs.
  if (auto attr = fusedOp->getAttrOfType<ArrayAttr>("transposePattern"))
    transposePattern = attr;
  else
    transposePattern = std::nullopt;
  if (auto attr = fusedOp->getAttrOfType<StringAttr>("finalLayout"))
    finalLayout = attr;
  else
    finalLayout = std::nullopt;
  return true;
}

bool ExtLayoutTransformFusionHelper::verify() const {
  // Expected op count from the stored params.
  int expected = 1; // ops[0]: initial ONNXLayoutTransformOp
  if (reshapeSplitAxis != -1)
    ++expected;
  if (transposePattern.has_value())
    ++expected;
  if (reshapeMergeAxis != -1)
    ++expected;
  if (dlf16ToF32 || finalLayout.has_value())
    ++expected;

  if ((int64_t)ops.size() != expected) {
    LLVM_DEBUG(llvm::dbgs() << "ELT verify: op count " << ops.size()
                            << " != expected " << expected << "\n");
    return false;
  }

  int idx = 0;

  // ops[0]: initial ONNXLayoutTransformOp with no target layout.
  auto lt0 = dyn_cast<ONNXLayoutTransformOp>(ops[idx++]);
  if (!lt0 || lt0.getTargetLayout().has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "ELT verify: ops[0] not initial LT\n");
    return false;
  }

  // Optional split reshape.
  if (reshapeSplitAxis != -1) {
    auto reshape = dyn_cast<ONNXReshapeOp>(ops[idx++]);
    if (!reshape) {
      LLVM_DEBUG(llvm::dbgs() << "ELT verify: expected split Reshape\n");
      return false;
    }
    auto inType = cast<ShapedType>(reshape.getData().getType());
    auto outType = cast<ShapedType>(reshape.getReshaped().getType());
    if (outType.getRank() != inType.getRank() + 1) {
      LLVM_DEBUG(llvm::dbgs() << "ELT verify: split Reshape rank mismatch\n");
      return false;
    }
    if (outType.getShape()[reshapeSplitAxis + 1] != reshapeSplitFactor) {
      LLVM_DEBUG(llvm::dbgs() << "ELT verify: split factor mismatch\n");
      return false;
    }
  }

  // Optional transpose.
  if (transposePattern.has_value()) {
    auto transpose = dyn_cast<ONNXTransposeOp>(ops[idx++]);
    if (!transpose) {
      LLVM_DEBUG(llvm::dbgs() << "ELT verify: expected Transpose\n");
      return false;
    }
    auto perm = transpose.getPerm();
    if (!perm.has_value() || perm.value() != *transposePattern) {
      LLVM_DEBUG(llvm::dbgs() << "ELT verify: transpose perm mismatch\n");
      return false;
    }
  }

  // Optional merge reshape.
  if (reshapeMergeAxis != -1) {
    auto reshape = dyn_cast<ONNXReshapeOp>(ops[idx++]);
    if (!reshape) {
      LLVM_DEBUG(llvm::dbgs() << "ELT verify: expected merge Reshape\n");
      return false;
    }
    auto inType = cast<ShapedType>(reshape.getData().getType());
    auto outType = cast<ShapedType>(reshape.getReshaped().getType());
    if (outType.getRank() != inType.getRank() - 1) {
      LLVM_DEBUG(llvm::dbgs() << "ELT verify: merge Reshape rank mismatch\n");
      return false;
    }
  }

  // Optional final step.
  if (dlf16ToF32) {
    if (!dyn_cast<ZHighDLF16ToF32Op>(ops[idx++])) {
      LLVM_DEBUG(llvm::dbgs() << "ELT verify: expected DLF16ToF32\n");
      return false;
    }
  } else if (finalLayout.has_value()) {
    auto lt = dyn_cast<ONNXLayoutTransformOp>(ops[idx++]);
    if (!lt || !lt.getTargetLayout().has_value()) {
      LLVM_DEBUG(llvm::dbgs() << "ELT verify: expected final LT\n");
      return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// ExpandMulStickFusionHelper — static helpers
//===----------------------------------------------------------------------===//

/// Verify that expandOp expands only dim P (currently 1) to a static N >= 2,
/// with all other dims unchanged.  Returns N on success, -1 on failure.
static int64_t detectExpandedDim(
    ONNXExpandOp expandOp, int64_t P, const DimAnalysis *dimAnalysis) {
  auto fail = [](llvm::StringRef msg) -> int64_t {
    LLVM_DEBUG(llvm::dbgs() << "  detectExpandedDim: " << msg << "\n");
    return -1;
  };
  Value inVal = expandOp.getInput();
  Value outVal = expandOp.getOutput();
  if (!hasShapeAndRank(inVal) || !hasShapeAndRank(outVal))
    return fail("no shape/rank");
  auto inType = cast<ShapedType>(inVal.getType());
  auto outType = cast<ShapedType>(outVal.getType());
  if (inType.getRank() != outType.getRank())
    return fail("rank mismatch");
  int64_t rank = inType.getRank();
  if (P < 0 || P >= rank)
    return fail("P out of range");
  if (inType.getShape()[P] != 1)
    return fail("dim P not 1 in expand input");
  int64_t N = outType.getShape()[P];
  if (N == ShapedType::kDynamic || N < 2)
    return fail("dim P output dynamic or < 2");
  for (int64_t i = 0; i < rank; ++i) {
    if (i == P)
      continue;
    if (!dimAnalysis->sameDim(inVal, i, outVal, i) &&
        inType.getShape()[i] != outType.getShape()[i])
      return fail("non-P dim changed");
  }
  return N;
}

/// Verify that reshapeOp only collapses dims in [0..P]; dims strictly after P
/// (positions P+1..inRank-1) must be identical in the output.  Fills
/// firstCollapsedDim and collapsedCount (0 = no-op reshape).  Returns true on
/// success.
static bool detectUpperCollapse(ONNXReshapeOp reshapeOp, int64_t P,
    int64_t &firstCollapsedDim, int64_t &collapsedCount,
    const DimAnalysis *dimAnalysis) {
  auto fail = [](llvm::StringRef msg) -> bool {
    LLVM_DEBUG(llvm::dbgs() << "  detectUpperCollapse: " << msg << "\n");
    return false;
  };
  Value inVal = reshapeOp.getData();
  Value outVal = reshapeOp.getReshaped();
  if (!hasShapeAndRank(inVal) || !hasShapeAndRank(outVal))
    return fail("no shape/rank");
  auto inType = cast<ShapedType>(inVal.getType());
  auto outType = cast<ShapedType>(outVal.getType());
  int64_t inRank = inType.getRank();
  int64_t outRank = outType.getRank();
  if (outRank > inRank)
    return fail("reshape increases rank");

  int64_t numExtra = inRank - outRank; // dims that disappeared

  // No-op reshape.
  if (numExtra == 0) {
    firstCollapsedDim = -1;
    collapsedCount = 0;
    return true;
  }

  // Tail: input dims P+1..inRank-1 must match the last (inRank-P-1) output
  // dims.
  int64_t tailLen = inRank - (P + 1);
  auto inShape = inType.getShape();
  auto outShape = outType.getShape();
  for (int64_t i = 0; i < tailLen; ++i) {
    int64_t inIdx = P + 1 + i;
    int64_t outIdx = outRank - tailLen + i;
    if (outIdx < 0)
      return fail("tail dims shifted out of output");
    if (!dimAnalysis->sameDim(inVal, inIdx, outVal, outIdx)) {
      if (inShape[inIdx] == ShapedType::kDynamic ||
          outShape[outIdx] == ShapedType::kDynamic ||
          inShape[inIdx] != outShape[outIdx])
        return fail("tail dim differs");
    }
  }

  // Head: input dims 0..P (P+1 dims) -> output dims 0..(outRank-tailLen-1).
  int64_t headOutputEnd = outRank - tailLen; // exclusive
  firstCollapsedDim = -1;
  int64_t din = 0, dout = 0;
  while (din <= P && dout < headOutputEnd) {
    if (dimAnalysis->sameDim(inVal, din, outVal, dout)) {
      ++din;
      ++dout;
    } else {
      if (firstCollapsedDim != -1)
        return fail("more than one merge run in head");
      firstCollapsedDim = din;
      // The run spans (numExtra+1) input dims merged into 1 output dim.
      din += numExtra + 1;
      ++dout;
    }
  }
  if (din != P + 1 || dout != headOutputEnd)
    return fail("head walk end mismatch");
  if (firstCollapsedDim == -1)
    return fail("rank changed but no merge run found");
  collapsedCount = numExtra + 1;
  return true;
}

//===----------------------------------------------------------------------===//
// Shared tail-matching helpers -- used by both ExpandMulStickFusionHelper's
// own tail (Expand -> Mul? -> Reshape -> Stick) and
// ConcatExpandStickFusionHelper's 1-step stick tail, which is structurally
// identical from Expand onward.
//===----------------------------------------------------------------------===//

/// If `current`'s single user is an ONNXMulOp by an F32/I32/I64 scalar
/// constant (either operand order), sets `mulScalar` to that value and
/// returns the ONNXMulOp. Otherwise returns null and leaves `mulScalar`
/// untouched -- absence of a Mul is not a failure, just means the neutral
/// (1.f) scalar applies; callers should keep matching against `current`
/// unchanged in that case.
static ONNXMulOp detectOptionalScalarMul(Value current, float &mulScalar) {
  auto mulOp = singleUserOfOpType<ONNXMulOp>(current);
  if (!mulOp)
    return nullptr;
  // Identify the scalar operand (accept either argument order): the other
  // operand must be a constant, since both extraction paths below require
  // one.
  ONNXConstantOp cst;
  if (!matchValueAndOp<ONNXConstantOp>(
          mulOp.getA(), mulOp.getB(), current, cst)) {
    LLVM_DEBUG(llvm::dbgs() << "  detectOptionalScalarMul: scalar operand "
                               "not found or not a constant\n");
    return nullptr;
  }

  std::optional<float> sv = std::nullopt;
  // F32 path: reuse existing NNPA helper.
  if (auto fa = getScalarF32AttrFromConstant(cst.getResult()))
    sv = fa.getValue().convertToFloat();
  // Integer path: fall back to getScalarValue (handles I32 / I64).
  else {
    Type et = cast<ShapedType>(cst.getType()).getElementType();
    if (et.isInteger(32) || et.isInteger(64))
      sv = static_cast<float>(getScalarValue<double>(cst));
  }
  if (!sv) {
    LLVM_DEBUG(llvm::dbgs() << "  detectOptionalScalarMul: scalar operand is "
                               "not F32/I32/I64 constant\n");
    return nullptr;
  }
  mulScalar = *sv;
  return mulOp;
}

/// If `current`'s single user is a ZHighStickOp targeting a supported layout
/// (3D, 3DS, or 4D), sets `stickFormat` and returns the ZHighStickOp.
/// Otherwise returns null.
static ZHighStickOp detectStickTail(
    Value current, std::optional<StringAttr> &stickFormat) {
  auto stickOp = singleUserOfOpType<ZHighStickOp>(current);
  if (!stickOp)
    return nullptr;
  auto layoutAttr = stickOp.getLayout();
  if (!layoutAttr)
    return nullptr;
  if (*layoutAttr != LAYOUT_3D && *layoutAttr != LAYOUT_3DS &&
      *layoutAttr != LAYOUT_4D)
    return nullptr;
  stickFormat = StringAttr::get(stickOp->getContext(), *layoutAttr);
  return stickOp;
}

//===----------------------------------------------------------------------===//
// ExpandMulStickFusionHelper — virtual method implementations
//===----------------------------------------------------------------------===//

bool ExpandMulStickFusionHelper::detectIfBeneficial(
    const DimAnalysis *dimAnalysis, ONNXUnsqueezeOp startOp) {
  auto returnFailure = [](llvm::StringRef msg) -> bool {
    LLVM_DEBUG(llvm::dbgs()
               << "  detectIfBeneficial expand-mul-stick: " << msg << "\n");
    return false;
  };

  // Reset all fields.
  ops.clear();
  finalResults.clear();
  unsqueezedPosition = -1;
  expansionN = -1;
  mulScalar = 1.f;
  reshapeFirstCollapsedDim = -1;
  reshapeCollapsedCount = 0;
  stickFormat = std::nullopt;

  if (isInsideFusedOp(startOp))
    return returnFailure("already inside a fused op body");

  LLVM_DEBUG({
    llvm::dbgs() << "Attempt to fuse expand-mul-stick from\n  ";
    startOp.dump();
  });

  // ---- Step 1: Unsqueeze --------------------------------------------------
  // axes operand must be a constant with exactly one element.
  Value axesVal = startOp.getAxes();
  auto axesAttr = getElementAttributeFromONNXValue(axesVal);
  if (!axesAttr || axesAttr.getNumElements() != 1)
    return returnFailure("unsqueeze: must have exactly one axis");

  int64_t P = (*axesAttr.getValues<int64_t>().begin());
  Value inputData = startOp.getData();
  if (!hasShapeAndRank(inputData))
    return returnFailure("unsqueeze: input has no shape/rank");
  int64_t outputRank = cast<ShapedType>(inputData.getType()).getRank() + 1;
  if (P < 0)
    P += outputRank; // normalize negative axis
  if (P < 0 || P >= outputRank)
    return returnFailure("unsqueeze: axis out of range after normalization");

  Value unsqOut = startOp.getExpanded();
  if (!hasStaticInnermostDimMod(unsqOut, 64))
    return returnFailure("unsqueeze: innermost dim not static mod 64");

  ops.push_back(startOp.getOperation());
  Value current = unsqOut;
  unsqueezedPosition = P;

  // ---- Step 2: Expand -----------------------------------------------------
  auto expandOp = singleUserOfOpType<ONNXExpandOp>(current);
  if (!expandOp)
    return returnFailure("expand: not single user of type ONNXExpandOp");
  int64_t N = detectExpandedDim(expandOp, P, dimAnalysis);
  if (N < 0)
    return returnFailure("expand: dim P not expanded to static N >= 2");
  expansionN = N;
  ops.push_back(expandOp.getOperation());
  current = expandOp.getOutput();

  // ---- Step 3: Mul (optional) ----------------------------------------------
  // If the expand output's single user is not a viable scalar Mul, skip this
  // step and leave mulScalar at its neutral default (1.f); `current` still
  // points at the expand output, so Step 4 matches the reshape directly
  // against it.
  if (auto mulOp = detectOptionalScalarMul(current, mulScalar)) {
    ops.push_back(mulOp.getOperation());
    current = mulOp.getC();
  }

  // ---- Step 4: Reshape ----------------------------------------------------
  auto reshapeOp = singleUserOfOpType<ONNXReshapeOp>(current);
  if (!reshapeOp)
    return returnFailure("reshape: not single user of type ONNXReshapeOp");
  if (!detectUpperCollapse(reshapeOp, P, reshapeFirstCollapsedDim,
          reshapeCollapsedCount, dimAnalysis))
    return returnFailure("reshape: invalid collapse");
  ops.push_back(reshapeOp.getOperation());
  current = reshapeOp.getReshaped();

  // ---- Step 5: Stick (single-use check included in singleUserOfOpType) ------
  auto stickOp = detectStickTail(current, stickFormat);
  if (!stickOp)
    return returnFailure(
        "stick: not single user of type ZHighStickOp, or unsupported layout");
  ops.push_back(stickOp.getOperation());
  finalResults.push_back(stickOp.getOut());

  LLVM_DEBUG(llvm::dbgs() << "  expand-mul-stick: successful\n");
  return true;
}

void ExpandMulStickFusionHelper::embedAttrs(ONNXFusedOp fusedOp) const {
  Builder b(fusedOp->getContext());
  fusedOp->setAttr(
      "unsqueezedPosition", b.getI64IntegerAttr(unsqueezedPosition));
  fusedOp->setAttr("expansionN", b.getI64IntegerAttr(expansionN));
  fusedOp->setAttr("mulScalar",
      b.getFloatAttr(b.getF32Type(), static_cast<double>(mulScalar)));
  fusedOp->setAttr("reshapeFirstCollapsedDim",
      b.getI64IntegerAttr(reshapeFirstCollapsedDim));
  fusedOp->setAttr(
      "reshapeCollapsedCount", b.getI64IntegerAttr(reshapeCollapsedCount));
  fusedOp->setAttr("stickFormat", *stickFormat);
}

bool ExpandMulStickFusionHelper::retrieveAttrs(ONNXFusedOp fusedOp) {
  auto getI64 = [&](StringRef name, int64_t &out) -> bool {
    auto attr = fusedOp->getAttrOfType<IntegerAttr>(name);
    if (!attr)
      return false;
    out = attr.getInt();
    return true;
  };
  if (!getI64("unsqueezedPosition", unsqueezedPosition))
    return false;
  if (!getI64("expansionN", expansionN))
    return false;
  if (!getI64("reshapeFirstCollapsedDim", reshapeFirstCollapsedDim))
    return false;
  if (!getI64("reshapeCollapsedCount", reshapeCollapsedCount))
    return false;
  auto scalarAttr = fusedOp->getAttrOfType<FloatAttr>("mulScalar");
  if (!scalarAttr)
    return false;
  mulScalar = scalarAttr.getValue().convertToFloat();
  auto fmtAttr = fusedOp->getAttrOfType<StringAttr>("stickFormat");
  if (!fmtAttr)
    return false;
  stickFormat = fmtAttr;
  return true;
}

bool ExpandMulStickFusionHelper::verify() const {
  constexpr int expectedWithMul =
      5; // unsqueeze + expand + mul + reshape + stick
  constexpr int expectedWithoutMul = 4; // unsqueeze + expand + reshape + stick
  bool hasMul;
  if ((int64_t)ops.size() == expectedWithMul) {
    hasMul = true;
  } else if ((int64_t)ops.size() == expectedWithoutMul) {
    hasMul = false;
  } else {
    LLVM_DEBUG(llvm::dbgs() << "EMS verify: op count " << ops.size()
                            << " != " << expectedWithMul << " or "
                            << expectedWithoutMul << "\n");
    return false;
  }
  int idx = 0;

  // ops[0]: ONNXUnsqueezeOp — axes constant has exactly one element.
  auto unsq = dyn_cast<ONNXUnsqueezeOp>(ops[idx++]);
  if (!unsq) {
    LLVM_DEBUG(llvm::dbgs() << "EMS verify: ops[0] not Unsqueeze\n");
    return false;
  }
  {
    auto axesAttr = getElementAttributeFromONNXValue(unsq.getAxes());
    if (!axesAttr || axesAttr.getNumElements() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "EMS verify: unsqueeze axes changed\n");
      return false;
    }
  }

  // ops[1]: ONNXExpandOp — output dim P must equal expansionN.
  auto exp = dyn_cast<ONNXExpandOp>(ops[idx++]);
  if (!exp) {
    LLVM_DEBUG(llvm::dbgs() << "EMS verify: ops[1] not Expand\n");
    return false;
  }
  {
    auto outType = cast<ShapedType>(exp.getOutput().getType());
    if (outType.getRank() <= unsqueezedPosition ||
        outType.getShape()[unsqueezedPosition] != expansionN) {
      LLVM_DEBUG(llvm::dbgs() << "EMS verify: expand N mismatch\n");
      return false;
    }
  }

  // ops[2]: ONNXMulOp (only present when the pattern includes a Mul).
  if (hasMul) {
    if (!dyn_cast<ONNXMulOp>(ops[idx++])) {
      LLVM_DEBUG(llvm::dbgs() << "EMS verify: ops[2] not Mul\n");
      return false;
    }
  }

  // ops[idx]: ONNXReshapeOp — rank delta consistent with reshapeCollapsedCount.
  auto reshape = dyn_cast<ONNXReshapeOp>(ops[idx++]);
  if (!reshape) {
    LLVM_DEBUG(llvm::dbgs() << "EMS verify: ops[idx] not Reshape\n");
    return false;
  }
  if (reshapeCollapsedCount > 0) {
    int64_t inRank = cast<ShapedType>(reshape.getData().getType()).getRank();
    int64_t outRank =
        cast<ShapedType>(reshape.getReshaped().getType()).getRank();
    // reshapeCollapsedCount input dims merge into 1: net loss = count - 1.
    if (inRank - outRank != reshapeCollapsedCount - 1) {
      LLVM_DEBUG(llvm::dbgs() << "EMS verify: reshape rank delta mismatch\n");
      return false;
    }
  }

  // ops[idx]: ZHighStickOp — layout matches stored stickFormat.
  auto stick = dyn_cast<ZHighStickOp>(ops[idx++]);
  if (!stick) {
    LLVM_DEBUG(llvm::dbgs() << "EMS verify: ops[idx] not Stick\n");
    return false;
  }
  {
    auto layoutAttr = stick.getLayout();
    if (!layoutAttr || !stickFormat.has_value() ||
        *layoutAttr != stickFormat->getValue()) {
      LLVM_DEBUG(llvm::dbgs() << "EMS verify: stick layout mismatch\n");
      return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// ConcatExpandStickFusionHelper — virtual method implementations
//===----------------------------------------------------------------------===//

bool ConcatExpandStickFusionHelper::detectIfBeneficial(
    const DimAnalysis *dimAnalysis, ONNXConcatOp startOp) {
  auto returnFailure = [](llvm::StringRef msg) -> bool {
    LLVM_DEBUG(llvm::dbgs()
               << "  detectIfBeneficial concat-expand-stick: " << msg << "\n");
    return false;
  };

  // Reset all fields.
  ops.clear();
  finalResults.clear();
  concatAxis = -1;
  unsqueezedPosition = -1;
  expansionN = -1;
  noSaturation = false;
  reshapeFirstCollapsedDim = -1;
  reshapeCollapsedCount = 0;
  finalLayout = std::nullopt;
  yieldConcatResult = false;
  mulScalar = 1.f;
  stickFormat = std::nullopt;

  if (isInsideFusedOp(startOp))
    return returnFailure("already inside a fused op body");

  LLVM_DEBUG({
    llvm::dbgs() << "Attempt to fuse concat-expand-stick from\n  ";
    startOp.dump();
  });

  // ---- Step 1: Concat -----------------------------------------------------
  auto concatInputs = startOp.getInputs();
  if (concatInputs.size() != 2)
    return returnFailure("concat: must have exactly two inputs");

  Value concatOut = startOp.getConcatResult();
  if (!hasShapeAndRank(concatOut))
    return returnFailure("concat: output has no shape/rank");
  int64_t concatRank = cast<ShapedType>(concatOut.getType()).getRank();
  int64_t A = startOp.getAxis();
  if (A < 0)
    A += concatRank; // normalize negative axis
  if (A < 0 || A >= concatRank)
    return returnFailure("concat: axis out of range after normalization");
  if (A == concatRank - 1)
    return returnFailure("concat: do not support concat on innermost dim");
  concatAxis = A;
  // Because of limitation in the code gen scheme to innermost dims being
  // multiple of 64, check that assertion here. Since innermost is not the
  // concat dim, by definition all inputs must have the same size in the
  // innermost dim, so we can just check the first input.
  if (!hasStaticInnermostDimMod(concatInputs[0], 64))
    return returnFailure("concat: input innermost dim not static mod 64");

  ops.push_back(startOp.getOperation());
  Value current = concatOut;

  // ---- Step 2: Unsqueeze ---------------------------------------------------
  // The concat result may have other, non-chain uses (e.g. it also feeds a
  // KV-cache passthrough output) -- allow that here, as long as exactly one
  // of its users is the Unsqueeze that starts the rest of the chain.  When
  // such other uses exist, the concat result is threaded through as a
  // second FusedOp output (see yieldConcatResult below).
  auto unsqOp = uniqueUserOfOpType<ONNXUnsqueezeOp>(current);
  if (!unsqOp)
    return returnFailure(
        "unsqueeze: not the unique user of type ONNXUnsqueezeOp");
  yieldConcatResult = !current.hasOneUse();

  Value axesVal = unsqOp.getAxes();
  auto axesAttr = getElementAttributeFromONNXValue(axesVal);
  if (!axesAttr || axesAttr.getNumElements() != 1)
    return returnFailure("unsqueeze: must have exactly one axis");

  int64_t P = (*axesAttr.getValues<int64_t>().begin());
  int64_t outputRank = concatRank + 1;
  if (P < 0)
    P += outputRank; // normalize negative axis
  if (P < 0 || P >= outputRank)
    return returnFailure("unsqueeze: axis out of range after normalization");
  if (P >= concatRank)
    return returnFailure("unsqueeze: axis must be < concatRank");
  ops.push_back(unsqOp.getOperation());
  current = unsqOp.getExpanded();
  unsqueezedPosition = P;

  // ---- Step 3: F32ToDLF16 (optional) ---------------------------------------
  // Present <=> the existing 2-step tail (ends in a LayoutTransform);
  // absent <=> the 1-step tail (ends in an explicit ZHighStickOp). Expand
  // and Reshape are identical either way -- shared below -- only this step,
  // the optional Mul (Step 5, stick-tail only -- see its own comment for
  // why), and the final step (Step 6) differ per tail.
  bool hasSingleStick;
  if (auto dlfOp = singleUserOfOpType<ZHighF32ToDLF16Op>(current)) {
    hasSingleStick = false;
    if (auto ns = dlfOp.getNoSaturation())
      noSaturation = (*ns != 0);
    ops.push_back(dlfOp.getOperation());
    current = dlfOp.getOut();
  } else {
    hasSingleStick = true;
  }

  // ---- Step 4: Expand -------------------------------------------------------
  auto expandOp = singleUserOfOpType<ONNXExpandOp>(current);
  if (!expandOp)
    return returnFailure("expand: not single user of type ONNXExpandOp");
  int64_t N = detectExpandedDim(expandOp, P, dimAnalysis);
  if (N < 0)
    return returnFailure("expand: dim P not expanded to static N >= 2");
  expansionN = N;
  ops.push_back(expandOp.getOperation());
  current = expandOp.getOutput();

  // ---- Step 5: Mul (optional, stick-tail only) ------------------------------
  // Only tried for the 1-step tail: by this point in the 2-step tail, data is
  // already DLF16 (F32ToDLF16 ran in Step 3), so a real Mul here would need
  // an F16-typed scalar constant (ONNX's Mul requires both operands to share
  // one element type) -- detectOptionalScalarMul only recognizes F32/I32/I64
  // scalars, so that combination could never actually match; skip trying it
  // rather than leave dead code. If the expand output's single user is not a
  // viable scalar Mul, skip this step and leave mulScalar at its neutral
  // default (1.f); `current` still points at the expand output either way,
  // so Step 5.5 matches the reshape directly against it.
  if (hasSingleStick) {
    if (auto mulOp = detectOptionalScalarMul(current, mulScalar)) {
      ops.push_back(mulOp.getOperation());
      current = mulOp.getC();
    }
  }

  // ---- Step 5.5: Reshape
  // -----------------------------------------------------
  auto reshapeOp = singleUserOfOpType<ONNXReshapeOp>(current);
  if (!reshapeOp)
    return returnFailure("reshape: not single user of type ONNXReshapeOp");
  if (!detectUpperCollapse(reshapeOp, P, reshapeFirstCollapsedDim,
          reshapeCollapsedCount, dimAnalysis))
    return returnFailure("reshape: invalid collapse");
  ops.push_back(reshapeOp.getOperation());
  current = reshapeOp.getReshaped();

  // ---- Step 6: LayoutTransform or Stick, per hasSingleStick -----------------
  if (!hasSingleStick) {
    auto ltOp = singleUserOfOpType<ONNXLayoutTransformOp>(current);
    if (!ltOp)
      return returnFailure(
          "layout-transform: not single user of type ONNXLayoutTransformOp");
    auto layoutAttr = ltOp.getTargetLayout();
    if (!layoutAttr.has_value())
      return returnFailure("layout-transform: must target a zTensor layout");
    if (!supportedLayoutForCompilerGeneratedStickUnstick(
            ltOp.getOutput(), /*nhwc=*/false))
      return returnFailure("layout-transform: unsupported target zTensor type");
    OpBuilder b(ltOp);
    StringAttr layoutStrAttr =
        getZTensorLayoutAttr(b, cast<ZTensorEncodingAttr>(layoutAttr.value()));
    StringRef layoutStr = layoutStrAttr.getValue();
    if (layoutStr != LAYOUT_3D && layoutStr != LAYOUT_3DS &&
        layoutStr != LAYOUT_4D)
      return returnFailure(
          "layout-transform: unsupported layout (need 3D, 3DS, or 4D)");
    finalLayout = layoutStrAttr;
    ops.push_back(ltOp.getOperation());
    finalResults.push_back(ltOp.getOutput());
  } else {
    auto stickOp = detectStickTail(current, stickFormat);
    if (!stickOp)
      return returnFailure("stick: not single user of type ZHighStickOp, or "
                           "unsupported layout");
    ops.push_back(stickOp.getOperation());
    finalResults.push_back(stickOp.getOut());
  }

  // The primary result is always output 0; the concat result, when also
  // needed outside the chain, is threaded through as output 1.
  if (yieldConcatResult)
    finalResults.push_back(concatOut);

  LLVM_DEBUG(llvm::dbgs() << "  concat-expand-stick: successful\n");
  return true;
}

void ConcatExpandStickFusionHelper::embedAttrs(ONNXFusedOp fusedOp) const {
  Builder b(fusedOp->getContext());
  fusedOp->setAttr("concatAxis", b.getI64IntegerAttr(concatAxis));
  fusedOp->setAttr(
      "unsqueezedPosition", b.getI64IntegerAttr(unsqueezedPosition));
  fusedOp->setAttr("expansionN", b.getI64IntegerAttr(expansionN));
  fusedOp->setAttr("noSaturation", b.getBoolAttr(noSaturation));
  fusedOp->setAttr("reshapeFirstCollapsedDim",
      b.getI64IntegerAttr(reshapeFirstCollapsedDim));
  fusedOp->setAttr(
      "reshapeCollapsedCount", b.getI64IntegerAttr(reshapeCollapsedCount));
  fusedOp->setAttr("mulScalar",
      b.getFloatAttr(b.getF32Type(), static_cast<double>(mulScalar)));
  // Exactly one of the two is set after a successful match (2-step tail vs.
  // 1-step stick tail) -- only write the one that applies.
  if (finalLayout.has_value())
    fusedOp->setAttr("finalLayout", *finalLayout);
  if (stickFormat.has_value())
    fusedOp->setAttr("stickFormat", *stickFormat);
  fusedOp->setAttr("yieldConcatResult", b.getBoolAttr(yieldConcatResult));
}

bool ConcatExpandStickFusionHelper::retrieveAttrs(ONNXFusedOp fusedOp) {
  auto getI64 = [&](StringRef name, int64_t &out) -> bool {
    auto attr = fusedOp->getAttrOfType<IntegerAttr>(name);
    if (!attr)
      return false;
    out = attr.getInt();
    return true;
  };
  if (!getI64("concatAxis", concatAxis))
    return false;
  if (!getI64("unsqueezedPosition", unsqueezedPosition))
    return false;
  if (!getI64("expansionN", expansionN))
    return false;
  if (!getI64("reshapeFirstCollapsedDim", reshapeFirstCollapsedDim))
    return false;
  if (!getI64("reshapeCollapsedCount", reshapeCollapsedCount))
    return false;
  auto satAttr = fusedOp->getAttrOfType<BoolAttr>("noSaturation");
  if (!satAttr)
    return false;
  noSaturation = satAttr.getValue();
  auto yieldAttr = fusedOp->getAttrOfType<BoolAttr>("yieldConcatResult");
  if (!yieldAttr)
    return false;
  yieldConcatResult = yieldAttr.getValue();
  auto scalarAttr = fusedOp->getAttrOfType<FloatAttr>("mulScalar");
  if (!scalarAttr)
    return false;
  mulScalar = scalarAttr.getValue().convertToFloat();
  // Optional attrs -- exactly one should be present after a successful
  // match; verify() checks that (retrieveAttrs must not fail just because a
  // chain-shape-specific attr is, correctly, absent).
  if (auto attr = fusedOp->getAttrOfType<StringAttr>("finalLayout"))
    finalLayout = attr;
  else
    finalLayout = std::nullopt;
  if (auto attr = fusedOp->getAttrOfType<StringAttr>("stickFormat"))
    stickFormat = attr;
  else
    stickFormat = std::nullopt;
  return true;
}

bool ConcatExpandStickFusionHelper::verify() const {
  // Exactly one of the two tail shapes' target-layout field must be set --
  // anything else means retrieveAttrs picked up an inconsistent op.
  bool hasSingleStick = stickFormat.has_value();
  if (hasSingleStick == finalLayout.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "CES verify: expected exactly one of "
                               "finalLayout/stickFormat to be set\n");
    return false;
  }

  // Base count (no Mul): concat + unsqueeze + [f32-to-dlf16] + expand +
  // reshape + [layout-transform | stick] -- 5 fixed ops, plus 1 more for
  // f32-to-dlf16 when this is the 2-step tail. Mul, when present, adds one
  // -- but only for the stick tail (see detectIfBeneficial's Step 5 comment
  // for why the 2-step tail can never actually have one).
  int64_t baseCount = hasSingleStick ? 5 : 6;
  bool hasMul;
  if (hasSingleStick && (int64_t)ops.size() == baseCount + 1) {
    hasMul = true;
  } else if ((int64_t)ops.size() == baseCount) {
    hasMul = false;
  } else {
    LLVM_DEBUG(llvm::dbgs() << "CES verify: op count " << ops.size() << " != "
                            << baseCount << " or " << baseCount + 1 << "\n");
    return false;
  }
  int idx = 0;

  // ops[0]: ONNXConcatOp — exactly two inputs.
  auto concat = dyn_cast<ONNXConcatOp>(ops[idx++]);
  if (!concat || concat.getInputs().size() != 2) {
    LLVM_DEBUG(llvm::dbgs() << "CES verify: ops[0] not a 2-input Concat\n");
    return false;
  }

  // ops[1]: ONNXUnsqueezeOp — axes constant has exactly one element.
  auto unsq = dyn_cast<ONNXUnsqueezeOp>(ops[idx++]);
  if (!unsq) {
    LLVM_DEBUG(llvm::dbgs() << "CES verify: ops[1] not Unsqueeze\n");
    return false;
  }
  {
    auto axesAttr = getElementAttributeFromONNXValue(unsq.getAxes());
    if (!axesAttr || axesAttr.getNumElements() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "CES verify: unsqueeze axes changed\n");
      return false;
    }
  }

  // ops[idx]: ZHighF32ToDLF16Op (only present for the 2-step tail).
  if (!hasSingleStick) {
    if (!dyn_cast<ZHighF32ToDLF16Op>(ops[idx++])) {
      LLVM_DEBUG(llvm::dbgs() << "CES verify: ops[idx] not F32ToDLF16\n");
      return false;
    }
  }

  // ops[idx]: ONNXExpandOp — output dim P must equal expansionN.
  auto exp = dyn_cast<ONNXExpandOp>(ops[idx++]);
  if (!exp) {
    LLVM_DEBUG(llvm::dbgs() << "CES verify: ops[idx] not Expand\n");
    return false;
  }
  {
    auto outType = cast<ShapedType>(exp.getOutput().getType());
    if (outType.getRank() <= unsqueezedPosition ||
        outType.getShape()[unsqueezedPosition] != expansionN) {
      LLVM_DEBUG(llvm::dbgs() << "CES verify: expand N mismatch\n");
      return false;
    }
  }

  // ops[idx]: ONNXMulOp (only present when the pattern includes a Mul).
  if (hasMul) {
    if (!dyn_cast<ONNXMulOp>(ops[idx++])) {
      LLVM_DEBUG(llvm::dbgs() << "CES verify: ops[idx] not Mul\n");
      return false;
    }
  }

  // ops[idx]: ONNXReshapeOp — rank delta consistent with reshapeCollapsedCount.
  auto reshape = dyn_cast<ONNXReshapeOp>(ops[idx++]);
  if (!reshape) {
    LLVM_DEBUG(llvm::dbgs() << "CES verify: ops[idx] not Reshape\n");
    return false;
  }
  if (reshapeCollapsedCount > 0) {
    int64_t inRank = cast<ShapedType>(reshape.getData().getType()).getRank();
    int64_t outRank =
        cast<ShapedType>(reshape.getReshaped().getType()).getRank();
    if (inRank - outRank != reshapeCollapsedCount - 1) {
      LLVM_DEBUG(llvm::dbgs() << "CES verify: reshape rank delta mismatch\n");
      return false;
    }
  }

  // ops[idx]: ONNXLayoutTransformOp or ZHighStickOp, per hasSingleStick.
  if (!hasSingleStick) {
    auto lt = dyn_cast<ONNXLayoutTransformOp>(ops[idx++]);
    if (!lt) {
      LLVM_DEBUG(llvm::dbgs() << "CES verify: ops[idx] not LayoutTransform\n");
      return false;
    }
    auto layoutAttr = lt.getTargetLayout();
    if (!layoutAttr.has_value()) {
      LLVM_DEBUG(llvm::dbgs() << "CES verify: missing target layout\n");
      return false;
    }
    OpBuilder b(lt.getContext());
    StringAttr layoutStrAttr =
        getZTensorLayoutAttr(b, cast<ZTensorEncodingAttr>(layoutAttr.value()));
    if (layoutStrAttr != *finalLayout) {
      LLVM_DEBUG(llvm::dbgs() << "CES verify: layout mismatch\n");
      return false;
    }
  } else {
    auto stick = dyn_cast<ZHighStickOp>(ops[idx++]);
    if (!stick) {
      LLVM_DEBUG(llvm::dbgs() << "CES verify: ops[idx] not Stick\n");
      return false;
    }
    auto layoutAttr = stick.getLayout();
    if (!layoutAttr || *layoutAttr != stickFormat->getValue()) {
      LLVM_DEBUG(llvm::dbgs() << "CES verify: stick layout mismatch\n");
      return false;
    }
  }

  // finalResults: output 0 is always the primary result; output 1, when
  // yieldConcatResult is set, must be the concat's own result.
  size_t expectedResults = yieldConcatResult ? 2 : 1;
  if (finalResults.size() != expectedResults) {
    LLVM_DEBUG(llvm::dbgs()
               << "CES verify: result count " << finalResults.size()
               << " != " << expectedResults << "\n");
    return false;
  }
  if (yieldConcatResult && finalResults[1] != concat.getConcatResult()) {
    LLVM_DEBUG(llvm::dbgs() << "CES verify: second result is not the "
                               "concat result\n");
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// UnstickSplitHeadsFusionHelper
//===----------------------------------------------------------------------===//

// Rank of the reshaped (A, S, N, H, D) value, and the position of N in it.
static constexpr int64_t kSplitHeadsRank = 5;
static constexpr int64_t kSplitHeadsNAxis = 2;

/// Return true when the head dim \p D and head count \p H can be processed in
/// whole sticks: each of the N slices of the innermost dim (H * D elements)
/// starts on a stick boundary, and every 64-element stick holds either two
/// whole heads (D == 32) or a 64-aligned part of a single head (D % 64 == 0).
static bool supportedSplitHeadsDims(int64_t H, int64_t D) {
  if (H <= 0 || D <= 0)
    return false;
  if (D != 32 && D % 64 != 0)
    return false;
  return (H * D) % 64 == 0;
}

/// Position, after the optional transpose, of the reshaped N axis.
static int64_t splitHeadsNAxisAfterTranspose(
    std::optional<ArrayAttr> transposePattern) {
  if (!transposePattern.has_value())
    return kSplitHeadsNAxis;
  for (int64_t p = 0; p < kSplitHeadsRank; ++p)
    if (ArrayAttrIntVal(transposePattern, p) == kSplitHeadsNAxis)
      return p;
  return -1;
}

/// Return true if the Split \p splitOp cuts its input into \p N slices of
/// size 1 along its axis: either no split operand (equal split) or a
/// constant split operand made of N ones.
static bool splitsIntoUnitSlices(ONNXSplitOp splitOp, int64_t N) {
  if ((int64_t)splitOp.getNumResults() != N)
    return false;
  Value split = splitOp.getSplit();
  if (isNoneValue(split))
    return true;
  SmallVector<int64_t, 4> sizes;
  if (!getI64ValuesFromONNXConstantOp(split, sizes))
    return false;
  if ((int64_t)sizes.size() != N)
    return false;
  return llvm::all_of(sizes, [](int64_t s) { return s == 1; });
}

/// Return true if, once the unit N axis is squeezed out, the Split results
/// are ordered (A, H, S, D), i.e. the reshaped dims (0, 3, 1, 4). Namely the
/// dim 2 disappeared because it was squeezed out. Required by the stick-3DS
/// outputs, whose sticks are rows of D values for (a * H + h, s).
static bool splitHeadsSqueezedIsAHSD(
    std::optional<ArrayAttr> transposePattern) {
  if (!transposePattern.has_value())
    return false;
  SmallVector<int64_t, 4> order;
  for (int64_t p = 0; p < kSplitHeadsRank; ++p) {
    int64_t d = ArrayAttrIntVal(transposePattern, p);
    if (d != kSplitHeadsNAxis)
      order.emplace_back(d);
  }
  return order == SmallVector<int64_t, 4>{0, 3, 1, 4};
}

/// Return true if \p squeezeOp removes exactly the axis \p axis of its rank 5
/// input.
static bool squeezesOnlyAxis(ONNXSqueezeOp squeezeOp, int64_t axis) {
  SmallVector<int64_t, 1> axes;
  if (!getI64ValuesFromONNXConstantOp(squeezeOp.getAxes(), axes) ||
      axes.size() != 1)
    return false;
  int64_t a = axes[0] < 0 ? axes[0] + kSplitHeadsRank : axes[0];
  return a == axis;
}

/// The Squeeze -> Reshape -> 3DS Stick chain of a stick-3DS output.
struct SplitHeadsStickTail {
  ONNXSqueezeOp squeezeOp;
  ONNXReshapeOp reshapeOp;
  ZHighStickOp stickOp;
};

/// Return true if \p squeezeOp, \p reshapeOp and \p stickOp form the chain of
/// a stick-3DS output, see the UnstickSplitHeadsFusionHelper class comment.
/// The use counts are not checked. When \p dimAnalysis is null, the dynamic
/// sequence dims are compared by the static shapes only (used by verify(),
/// after detection already proved them equal).
static bool isSplitHeadsStickTail(const DimAnalysis *dimAnalysis,
    Value splitResult, ONNXSqueezeOp squeezeOp, ONNXReshapeOp reshapeOp,
    ZHighStickOp stickOp, int64_t splitAxis, int64_t D) {
  if (squeezeOp.getData() != splitResult ||
      !squeezesOnlyAxis(squeezeOp, splitAxis))
    return false;
  Value squeezed = squeezeOp.getSqueezed();
  if (!hasShapeAndRank(squeezed) || getRank(squeezed.getType()) != 4)
    return false;
  // (A, H, S, D) => (A * H, S, D): S and D unchanged. The total size is
  // unchanged, so the merged dim is A * H.
  if (reshapeOp.getData() != squeezed)
    return false;
  Value reshaped = reshapeOp.getReshaped();
  if (!hasShapeAndRank(reshaped) || getRank(reshaped.getType()) != 3)
    return false;
  if (getShape(reshaped.getType(), 2) != D)
    return false;
  int64_t inS = getShape(squeezed.getType(), 2);
  int64_t outS = getShape(reshaped.getType(), 1);
  bool sameS = dimAnalysis && dimAnalysis->sameDim(squeezed, 2, reshaped, 1);
  // Static fallback; in verify() (no DimAnalysis), a dynamic S that stays
  // dynamic is accepted too.
  if (!sameS)
    sameS = inS == outS && (inS != ShapedType::kDynamic || !dimAnalysis);
  if (!sameS)
    return false;
  if (stickOp.getIn() != reshaped || !isZTensor(stickOp.getOut().getType()))
    return false;
  return getZTensorLayout(stickOp.getOut().getType()) ==
         ZTensorEncodingAttr::DataLayout::_3DS;
}

/// Match the stick-3DS chain of \p splitResult, each op being the single use
/// of the previous value. Returns std::nullopt if there is none.
static std::optional<SplitHeadsStickTail> matchSplitHeadsStickTail(
    const DimAnalysis *dimAnalysis, Value splitResult, int64_t splitAxis,
    int64_t D, Operation *startOp) {
  auto squeezeOp = singleUserOfOpType<ONNXSqueezeOp>(splitResult);
  if (!squeezeOp)
    return std::nullopt;
  auto reshapeOp = singleUserOfOpType<ONNXReshapeOp>(squeezeOp.getSqueezed());
  if (!reshapeOp)
    return std::nullopt;
  auto stickOp = singleUserOfOpType<ZHighStickOp>(reshapeOp.getReshaped());
  if (!stickOp)
    return std::nullopt;
  if (!isSplitHeadsStickTail(dimAnalysis, splitResult, squeezeOp, reshapeOp,
          stickOp, splitAxis, D))
    return std::nullopt;
  // The reshape shape is cloned into the body when it is a constant or a
  // Concat of constants and Dims (absorbShapeConcatOfDims()); the Dims, or
  // a non-absorbable shape, become inputs. Require those inputs to be
  // defined before the anchor, so that the stick-3DS output never makes the
  // FusedOp placement infeasible: placement then has the same constraints as
  // with all outputs in f32.
  auto isConstant = [](Operation *def) {
    return def->hasTrait<mlir::OpTrait::ConstantLike>() ||
           isa<ONNXConstantOp>(def);
  };
  auto isEarly = [&](Value v) {
    Operation *def = v.getDefiningOp();
    if (!def || isConstant(def))
      return true; // Block argument or constant.
    return def->getBlock() == startOp->getBlock() &&
           def->isBeforeInBlock(startOp);
  };
  Value shape = reshapeOp.getShape();
  if (!isEarly(shape)) {
    // Must be absorbed: a Concat of constants and early Dims.
    auto concatOp = shape.getDefiningOp<ONNXConcatOp>();
    if (!concatOp)
      return std::nullopt;
    for (Value operand : concatOp.getInputs()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !(isConstant(def) || isa<ONNXDimOp>(def)) ||
          !isEarly(operand))
        return std::nullopt;
    }
  }
  return SplitHeadsStickTail{squeezeOp, reshapeOp, stickOp};
}

bool UnstickSplitHeadsFusionHelper::detectIfBeneficial(
    const DimAnalysis *dimAnalysis, ZHighUnstickOp startOp) {
  assert(dimAnalysis && "unstick-split-heads requires a non-null DimAnalysis");
  auto returnFailure = [](llvm::StringRef msg) -> bool {
    LLVM_DEBUG(llvm::dbgs()
               << "  detectIfBeneficial unstick-split-heads failed: " << msg
               << "\n");
    return false;
  };

  // Reset all fields.
  ops.clear();
  finalResults.clear();
  numSplits = -1;
  numHeads = -1;
  headDim = -1;
  transposePattern = std::nullopt;
  splitAxis = -1;
  outputModes.clear();

  if (isInsideFusedOp(startOp))
    return returnFailure("already inside a fused op body");

  // ---- Step 1: Unstick of a 3D / 3DS ZTensor with a static innermost dim --
  Value inputData = startOp.getIn();
  if (!isZTensor(inputData.getType()))
    return returnFailure("unstick input is not a zTensor");
  ZTensorEncodingAttr::DataLayout layout =
      getZTensorLayout(inputData.getType());
  if (layout != ZTensorEncodingAttr::DataLayout::_3D &&
      layout != ZTensorEncodingAttr::DataLayout::_3DS)
    return returnFailure("unstick layout is not 3D or 3DS");
  if (!supportedLayoutForCompilerGeneratedStickUnstick(
          inputData, /*nhwc=*/false))
    return returnFailure("zTensor layout not supported");
  Value unstickedVal = startOp.getOut();
  if (!hasShapeAndRank(unstickedVal) || getRank(unstickedVal.getType()) != 3)
    return returnFailure("unstick result is not rank 3");
  int64_t C = getShape(unstickedVal.getType(), 2);
  if (C == ShapedType::kDynamic)
    return returnFailure("innermost dim is dynamic");
  ops.push_back(startOp.getOperation());

  // ---- Step 2: Reshape (A, S, C) => (A, S, N, H, D) -------------------------
  auto reshapeOp = singleUserOfOpType<ONNXReshapeOp>(unstickedVal);
  if (!reshapeOp)
    return returnFailure("unstick not single-used by a Reshape");
  Value reshapedVal = reshapeOp.getReshaped();
  if (!hasShapeAndRank(reshapedVal) ||
      getRank(reshapedVal.getType()) != kSplitHeadsRank)
    return returnFailure("reshape result is not rank 5");
  // Leading dims A and S must be unchanged.
  for (int64_t d = 0; d < 2; ++d) {
    if (dimAnalysis->sameDim(unstickedVal, d, reshapedVal, d))
      continue;
    int64_t inDim = getShape(unstickedVal.getType(), d);
    int64_t outDim = getShape(reshapedVal.getType(), d);
    if (inDim == ShapedType::kDynamic || inDim != outDim)
      return returnFailure("reshape changes a leading dim");
  }
  int64_t N = getShape(reshapedVal.getType(), 2);
  int64_t H = getShape(reshapedVal.getType(), 3);
  int64_t D = getShape(reshapedVal.getType(), 4);
  if (N == ShapedType::kDynamic || H == ShapedType::kDynamic ||
      D == ShapedType::kDynamic)
    return returnFailure("reshape split factors are not static");
  if (N < 2)
    return returnFailure("fewer than 2 splits");
  if (N * H * D != C)
    return returnFailure("reshape does not split the innermost dim");
  if (!supportedSplitHeadsDims(H, D))
    return returnFailure("head dims not stick aligned");
  numSplits = N;
  numHeads = H;
  headDim = D;
  ops.push_back(reshapeOp.getOperation());
  Value current = reshapedVal;

  // ---- Step 3: optional Transpose keeping D last ---------------------------
  if (auto transposeOp = singleUserOfOpType<ONNXTransposeOp>(current)) {
    auto perm = transposeOp.getPerm();
    if (!perm.has_value())
      return returnFailure("default perm unsupported");
    if ((int64_t)ArrayAttrSize(perm) != kSplitHeadsRank ||
        !transposeKeepsLastDim(*perm))
      return returnFailure("transpose moves the innermost dim");
    transposePattern = perm;
    ops.push_back(transposeOp.getOperation());
    current = transposeOp.getTransposed();
  }

  // ---- Step 4: Split on the N axis into N unit slices ----------------------
  auto splitOp = singleUserOfOpType<ONNXSplitOp>(current);
  if (!splitOp || splitOp.getInput() != current)
    return returnFailure("not single-used by a Split on its data input");
  int64_t axis = splitOp.getAxis();
  if (axis < 0)
    axis += kSplitHeadsRank;
  int64_t expectedAxis = splitHeadsNAxisAfterTranspose(transposePattern);
  if (axis != expectedAxis)
    return returnFailure("split axis is not the N axis");
  if (!splitsIntoUnitSlices(splitOp, N))
    return returnFailure("split is not N slices of size 1");
  for (Value result : splitOp.getResults())
    if (!hasShapeAndRank(result))
      return returnFailure("split result has no shape");
  splitAxis = axis;
  ops.push_back(splitOp.getOperation());

  // ---- Step 5: per output, optional Squeeze -> Reshape -> 3DS Stick -------
  // Tails are appended in the block order of their Stick, so that ops.back()
  // is the latest chain op (default FusedOp insertion point).
  SmallVector<SplitHeadsStickTail, 4> tails;
  for (Value result : splitOp.getResults()) {
    std::optional<SplitHeadsStickTail> tail;
    if (splitHeadsSqueezedIsAHSD(transposePattern))
      tail = matchSplitHeadsStickTail(dimAnalysis, result, axis, D, startOp);
    if (tail.has_value()) {
      outputModes.emplace_back(OutputMode::Stick3DS);
      finalResults.push_back(tail->stickOp.getOut());
      tails.emplace_back(*tail);
    } else {
      outputModes.emplace_back(OutputMode::F32);
      finalResults.push_back(result);
    }
  }
  llvm::sort(tails, [](SplitHeadsStickTail &a, SplitHeadsStickTail &b) {
    return a.stickOp->isBeforeInBlock(b.stickOp);
  });
  for (SplitHeadsStickTail &tail : tails) {
    ops.push_back(tail.squeezeOp.getOperation());
    ops.push_back(tail.reshapeOp.getOperation());
    ops.push_back(tail.stickOp.getOperation());
  }

  // Always beneficial: replaces the unstick, transpose, and N split copy
  // loops by a single pass over the stickified data.
  LLVM_DEBUG(llvm::dbgs() << "  unstick-split-heads: successful\n");
  return true;
}

void UnstickSplitHeadsFusionHelper::embedAttrs(ONNXFusedOp fusedOp) const {
  Builder b(fusedOp->getContext());
  fusedOp->setAttr("numSplits", b.getI64IntegerAttr(numSplits));
  fusedOp->setAttr("numHeads", b.getI64IntegerAttr(numHeads));
  fusedOp->setAttr("headDim", b.getI64IntegerAttr(headDim));
  fusedOp->setAttr("splitAxis", b.getI64IntegerAttr(splitAxis));
  if (transposePattern.has_value())
    fusedOp->setAttr("transposePattern", *transposePattern);
  SmallVector<StringRef, 4> modes;
  for (OutputMode mode : outputModes)
    modes.emplace_back(mode == OutputMode::F32 ? kModeF32 : kModeStick3DS);
  fusedOp->setAttr("outputModes", b.getStrArrayAttr(modes));
}

bool UnstickSplitHeadsFusionHelper::retrieveAttrs(ONNXFusedOp fusedOp) {
  auto getI64 = [&](StringRef name, int64_t &out) -> bool {
    auto attr = fusedOp->getAttrOfType<IntegerAttr>(name);
    if (!attr)
      return false;
    out = attr.getInt();
    return true;
  };
  if (!getI64("numSplits", numSplits))
    return false;
  if (!getI64("numHeads", numHeads))
    return false;
  if (!getI64("headDim", headDim))
    return false;
  if (!getI64("splitAxis", splitAxis))
    return false;
  outputModes.clear();
  auto modes = fusedOp->getAttrOfType<ArrayAttr>("outputModes");
  if (!modes)
    return false;
  for (Attribute attr : modes) {
    auto mode = dyn_cast<StringAttr>(attr);
    if (!mode)
      return false;
    if (mode.getValue() == kModeF32)
      outputModes.emplace_back(OutputMode::F32);
    else if (mode.getValue() == kModeStick3DS)
      outputModes.emplace_back(OutputMode::Stick3DS);
    else
      return false;
  }
  // Optional attr.
  if (auto attr = fusedOp->getAttrOfType<ArrayAttr>("transposePattern"))
    transposePattern = attr;
  else
    transposePattern = std::nullopt;
  return true;
}

bool UnstickSplitHeadsFusionHelper::verify() const {
  auto fail = [](llvm::StringRef msg) -> bool {
    LLVM_DEBUG(llvm::dbgs() << "unstick-split-heads verify: " << msg << "\n");
    return false;
  };
  if ((int64_t)outputModes.size() != numSplits)
    return fail("output mode count mismatch");
  int64_t numStickOutputs = llvm::count(outputModes, OutputMode::Stick3DS);
  if (numStickOutputs > 0 && !splitHeadsSqueezedIsAHSD(transposePattern))
    return fail("stick-3DS output needs (A, H, S, D) squeezed dims");
  int64_t numChainOps = transposePattern.has_value() ? 4 : 3;
  if ((int64_t)ops.size() != numChainOps + 3 * numStickOutputs)
    return fail("op count mismatch");
  if (numSplits < 2 || !supportedSplitHeadsDims(numHeads, headDim))
    return fail("unsupported N, H, or D");
  if (splitAxis != splitHeadsNAxisAfterTranspose(transposePattern))
    return fail("split axis does not hold N");
  if ((int64_t)finalResults.size() != numSplits)
    return fail("result count mismatch");

  int64_t idx = 0;
  auto unstickOp = dyn_cast<ZHighUnstickOp>(ops[idx++]);
  if (!unstickOp)
    return fail("expected Unstick");
  ZTensorEncodingAttr::DataLayout layout =
      getZTensorLayout(unstickOp.getIn().getType());
  if (layout != ZTensorEncodingAttr::DataLayout::_3D &&
      layout != ZTensorEncodingAttr::DataLayout::_3DS)
    return fail("unstick layout is not 3D or 3DS");

  auto reshapeOp = dyn_cast<ONNXReshapeOp>(ops[idx++]);
  if (!reshapeOp || reshapeOp.getData() != unstickOp.getOut())
    return fail("expected Reshape of the Unstick");
  Type reshapedType = reshapeOp.getReshaped().getType();
  if (getRank(reshapedType) != kSplitHeadsRank ||
      getShape(reshapedType, 2) != numSplits ||
      getShape(reshapedType, 3) != numHeads ||
      getShape(reshapedType, 4) != headDim)
    return fail("reshape shape mismatch");
  if (getShape(unstickOp.getOut().getType(), 2) !=
      numSplits * numHeads * headDim)
    return fail("innermost dim mismatch");
  Value current = reshapeOp.getReshaped();

  if (transposePattern.has_value()) {
    auto transposeOp = dyn_cast<ONNXTransposeOp>(ops[idx++]);
    if (!transposeOp || transposeOp.getData() != current)
      return fail("expected Transpose of the Reshape");
    auto perm = transposeOp.getPerm();
    if (!perm.has_value() || perm.value() != *transposePattern)
      return fail("transpose perm mismatch");
    current = transposeOp.getTransposed();
  }

  auto splitOp = dyn_cast<ONNXSplitOp>(ops[idx++]);
  if (!splitOp || splitOp.getInput() != current)
    return fail("expected Split of the chain");
  int64_t axis = splitOp.getAxis();
  if (axis < 0)
    axis += kSplitHeadsRank;
  if (axis != splitAxis)
    return fail("split axis mismatch");
  if (!splitsIntoUnitSlices(splitOp, numSplits))
    return fail("split sizes mismatch");

  // The remaining ops are the stick-3DS tails, 3 ops each, in any order.
  SmallVector<bool, 4> tailSeen(numSplits, false);
  for (; idx < (int64_t)ops.size(); idx += 3) {
    auto squeezeOp = dyn_cast<ONNXSqueezeOp>(ops[idx]);
    auto reshapeOp = dyn_cast<ONNXReshapeOp>(ops[idx + 1]);
    auto stickOp = dyn_cast<ZHighStickOp>(ops[idx + 2]);
    if (!squeezeOp || !reshapeOp || !stickOp)
      return fail("expected Squeeze -> Reshape -> Stick");
    auto splitResult = dyn_cast<OpResult>(squeezeOp.getData());
    if (!splitResult || splitResult.getOwner() != splitOp)
      return fail("Squeeze does not consume a Split result");
    int64_t k = splitResult.getResultNumber();
    if (outputModes[k] != OutputMode::Stick3DS || tailSeen[k])
      return fail("unexpected Squeeze -> Reshape -> Stick");
    tailSeen[k] = true;
    if (!isSplitHeadsStickTail(/*dimAnalysis=*/nullptr, splitResult, squeezeOp,
            reshapeOp, stickOp, splitAxis, headDim))
      return fail("Squeeze -> Reshape -> Stick shape mismatch");
    if (finalResults[k] != stickOp.getOut())
      return fail("yielded value is not the Stick result");
  }
  for (int64_t k = 0; k < numSplits; ++k)
    if (outputModes[k] == OutputMode::F32 &&
        finalResults[k] != splitOp.getResult(k))
      return fail("yielded value is not the Split result");
  return true;
}

} // namespace zhigh
} // namespace onnx_mlir
