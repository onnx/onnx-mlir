/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- OpFusionHelper.cpp - ZHigh Fusion Helper Functions ----------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpFusionHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;
using namespace llvm;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Internal helpers
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

/// Return the single user of \p val if it is of type \p T, null otherwise.
template <typename T>
static T singleUserOfType(Value val) {
  if (!val.hasOneUse())
    return nullptr;
  return dyn_cast<T>(*val.getUsers().begin());
}

/// Return true if \p perm keeps the last dimension in place.
static bool transposeKeepsLastDim(ArrayAttr perm) {
  int64_t rank = static_cast<int64_t>(perm.size());
  return ArrayAttrIntVal(perm, rank - 1) == rank - 1;
}

/// Return true if dim \p dinA of \p valA and dim \p dinB of \p valB represent
/// the same dimension.  Uses DimAnalysis when available (full precision);
/// falls back to static-value comparison with a conservative treatment of
/// dynamic dims when dimAnalysis is null.
static bool sameDim(const DimAnalysis *dimAnalysis, Value valA, int64_t dinA,
    Value valB, int64_t dinB) {
  if (dimAnalysis)
    return dimAnalysis->sameDim(valA, dinA, valB, dinB);
  // Fallback: compare the static shape values directly.
  int64_t a = cast<ShapedType>(valA.getType()).getShape()[dinA];
  int64_t b = cast<ShapedType>(valB.getType()).getShape()[dinB];
  if (a != ShapedType::kDynamic && b != ShapedType::kDynamic)
    return a == b;
  // Both dynamic: treat conservatively as the same (approximation).
  return a == ShapedType::kDynamic && b == ShapedType::kDynamic;
}

/// Try to interpret \p reshape as a split (outRank == inRank + 1).
/// Mirrors PatternsForExtendedLayoutTransform::locateReshapeSplit exactly.
/// On success fills \p axis and \p factor and returns true.
/// On failure sets \p msg to a human-readable reason.
static bool detectSplitReshape(ONNXReshapeOp reshape, int64_t &axis,
    int64_t &factor, std::string &msg, const DimAnalysis *dimAnalysis) {
  Value inputVal = reshape.getData();
  Value reshapedVal = reshape.getReshaped();
  int64_t inputRank = cast<ShapedType>(inputVal.getType()).getRank();
  int64_t reshapedRank = cast<ShapedType>(reshapedVal.getType()).getRank();
  if (reshapedRank != inputRank + 1) {
    msg = "Reshape expected to split one dim (ranks)";
    return false;
  }

  // Walk dimensions in parallel; find the single axis where the split occurs.
  axis = -1;
  int64_t din = 0, dout = 0;
  for (; din < inputRank; ++din, ++dout) {
    if (dout >= reshapedRank) {
      msg = "Reshape expected to split one dim (out of dout)";
      return false;
    }
    if (sameDim(dimAnalysis, inputVal, din, reshapedVal, dout))
      continue;
    // Found a difference — this must be the only split axis.
    if (axis != -1) {
      msg = "Reshape expected to split one dim (second split)";
      return false;
    }
    axis = din;
    ++dout; // skip the extra output dim introduced by the split
  }
  if (din != inputRank || dout != inputRank + 1) {
    msg = "Reshape expected to split one dim (end condition)";
    return false;
  }

  // The second split component is at outShape[axis + 1].
  factor = cast<ShapedType>(reshapedVal.getType()).getShape()[axis + 1];
  // Factor must be a static constant so the lowering can emit LitIE(factor).
  if (factor == ShapedType::kDynamic) {
    msg = "Reshape expected to split one dim (const in 2nd place)";
    return false;
  }
  // When splitting the last (innermost) dimension, the factor must be a
  // multiple of 64 to remain compatible with NNPA stick alignment.
  if (axis == inputRank - 1 && factor % 64 != 0) {
    msg = "Reshape of last dim supports only 0 mod 64 static shape";
    return false;
  }
  return true;
}

/// Try to interpret \p reshape as a merge (outRank == inRank - 1).
/// Mirrors PatternsForExtendedLayoutTransform::locateReshapeMerge exactly.
/// On success fills \p axis and returns true.
/// On failure sets \p msg to a human-readable reason.
static bool detectMergeReshape(ONNXReshapeOp reshape, int64_t &axis,
    std::string &msg, const DimAnalysis *dimAnalysis) {
  Value inputVal = reshape.getData();
  Value reshapedVal = reshape.getReshaped();
  int64_t inputRank = cast<ShapedType>(inputVal.getType()).getRank();
  int64_t reshapedRank = cast<ShapedType>(reshapedVal.getType()).getRank();
  if (reshapedRank != inputRank - 1) {
    msg = "Reshape expected to merge two dims (ranks)";
    return false;
  }

  // Walk dimensions in parallel; find the single axis where the merge occurs.
  axis = -1;
  int64_t din = 0, dout = 0;
  for (; dout < reshapedRank; ++dout, ++din) {
    if (din >= inputRank) {
      msg = "Reshape expected to merge one dim (out of din)";
      return false;
    }
    if (sameDim(dimAnalysis, inputVal, din, reshapedVal, dout))
      continue;
    // Found a difference — this must be the only merge axis.
    if (axis != -1) {
      msg = "Reshape expected to merge one dim (second merge)";
      return false;
    }
    axis = din;
    ++din; // skip the extra input dim consumed by the merge
  }
  if (din != reshapedRank + 1 || dout != reshapedRank) {
    msg = "Reshape expected to merge one dim (end condition)";
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Pattern-creation overload
//===----------------------------------------------------------------------===//

FailureOr<ExtLayoutTransformChain>
locateExtLayoutTransformFusion(
    ONNXLayoutTransformOp startOp, const DimAnalysis *dimAnalysis) {
  ExtLayoutTransformChain chain;
  ExtLayoutTransformFusionParams &p = chain.params;

  // ---- Step 1: validate and record the initial layout transform --------
  // Must be a ZTensor -> CPU conversion (no target layout means CPU).
  if (startOp.getTargetLayout().has_value())
    return failure();
  Value inputData = startOp.getData();
  if (!isZTensor(inputData.getType()))
    return failure();
  if (!supportedLayoutForCompilerGeneratedStickUnstick(inputData, /*nhwc=*/false))
    return failure();
  if (!hasStaticInnermostDimMod(inputData, 64))
    return failure();

  chain.ops.push_back(startOp.getOperation());
  Value current = startOp.getOutput();

  // ---- Step 2: optional split reshape ----------------------------------
  bool reshapeMayBeMerge = false;
  if (auto splitReshape = singleUserOfType<ONNXReshapeOp>(current)) {
    std::string splitMsg;
    if (detectSplitReshape(splitReshape, p.reshapeSplitAxis,
            p.reshapeSplitFactor, splitMsg, dimAnalysis)) {
      chain.ops.push_back(splitReshape.getOperation());
      current = splitReshape.getReshaped();
    } else {
      reshapeMayBeMerge = true; // might be a merge — don't advance yet
    }
  }

  // ---- Step 3: optional transpose (only when no pending merge) ----------
  if (!reshapeMayBeMerge) {
    if (auto transpose = singleUserOfType<ONNXTransposeOp>(current)) {
      auto perm = transpose.getPerm();
      if (!perm.has_value())
        return failure(); // default perm moves last dim — unsupported
      if (!transposeKeepsLastDim(perm.value()))
        return failure();
      p.transposePattern = perm;
      chain.ops.push_back(transpose.getOperation());
      current = transpose.getTransposed();
    }
  }

  // ---- Step 4: optional merge reshape ----------------------------------
  if (auto mergeReshape = singleUserOfType<ONNXReshapeOp>(current)) {
    std::string mergeMsg;
    if (detectMergeReshape(
            mergeReshape, p.reshapeMergeAxis, mergeMsg, dimAnalysis)) {
      chain.ops.push_back(mergeReshape.getOperation());
      current = mergeReshape.getReshaped();
    }
    // If detectMergeReshape fails here we just stop — no merge found.
  }

  // ---- Step 5: optional final layout transform or DLF16->F32 -----------
  if (auto finalLT = singleUserOfType<ONNXLayoutTransformOp>(current)) {
    auto layoutAttr = finalLT.getTargetLayout();
    if (!layoutAttr.has_value())
      return failure(); // second LT must target a ZTensor layout
    if (!supportedLayoutForCompilerGeneratedStickUnstick(
            finalLT.getOutput(), /*nhwc=*/false))
      return failure();
    OpBuilder b(finalLT);
    p.finalLayout = getZTensorLayoutAttr(
        b, cast<ZTensorEncodingAttr>(layoutAttr.value()));
    chain.ops.push_back(finalLT.getOperation());
    current = finalLT.getOutput();
  } else if (auto dlf = singleUserOfType<ZHighDLF16ToF32Op>(current)) {
    p.dlf16ToF32 = true;
    chain.ops.push_back(dlf.getOperation());
    current = dlf.getOut();
  }

  chain.finalResult = current;

  // ---- Step 6: beneficial check ----------------------------------------
  // Require at least: a transpose, OR a reshape together with a final LT/dlf16.
  bool hasTranspose = p.transposePattern.has_value();
  bool hasReshape = p.reshapeSplitAxis != -1 || p.reshapeMergeAxis != -1;
  bool hasFinalConv = p.finalLayout.has_value() || p.dlf16ToF32;
  if (!hasTranspose && !(hasReshape && hasFinalConv))
    return failure();

  return chain;
}


} // namespace zhigh
} // namespace onnx_mlir
