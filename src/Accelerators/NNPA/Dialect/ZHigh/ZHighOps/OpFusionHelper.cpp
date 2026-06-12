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

/// Try to interpret \p reshape as a split (outRank == inRank + 1).
/// On success fills \p axis and \p factor and returns true.
static bool detectSplitReshape(
    ONNXReshapeOp reshape, int64_t &axis, int64_t &factor) {
  auto inType = cast<ShapedType>(reshape.getData().getType());
  auto outType = cast<ShapedType>(reshape.getReshaped().getType());
  if (outType.getRank() != inType.getRank() + 1)
    return false;
  auto inShape = inType.getShape();
  auto outShape = outType.getShape();
  for (int64_t d = 0, e = inType.getRank(); d < e; ++d) {
    if (inShape[d] != outShape[d]) {
      axis = d;
      factor = outShape[d + 1];
      return true;
    }
  }
  return false;
}

/// Try to interpret \p reshape as a merge (outRank == inRank - 1).
/// On success fills \p axis and returns true.
static bool detectMergeReshape(ONNXReshapeOp reshape, int64_t &axis) {
  auto inType = cast<ShapedType>(reshape.getData().getType());
  auto outType = cast<ShapedType>(reshape.getReshaped().getType());
  if (outType.getRank() != inType.getRank() - 1)
    return false;
  auto inShape = inType.getShape();
  auto outShape = outType.getShape();
  for (int64_t d = 0, e = outType.getRank(); d < e; ++d) {
    if (inShape[d] != outShape[d]) {
      axis = d;
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Pattern-creation overload
//===----------------------------------------------------------------------===//

FailureOr<ExtLayoutTransformChain>
locateExtLayoutTransformFusion(ONNXLayoutTransformOp startOp) {
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
    if (detectSplitReshape(splitReshape, p.reshapeSplitAxis, p.reshapeSplitFactor)) {
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
    if (detectMergeReshape(mergeReshape, p.reshapeMergeAxis)) {
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
