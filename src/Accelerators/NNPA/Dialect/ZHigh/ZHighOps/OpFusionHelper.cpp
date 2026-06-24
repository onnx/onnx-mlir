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

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "op-fusion-helper"

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
/// Mirrors PatternsForExtendedLayoutTransform::locateReshapeSplit exactly.
/// On success fills \p axis and \p factor and returns true.
static bool detectSplitReshape(
    ONNXReshapeOp reshape, int64_t &axis, int64_t &factor,
    const DimAnalysis *dimAnalysis) {
  assert(dimAnalysis && "detectSplitReshape requires a non-null DimAnalysis");
  auto returnFailure = [](llvm::StringRef msg) -> bool {
    LLVM_DEBUG(llvm::dbgs() << "detectSplitReshape failed: " << msg << "\n");
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
    LLVM_DEBUG(llvm::dbgs() << "detectMergeReshape failed: " << msg << "\n");
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
// FusionOpChain — non-virtual method implementations
//===----------------------------------------------------------------------===//

ONNXFusedOp FusionOpChain::createFusedOp(
    PatternRewriter &rewriter, Location loc, StringRef kind) {
  // Build the set of values produced by the chain ops themselves; these
  // are visible inside the body via the clone mapping and never external.
  DenseSet<Value> chainProduced;
  for (Operation *op : ops)
    for (Value result : op->getResults())
      chainProduced.insert(result);

  // Pre-scan: collect ALL external values the chain ops need.
  //   - Constant-like ops (ONNXConstantOp, ConstantLike trait) will be
  //     cloned inside the body — they do NOT become block arguments.
  //   - Everything else (non-constant tensors, e.g. dynamically-computed
  //     reshape shape vectors) becomes an additional FusedOp input.
  // The first external non-constant operand of ops[0] becomes inputs[0].
  SmallVector<Value> fusedInputs;
  DenseSet<Value> inputSet;

  std::function<void(Value)> collectExternals = [&](Value v) {
    // Already tracked or produced by a chain op → nothing to do.
    if (inputSet.contains(v) || chainProduced.contains(v))
      return;
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return; // bare block argument — should already be tracked
    if (defOp->hasTrait<mlir::OpTrait::ConstantLike>() ||
        mlir::isa<ONNXNoneOp, ONNXConstantOp>(defOp)) {
      // Constant: will be cloned inside; recurse for its own operands.
      for (Value operand : defOp->getOperands())
        collectExternals(operand);
    } else {
      // Non-constant external tensor: thread through as a FusedOp input.
      inputSet.insert(v);
      fusedInputs.push_back(v);
    }
  };
  for (Operation *op : ops)
    for (Value operand : op->getOperands())
      collectExternals(operand);

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
  Block *body =
      rewriter.createBlock(&fusedOp.getBody(), {}, argTypes, argLocs);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);

  // Map every fusedInput to its corresponding block argument.
  IRMapping mapping;
  for (auto [v, arg] : llvm::zip(fusedInputs, body->getArguments()))
    mapping.map(v, arg);

  // Recursively clone constant-producing ops inside the body on demand.
  std::function<void(Value)> ensureInBody = [&](Value v) {
    if (mapping.contains(v) || chainProduced.contains(v))
      return;
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return;
    // All remaining unmapped externals must be constant-like; non-constants
    // were added to fusedInputs above and are already in the mapping.
    assert((defOp->hasTrait<mlir::OpTrait::ConstantLike>() ||
               mlir::isa<ONNXNoneOp, ONNXConstantOp>(defOp)) &&
           "non-constant external value not collected in pre-scan");
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

ONNXFusedOp FusionOpChain::fuse(PatternRewriter &rewriter, Location loc) {
  // Insert the FusedOp just before the last chain op.  ops is in chain order,
  // so ops.back() is defined after every non-constant external input to any
  // chain op — guaranteeing that all fusedInputs dominate the insertion point.
  assert(!ops.empty() && "fuse() called with empty ops list");
  rewriter.setInsertionPoint(ops.back());
  ONNXFusedOp fusedOp = create(rewriter, loc);
  replaceAndErase(rewriter, fusedOp);
  return fusedOp;
}

ONNXFusedOp FusionOpChain::create(PatternRewriter &rewriter, Location loc) {
  ONNXFusedOp fusedOp = createFusedOp(rewriter, loc, getKind());
  embedAttrs(fusedOp);
  return fusedOp;
}

void FusionOpChain::retrieveOpsAndOutputValues(ONNXFusedOp fusedOp) {
  ops.clear();
  finalResults.clear();
  Block &body = fusedOp.getBody().front();
  for (Operation &op : body) {
    if (isa<ONNXYieldOp>(&op)) {
      for (Value v : op.getOperands())
        finalResults.push_back(v);
    } else {
      ops.push_back(&op);
    }
  }
}

void FusionOpChain::replaceAndErase(
    PatternRewriter &rewriter, ONNXFusedOp fusedOp) {
  // Map each finalResult value to its FusedOp output index.
  DenseMap<Value, unsigned> outputMap;
  for (auto [idx, v] : llvm::enumerate(finalResults))
    outputMap[v] = idx;

  // Process ops back-to-front.  For output ops, replaceOp() rewires external
  // uses and erases the op.  For internal ops, eraseOp() is safe because the
  // next chain op (the sole consumer of its result) was removed in the
  // previous iteration.
  for (int i = (int)ops.size() - 1; i >= 0; --i) {
    auto it = outputMap.find(ops[i]->getResult(0));
    if (it != outputMap.end())
      rewriter.replaceOp(ops[i], fusedOp.getOutputs()[it->second]);
    else
      rewriter.eraseOp(ops[i]);
  }
}

bool FusionOpChain::verifyAndRetrieveAttrs(ONNXFusedOp fusedOp) {
  if (!retrieveAttrs(fusedOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "FusionOpChain: retrieveAttrs failed for kind '"
               << fusedOp.getKind() << "'\n");
    return false;
  }
  if (!verify()) {
    LLVM_DEBUG(llvm::dbgs()
               << "FusionOpChain: verify failed for kind '"
               << fusedOp.getKind() << "'\n");
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// ExtLayoutTransformFusion — virtual method implementations
//===----------------------------------------------------------------------===//

bool ExtLayoutTransformFusion::detectIfBeneficial(
    const DimAnalysis *dimAnalysis, ONNXLayoutTransformOp startOp) {
  // Reset all fields.
  ops.clear();
  finalResults.clear();
  reshapeSplitAxis = -1;
  reshapeSplitFactor = 1;
  reshapeMergeAxis = -1;
  transposePattern = std::nullopt;
  dlf16ToF32 = false;
  finalLayout = std::nullopt;

  // Do not fire on ops that are already inside an ONNXFusedOp body.
  if (mlir::isa<ONNXFusedOp>(startOp->getParentOp()))
    return false;

  // ---- Step 1: validate and record the initial layout transform --------
  // Must be a ZTensor -> CPU conversion (no target layout means CPU).
  if (startOp.getTargetLayout().has_value())
    return false;
  Value inputData = startOp.getData();
  if (!isZTensor(inputData.getType()))
    return false;
  if (!supportedLayoutForCompilerGeneratedStickUnstick(inputData, /*nhwc=*/false))
    return false;
  if (!hasStaticInnermostDimMod(inputData, 64))
    return false;

  ops.push_back(startOp.getOperation());
  Value current = startOp.getOutput();

  // ---- Step 2: optional split reshape ----------------------------------
  bool reshapeMayBeMerge = false;
  if (auto splitReshape = singleUserOfType<ONNXReshapeOp>(current)) {
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
    if (auto transpose = singleUserOfType<ONNXTransposeOp>(current)) {
      auto perm = transpose.getPerm();
      if (!perm.has_value())
        return false; // default perm moves last dim — unsupported
      if (!transposeKeepsLastDim(perm.value()))
        return false;
      transposePattern = perm;
      ops.push_back(transpose.getOperation());
      current = transpose.getTransposed();
    }
  }

  // ---- Step 4: optional merge reshape ----------------------------------
  if (auto mergeReshape = singleUserOfType<ONNXReshapeOp>(current)) {
    if (detectMergeReshape(mergeReshape, reshapeMergeAxis, dimAnalysis)) {
      ops.push_back(mergeReshape.getOperation());
      current = mergeReshape.getReshaped();
    }
    // If detectMergeReshape fails here we just stop — no merge found.
  }

  // ---- Step 5: optional final layout transform or DLF16->F32 -----------
  if (auto finalLT = singleUserOfType<ONNXLayoutTransformOp>(current)) {
    auto layoutAttr = finalLT.getTargetLayout();
    if (!layoutAttr.has_value())
      return false; // second LT must target a ZTensor layout
    if (!supportedLayoutForCompilerGeneratedStickUnstick(
            finalLT.getOutput(), /*nhwc=*/false))
      return false;
    OpBuilder b(finalLT);
    finalLayout = getZTensorLayoutAttr(
        b, cast<ZTensorEncodingAttr>(layoutAttr.value()));
    ops.push_back(finalLT.getOperation());
    current = finalLT.getOutput();
  } else if (auto dlf = singleUserOfType<ZHighDLF16ToF32Op>(current)) {
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
    return false;

  return true;
}

void ExtLayoutTransformFusion::embedAttrs(ONNXFusedOp fusedOp) const {
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

bool ExtLayoutTransformFusion::retrieveAttrs(ONNXFusedOp fusedOp) {
  // hi alex, not sure if needed.
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

bool ExtLayoutTransformFusion::verify() const {
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
      LLVM_DEBUG(
          llvm::dbgs() << "ELT verify: split factor mismatch\n");
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

} // namespace zhigh
} // namespace onnx_mlir
