/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ClampSoftmaxMask.cpp - Clamp mask constants of Softmax -------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass that raises large negative constants (e.g.
// -FLT_MAX or -inf), which only reach a Softmax input through value-moving
// ops, to getAttentionMaskFillValue() (-1e9 for f32). Such constants are the
// usual way to build additive attention masks. In reduced precision (f16, or
// dlfloat16 on NNPA) they overflow when added to the scores, and fully masked
// rows become NaN. See ClampSoftmaxMaskPass in Passes.td for the details.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "clamp-softmax-mask"

using namespace mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_CLAMPSOFTMAXMASKPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"

namespace {

// Largest constant that is inspected element by element. Masks built by
// models are scalars or small tensors; this only bounds the cost of a large
// constant that happens to be reached.
static constexpr int64_t maxConstantSize = 1 << 22;

// Return the operand indices of `op` through which a value reaching a Softmax
// input is followed backward. Each listed op cannot make its output more
// negative than its listed inputs, so raising a very negative input to the
// fill value only changes outputs that are themselves below it. Return false
// if `op` is not followed.
bool getFollowedOperands(Operation *op, SmallVectorImpl<unsigned> &indices) {
  // Ops that only move or broadcast their data operand.
  if (isa<ONNXExpandOp, ONNXReshapeOp, ONNXUnsqueezeOp, ONNXSqueezeOp,
          ONNXTransposeOp, ONNXSliceOp, ONNXGatherOp, ONNXIdentityOp>(op)) {
    indices.push_back(0);
    return true;
  }
  // A Cast that keeps the element type keeps the values.
  if (auto castOp = dyn_cast<ONNXCastOp>(op)) {
    if (getElementType(castOp.getInput().getType()) !=
        getElementType(castOp.getOutput().getType()))
      return false;
    indices.push_back(0);
    return true;
  }
  // Where selects between its two value inputs (not the condition).
  if (isa<ONNXWhereOp>(op)) {
    indices.append({1, 2});
    return true;
  }
  // Sub is monotone in its first operand only.
  if (isa<ONNXSubOp>(op)) {
    indices.push_back(0);
    return true;
  }
  // Ops monotone in every operand: combinations of masks and scores.
  if (isa<ONNXAddOp, ONNXMinOp, ONNXMaxOp, ONNXConcatOp>(op)) {
    for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i)
      indices.push_back(i);
    return true;
  }
  return false;
}

// Convert `f` to double (exact for f16, bf16, f32 and f64).
double toDouble(APFloat f) {
  bool losesInfo;
  f.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  return f.convertToDouble();
}

struct ClampSoftmaxMaskPass
    : public impl::ClampSoftmaxMaskPassBase<ClampSoftmaxMaskPass> {
  using ClampSoftmaxMaskPassBase::ClampSoftmaxMaskPassBase;

  void runOnOperation() override;

private:
  // Uses through which the backward walk went (sink operands included).
  DenseSet<OpOperand *> followedUses;
  // Sink operands: the Softmax inputs and onnx.Attention attn_mask inputs.
  DenseSet<OpOperand *> sinkUses;
  // Memoized result of isConfined().
  DenseMap<Value, bool> confined;

  // Walk backward from a sink operand, collecting reached constants.
  void walkFromSink(OpOperand &sink, SetVector<ONNXConstantOp> &constants);

  // True if every use of `v` was followed by the walk and leads, through ops
  // whose results are confined too, only to sinks. A constant is changed only
  // when its result is confined, so no other computation sees the new value.
  bool isConfined(Value v);

  // Raise the elements of `constOp` below its fill value to that value.
  // Return false (and leave it unchanged) when there is nothing to do.
  bool clampConstant(ONNXConstantOp constOp);
};

void ClampSoftmaxMaskPass::walkFromSink(
    OpOperand &sink, SetVector<ONNXConstantOp> &constants) {
  SmallVector<OpOperand *> worklist{&sink};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    OpOperand *use = worklist.pop_back_val();
    followedUses.insert(use);
    Value v = use->get();
    if (!visited.insert(v).second)
      continue;
    Operation *def = v.getDefiningOp();
    if (!def)
      continue; // Function argument: nothing to clamp.
    if (auto constOp = dyn_cast<ONNXConstantOp>(def)) {
      constants.insert(constOp);
      continue;
    }
    SmallVector<unsigned> indices;
    if (!getFollowedOperands(def, indices))
      continue;
    for (unsigned i : indices)
      worklist.push_back(&def->getOpOperand(i));
  }
}

bool ClampSoftmaxMaskPass::isConfined(Value v) {
  auto it = confined.find(v);
  if (it != confined.end())
    return it->second;
  // Assume false while visiting, which also ends any cycle conservatively.
  confined[v] = false;
  bool result = true;
  for (OpOperand &use : v.getUses()) {
    if (!followedUses.contains(&use)) {
      LLVM_DEBUG(llvm::dbgs()
                 << DEBUG_TYPE << ":   value also used by "
                 << use.getOwner()->getName() << " (operand "
                 << use.getOperandNumber() << "), not a mask use\n");
      result = false;
      break;
    }
    if (sinkUses.contains(&use))
      continue;
    // A followed use that is not a sink: the user forwards the value.
    for (Value r : use.getOwner()->getResults()) {
      if (!isConfined(r)) {
        result = false;
        break;
      }
    }
    if (!result)
      break;
  }
  confined[v] = result;
  return result;
}

bool ClampSoftmaxMaskPass::clampConstant(ONNXConstantOp constOp) {
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());
  if (!elementsAttr)
    return false;
  auto type = cast<ShapedType>(elementsAttr.getType());
  auto floatType = dyn_cast<FloatType>(type.getElementType());
  if (!floatType || !type.hasStaticShape())
    return false;
  if (type.getNumElements() > maxConstantSize) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << ": skip " << constOp.getLoc() << ": "
               << type.getNumElements() << " elements, too large to inspect\n");
    return false;
  }

  double fill = getAttentionMaskFillValue(floatType);
  APFloat fillF(fill);
  bool losesInfo;
  fillF.convert(
      floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);

  SmallVector<APFloat> values;
  values.reserve(type.getNumElements());
  int64_t numClamped = 0;
  double minValue = 0;
  for (APFloat f : elementsAttr.getValues<APFloat>()) {
    double d = toDouble(f);
    if (d < fill) { // False for NaN, which is left unchanged.
      minValue = std::min(minValue, d);
      ++numClamped;
      values.push_back(fillF);
    } else {
      values.push_back(f);
    }
  }
  if (numClamped == 0)
    return false;

  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": clamp " << constOp.getLoc()
                          << " (" << type << "): " << numClamped << " of "
                          << type.getNumElements() << " values, min "
                          << minValue << " -> " << fill << "\n");
  constOp.setValueAttr(DenseElementsAttr::get(type, values));
  return true;
}

void ClampSoftmaxMaskPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  followedUses.clear();
  sinkUses.clear();
  confined.clear();

  // Collect the sink operands.
  SmallVector<OpOperand *> sinks;
  funcOp.walk([&](Operation *op) {
    if (isa<ONNXSoftmaxOp, ONNXSoftmaxV11Op>(op)) {
      sinks.push_back(&op->getOpOperand(0));
    } else if (auto attentionOp = dyn_cast<ONNXAttentionOp>(op)) {
      Value mask = attentionOp.getAttnMask();
      if (!isNoneValue(mask) && isa<FloatType>(getElementType(mask.getType())))
        sinks.push_back(&attentionOp.getAttnMaskMutable());
    }
  });
  if (sinks.empty())
    return;
  for (OpOperand *sink : sinks)
    sinkUses.insert(sink);

  // Walk backward from every sink first, so that a value shared by several
  // sinks (e.g. one mask used by every layer) has all its uses followed
  // before it is checked.
  SetVector<ONNXConstantOp> constants;
  for (OpOperand *sink : sinks)
    walkFromSink(*sink, constants);
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": " << funcOp.getName() << ": "
                          << sinks.size() << " sink(s), " << constants.size()
                          << " constant(s) reached\n");

  for (ONNXConstantOp constOp : constants) {
    auto floatType = dyn_cast<FloatType>(getElementType(constOp.getType()));
    if (!floatType)
      continue;
    // Check the confinement only for constants that would change.
    double fill = getAttentionMaskFillValue(floatType);
    auto elementsAttr = dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());
    if (!elementsAttr || elementsAttr.getNumElements() > maxConstantSize)
      continue;
    bool hasLow = llvm::any_of(elementsAttr.getValues<APFloat>(),
        [&](const APFloat &f) { return toDouble(f) < fill; });
    if (!hasLow)
      continue;
    if (!isConfined(constOp.getResult())) {
      LLVM_DEBUG(llvm::dbgs()
                 << DEBUG_TYPE << ": keep " << constOp.getLoc()
                 << ": it also reaches ops other than a Softmax mask\n");
      continue;
    }
    clampConstant(constOp);
  }
}

} // namespace

std::unique_ptr<Pass> createClampSoftmaxMaskPass() {
  return std::make_unique<ClampSoftmaxMaskPass>();
}

} // namespace onnx_mlir
