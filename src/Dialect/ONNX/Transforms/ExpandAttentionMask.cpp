/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ExpandAttentionMask.cpp - ONNX NNPA Optimization ------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass to expand attention mask tensors in
// MatMul-Add-Softmax patterns to eliminate broadcasting in Add operations
// for NNPA compatibility.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "expand-attention-mask"

using namespace mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_EXPANDATTENTIONMASKPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"

namespace {

struct ExpandAttentionMaskPass
    : public impl::ExpandAttentionMaskPassBase<ExpandAttentionMaskPass> {
  using ExpandAttentionMaskPassBase::ExpandAttentionMaskPassBase;

  void runOnOperation() override;

private:
  DimAnalysis *dimAnalysis = nullptr;

  // Minimum uses threshold for mask expansion.
  static constexpr unsigned minUses = 8;

  // Analysis: collect eligible masks.
  void analyzeAttentionMasks();

  // Transformation: expand eligible masks.
  void expandEligibleMasks();

  // Check if Add is in MatMul-Add-Softmax pattern with broadcasting.
  bool isAttentionMaskAdd(ONNXAddOp addOp, DimAnalysis *dimAnalysis);

  // Data structures.
  DenseMap<Value, SmallVector<ONNXAddOp>> maskUsageMap;
  DenseSet<Value> eligibleMasks;
};

bool ExpandAttentionMaskPass::isAttentionMaskAdd(
    ONNXAddOp addOp, DimAnalysis *dimAnalysis) {
  // Fast path: check types first (cheapest check).
  auto firstType =
      mlir::dyn_cast<RankedTensorType>(addOp.getOperand(0).getType());
  auto secondType =
      mlir::dyn_cast<RankedTensorType>(addOp.getOperand(1).getType());

  if (!firstType || !secondType)
    return false;

  // Quick check: if shapes are identical, no broadcasting.
  // Use DimAnalysis::sameShape() to handle both static and dynamic dimensions.
  Value firstOperand = addOp.getOperand(0);
  Value secondOperand = addOp.getOperand(1);

  if (dimAnalysis->sameShape(firstOperand, secondOperand))
    return false; // No broadcasting.

  // Check pattern: MatMul -> Add -> Softmax.
  Operation *definingOp = firstOperand.getDefiningOp();
  if (!definingOp || !isa<ONNXMatMulOp>(definingOp))
    return false;

  // Check if output consumed by Softmax.
  Value result = addOp.getResult();
  for (Operation *user : result.getUsers()) {
    if (isa<ONNXSoftmaxOp>(user))
      return true;
  }

  return false;
}

void ExpandAttentionMaskPass::analyzeAttentionMasks() {
  func::FuncOp funcOp = getOperation();

  // Step 1: Collect all attention mask Add operations (single pass).
  funcOp.walk([&](ONNXAddOp addOp) {
    if (!isAttentionMaskAdd(addOp, dimAnalysis))
      return;

    // Get the mask (second operand being broadcast).
    Value maskTensor = addOp.getOperand(1);
    maskUsageMap[maskTensor].push_back(addOp);
  });

  // Step 2: Determine eligibility (efficient filtering).
  for (auto &entry : maskUsageMap) {
    Value maskTensor = entry.first;
    SmallVector<ONNXAddOp> &uses = entry.second;

    // Check minimum usage threshold.
    if (uses.size() < minUses)
      continue;

    // Check shape consistency using DimAnalysis (optimized).
    // All Add operations should have same first operand shape.
    bool allSame = true;
    Value firstAddOperand = uses[0].getOperand(0);

    for (size_t i = 1; i < uses.size(); ++i) {
      Value currentAddOperand = uses[i].getOperand(0);

      // Use DimAnalysis::sameShape() - handles both static and dynamic dims.
      if (!dimAnalysis->sameShape(firstAddOperand, currentAddOperand)) {
        allSame = false;
        break;
      }
    }

    if (allSame)
      eligibleMasks.insert(maskTensor);
  }
}

void ExpandAttentionMaskPass::expandEligibleMasks() {
  OpBuilder builder(&getContext());

  for (Value maskTensor : eligibleMasks) {
    SmallVector<ONNXAddOp> &uses = maskUsageMap[maskTensor];
    if (uses.empty())
      continue;

    // Get target shape from first use.
    ONNXAddOp firstAdd = uses[0];
    Value targetTensor = firstAdd.getOperand(0);
    Location loc = firstAdd.getLoc();

    // Build shape tensor dynamically using OnnxBuilder.
    builder.setInsertionPoint(firstAdd);
    OnnxBuilder onnxBuilder(builder, loc);

    auto targetType = mlir::dyn_cast<RankedTensorType>(targetTensor.getType());
    if (!targetType)
      continue;
    ArrayRef<int64_t> targetShape = targetType.getShape();

    SmallVector<Value> shapeParts;
    for (size_t i = 0; i < targetShape.size(); ++i) {
      if (targetShape[i] >= 0) {
        // Static dimension: create constant.
        Value constVal =
            onnxBuilder.constant(builder.getI64TensorAttr({targetShape[i]}));
        shapeParts.push_back(constVal);
      } else {
        // Dynamic dimension: use onnx.Dim.
        Value dimVal = onnxBuilder.dim(targetTensor, i);
        shapeParts.push_back(dimVal);
      }
    }

    // Concatenate shape parts to build shape tensor.
    // Determine output type for concat: tensor<NxI64> where N is rank.
    RankedTensorType shapeType = RankedTensorType::get(
        {static_cast<int64_t>(targetShape.size())}, builder.getI64Type());
    Value shapeValue = onnxBuilder.concat(shapeType, shapeParts, /*axis=*/0);

    // Create Expand operation.
    // Output type is same as target type.
    Value expandedMask = onnxBuilder.expand(targetType, maskTensor, shapeValue);

    // Replace all uses in Add operations.
    for (ONNXAddOp addOp : uses) {
      addOp.setOperand(1, expandedMask);
    }
  }
}

void ExpandAttentionMaskPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Initialize DimAnalysis.
  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  DimAnalysis analysis(module);
  analysis.analyze();
  dimAnalysis = &analysis;

  // Run analysis phase.
  analyzeAttentionMasks();

  // Early exit if no eligible masks.
  if (eligibleMasks.empty())
    return;

  // Run transformation phase.
  expandEligibleMasks();
}

} // namespace

std::unique_ptr<Pass> createExpandAttentionMaskPass() {
  return std::make_unique<ExpandAttentionMaskPass>();
}

} // namespace onnx_mlir
