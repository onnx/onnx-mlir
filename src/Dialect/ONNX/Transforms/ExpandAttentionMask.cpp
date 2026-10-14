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
// MatMul-[Mul]-Add-Softmax patterns to eliminate broadcasting in Add and Mul
// operations for NNPA compatibility. Mul is optional.
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
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "expand-attention-mask"

using namespace mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_EXPANDATTENTIONMASKPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"

namespace {

// Structure to hold detected attention pattern information.
struct AttentionPattern {
  ONNXMatMulOp matmulOp;
  ONNXMulOp mulOp; // Optional, can be nullptr.
  ONNXAddOp addOp;
  ONNXSoftmaxOp softmaxOp;
  Value scaleOperand; // For Mul (if present) - attention scale factor.
  Value maskOperand;  // For Add.
  bool hasMul;
};

struct ExpandAttentionMaskPass
    : public impl::ExpandAttentionMaskPassBase<ExpandAttentionMaskPass> {
  using ExpandAttentionMaskPassBase::ExpandAttentionMaskPassBase;

  void runOnOperation() override;

private:
  DimAnalysis *dimAnalysis = nullptr;

  // Minimum uses threshold for mask expansion.
  static constexpr unsigned minUses = 4;

  // Analysis: collect eligible masks and scales.
  void analyzeAttentionMasks();

  // Transformation: expand eligible masks and scales.
  void expandEligibleOperands();

  // Detect attention pattern (MatMul-[Mul]-Add-Softmax).
  std::optional<AttentionPattern> detectAttentionPattern(
      ONNXAddOp addOp, DimAnalysis *dimAnalysis);

  // Helper: Create expanded operand from source operand to target shape.
  Value createExpandedOperand(OpBuilder &builder, Location loc,
      Value operandToExpand, Value targetTensor);

  // Data structures.
  DenseMap<Value, SmallVector<AttentionPattern>> patternsByMask;
  DenseMap<Value, SmallVector<AttentionPattern>> patternsByScale;
  DenseSet<Value> eligibleMasks;
  DenseSet<Value> eligibleScales;
};

std::optional<AttentionPattern> ExpandAttentionMaskPass::detectAttentionPattern(
    ONNXAddOp addOp, DimAnalysis *dimAnalysis) {
  // Fast path: check types first (cheapest check).
  auto firstType =
      mlir::dyn_cast<RankedTensorType>(addOp.getOperand(0).getType());
  auto secondType =
      mlir::dyn_cast<RankedTensorType>(addOp.getOperand(1).getType());

  if (!firstType || !secondType)
    return std::nullopt;

  // Quick check: if shapes are identical, no broadcasting.
  Value firstOperand = addOp.getOperand(0);
  Value secondOperand = addOp.getOperand(1);

  if (dimAnalysis->sameShape(firstOperand, secondOperand))
    return std::nullopt; // No broadcasting.

  // Initialize pattern structure.
  AttentionPattern pattern;
  pattern.addOp = addOp;
  pattern.maskOperand = secondOperand;
  pattern.hasMul = false;
  pattern.mulOp = nullptr;
  pattern.scaleOperand = nullptr;

  // Check for optional Mul operation using definedBy<> template.
  if (definedBy<ONNXMulOp>(firstOperand)) {
    ONNXMulOp mulOp = firstOperand.getDefiningOp<ONNXMulOp>();

    // Mul is commutative: find which operand is MatMul and which is scale.
    Value mulOp0 = mulOp.getOperand(0);
    Value mulOp1 = mulOp.getOperand(1);
    Value matmulValue = nullptr;
    Value scaleCandidate = nullptr;

    if (definedBy<ONNXMatMulOp>(mulOp0)) {
      matmulValue = mulOp0;
      scaleCandidate = mulOp1;
    } else if (definedBy<ONNXMatMulOp>(mulOp1)) {
      matmulValue = mulOp1;
      scaleCandidate = mulOp0;
    }

    if (matmulValue && scaleCandidate && isScalarTensor(scaleCandidate)) {
      // Valid attention scaling pattern: MatMul * scalar_scale.
      pattern.matmulOp = matmulValue.getDefiningOp<ONNXMatMulOp>();
      pattern.mulOp = mulOp;
      pattern.scaleOperand = scaleCandidate;
      pattern.hasMul = true;
    } else if (matmulValue) {
      // Mul present but scale is not scalar — treat as MatMul->Add (mask only).
      pattern.matmulOp = matmulValue.getDefiningOp<ONNXMatMulOp>();
    } else {
      return std::nullopt;
    }
  } else if (definedBy<ONNXMatMulOp>(firstOperand)) {
    // Direct MatMul->Add pattern (no Mul).
    pattern.matmulOp = firstOperand.getDefiningOp<ONNXMatMulOp>();
  } else {
    return std::nullopt;
  }

  // Check if Add output is consumed by Softmax.
  Value result = addOp.getResult();
  for (Operation *user : result.getUsers()) {
    if (auto softmaxOp = dyn_cast<ONNXSoftmaxOp>(user)) {
      pattern.softmaxOp = softmaxOp;
      return pattern;
    }
  }

  return std::nullopt;
}

Value ExpandAttentionMaskPass::createExpandedOperand(OpBuilder &builder,
    Location loc, Value operandToExpand, Value targetTensor) {
  OnnxBuilder onnxBuilder(builder, loc);
  auto targetType = cast<RankedTensorType>(targetTensor.getType());
  ArrayRef<int64_t> targetShape = targetType.getShape();

  // Build shape tensor dynamically.
  SmallVector<Value> shapeParts;
  for (size_t i = 0; i < targetShape.size(); ++i) {
    if (ShapedType::isDynamic(targetShape[i])) {
      // Dynamic dimension - use onnxBuilder.dim().
      Value dimVal = onnxBuilder.dim(targetTensor, i);
      shapeParts.push_back(dimVal);
    } else {
      // Static dimension - use onnxBuilder.constant().
      Value constVal =
          onnxBuilder.constant(builder.getI64TensorAttr({targetShape[i]}));
      shapeParts.push_back(constVal);
    }
  }

  RankedTensorType shapeType = RankedTensorType::get(
      {static_cast<int64_t>(targetShape.size())}, builder.getI64Type());
  Value shapeValue = onnxBuilder.concat(shapeType, shapeParts, /*axis=*/0);

  // Create and return expanded operand.
  return onnxBuilder.expand(targetType, operandToExpand, shapeValue);
}

void ExpandAttentionMaskPass::analyzeAttentionMasks() {
  func::FuncOp funcOp = getOperation();

  // Step 1: Collect all attention patterns (single pass).
  funcOp.walk([&](ONNXAddOp addOp) {
    auto pattern = detectAttentionPattern(addOp, dimAnalysis);
    if (!pattern)
      return;

    // Track mask operand.
    Value mask = pattern->maskOperand;
    patternsByMask[mask].push_back(*pattern);

    // Track scale operand if Mul present.
    if (pattern->hasMul) {
      Value scale = pattern->scaleOperand;
      patternsByScale[scale].push_back(*pattern);
    }
  });

  // Step 2: Determine eligibility for masks.
  for (auto &entry : patternsByMask) {
    Value maskTensor = entry.first;
    SmallVector<AttentionPattern> &patterns = entry.second;

    // Check minimum usage threshold.
    if (patterns.size() < minUses)
      continue;

    // Check shape consistency using DimAnalysis.
    // All patterns should have same MatMul output shape.
    bool allSame = true;
    Value firstMatMulOutput = patterns[0].matmulOp.getResult();

    for (size_t i = 1; i < patterns.size(); ++i) {
      Value currentMatMulOutput = patterns[i].matmulOp.getResult();

      if (!dimAnalysis->sameShape(firstMatMulOutput, currentMatMulOutput)) {
        allSame = false;
        break;
      }
    }

    if (allSame)
      eligibleMasks.insert(maskTensor);
  }

  // Step 3: Determine eligibility for scales.
  for (auto &entry : patternsByScale) {
    Value scaleTensor = entry.first;
    SmallVector<AttentionPattern> &patterns = entry.second;

    // Check minimum usage threshold.
    if (patterns.size() < minUses)
      continue;

    // Check shape consistency.
    bool allSame = true;
    Value firstMatMulOutput = patterns[0].matmulOp.getResult();

    for (size_t i = 1; i < patterns.size(); ++i) {
      Value currentMatMulOutput = patterns[i].matmulOp.getResult();

      if (!dimAnalysis->sameShape(firstMatMulOutput, currentMatMulOutput)) {
        allSame = false;
        break;
      }
    }

    if (allSame)
      eligibleScales.insert(scaleTensor);
  }
}

void ExpandAttentionMaskPass::expandEligibleOperands() {
  OpBuilder builder(&getContext());

  // Expand eligible masks.
  for (Value maskTensor : eligibleMasks) {
    SmallVector<AttentionPattern> &patterns = patternsByMask[maskTensor];
    if (patterns.empty())
      continue;

    // Get target shape from first pattern's MatMul output.
    AttentionPattern &firstPattern = patterns[0];
    Value targetTensor = firstPattern.matmulOp.getResult();
    Location loc = firstPattern.addOp.getLoc();

    // Insert right after the first MatMul whose result the Dim ops reference.
    // This dominates all Add ops in the patterns (MatMul feeds into Add),
    // and the mask (a block arg or earlier constant) already dominates it.
    builder.setInsertionPointAfter(firstPattern.matmulOp);

    // Create expanded mask using helper function.
    Value expandedMask =
        createExpandedOperand(builder, loc, maskTensor, targetTensor);

    // Replace mask operand in all Add operations.
    for (AttentionPattern &pattern : patterns) {
      pattern.addOp.setOperand(1, expandedMask);
    }
  }

  // Expand eligible scales.
  for (Value scaleTensor : eligibleScales) {
    SmallVector<AttentionPattern> &patterns = patternsByScale[scaleTensor];
    if (patterns.empty())
      continue;

    // Get target shape from first pattern's MatMul output.
    AttentionPattern &firstPattern = patterns[0];
    Value targetTensor = firstPattern.matmulOp.getResult();
    Location loc = firstPattern.mulOp.getLoc();

    // Insert right after the first MatMul whose result the Dim ops reference.
    builder.setInsertionPointAfter(firstPattern.matmulOp);

    // Create expanded scale using helper function.
    Value expandedScale =
        createExpandedOperand(builder, loc, scaleTensor, targetTensor);

    // Replace scale operand in all Mul operations.
    for (AttentionPattern &pattern : patterns) {
      pattern.mulOp.setOperand(1, expandedScale);
    }
  }
}

void ExpandAttentionMaskPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Clear state from any prior invocation.
  patternsByMask.clear();
  patternsByScale.clear();
  eligibleMasks.clear();
  eligibleScales.clear();

  // Initialize DimAnalysis.
  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  DimAnalysis analysis(module);
  analysis.analyze();
  dimAnalysis = &analysis;

  // Run analysis phase.
  analyzeAttentionMasks();

  // Early exit if no eligible masks or scales.
  if (eligibleMasks.empty() && eligibleScales.empty())
    return;

  // Run transformation phase.
  expandEligibleOperands();
}

} // namespace

std::unique_ptr<Pass> createExpandAttentionMaskPass() {
  return std::make_unique<ExpandAttentionMaskPass>();
}

} // namespace onnx_mlir
