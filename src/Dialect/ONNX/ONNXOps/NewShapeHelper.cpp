/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- NewShapeHelper.hpp - help for shapes ---------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>

#define DEBUG_TYPE "shape-helper"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

/// Refine `inferredDims` using the output's shape if possible. For example,
/// replacing a dynamic dim in `inferredDims` by a static dim in the output's
/// shape.
static void refineDims(DimsExpr &inferredDims, Value output) {
  // Nothing to do if the output is unranked.
  if (!isRankedShapedType(output.getType()))
    return;

  llvm::ArrayRef<int64_t> existingDims = getShape(output.getType());
  // Do not handle the case of scalar tensor whose type can be tensor<f32>
  // or tensor<1xf32>. Just use the inferredShape in this case.
  if (existingDims.size() < 1 || inferredDims.size() < 1)
    return;

  assert((existingDims.size() == inferredDims.size()) &&
         "Inferred shape and existing shape are inconsistent in the number "
         "of elements");

  // Try to update inferredDim if existingDim is static.
  for (unsigned i = 0; i < existingDims.size(); ++i) {
    // existingDim is dynamic, nothing to do.
    if (existingDims[i] == -1)
      continue;

    // inferredDim is unknown at shape inference: update it.
    if (inferredDims[i].isQuestionmark()) {
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
      continue;
    }
    // inferredDim is unknown at lowering: use existing dim for efficiency.
    if (!inferredDims[i].isLiteral()) {
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
      continue;
    }
    // inferredDim is different from existingDim. Believe in existingDim.
    if (inferredDims[i].isLiteral() &&
        (existingDims[i] != inferredDims[i].getLiteral())) {
      // Warning for users.
      llvm::outs() << "Warning: [Shape inference] the inferred dim ("
                   << inferredDims[i].getLiteral()
                   << ") is different from the existing dim ("
                   << existingDims[i] << "). Use the existing dim instead.\n";
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
    }
  }
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

template <class OP>
NewONNXOpShapeHelper<OP>::NewONNXOpShapeHelper(OP *op, ValueRange operands,
    IndexExprBuilder *ieBuilder, IndexExprScope *scope)
    : op(op), operands(operands), createIE(ieBuilder), scope(scope),
      outputsDims(), ownScope(scope == nullptr) {
  assert(op && "Expecting a valid operation pointer");
  assert(createIE && "Expecting a valid index expression builder");
  if (ownScope)
    scope = new IndexExprScope(createIE->getBuilderPtr(), createIE->getLoc());
  setNumberOfOutputs(op->getNumResults());
}

template <class OP>
void NewONNXOpShapeHelper<OP>::setOutputDims(DimsExpr inferredDims, int n) {
  outputsDims[n] = inferredDims;
  // Try to refine outputsDims[n] using the output's shape if possible. For
  // example, replacing a dynamic dim in outputsDims[n] by a static dim in the
  // output's shape.
  Value output = getOutput(n);
  refineDims(outputsDims[n], output);
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper for Generic Unary Elementwise Operations
//===----------------------------------------------------------------------===//

LogicalResult NewONNXGenericOpUnaryShapeHelper::computeShape() {
  // Output and input have the same shape. Just pass the input shape to the
  // output.
  uint64_t rank = createIE->getShapeRank(operands[0]);
  DimsExpr outputDims;
  for (uint64_t i = 0; i < rank; ++i)
    outputDims.emplace_back(createIE->getShapeAsDim(operands[0], i));
  setOutputDims(outputDims);
  return success();
}

<<<<<<< HEAD
//===----------------------------------------------------------------------===//
// ONNX Broadcast Op Shape Helper
//===----------------------------------------------------------------------===//

template <class OP>
LogicalResult NewONNXOpBroadcastedShapeHelper<OP>::computeShape() {
  // if additionalOperand is not used, we expect a zero-sized vector.
  // A temporary IndexExpr vector for the output.
  DimsExpr dimsExpr;
  int64_t numOfInputs = this->operands.size();

  // Compute rank of the output. Rank of the output is the maximum rank of all
  // operands.
  int64_t additionalOperRank =
      additionalOperand ? -1 : additionalOperand->size();
  outputRank = additionalOperRank;
  for (int64_t i = 0; i < numOfInputs; ++i)
    outputRank = std::max(outputRank, this->createIE.getTypeRank(operands[i]));
  assert(outputRank >= 0 && "expected a scalar rank at the very least");
  dimsExpr.resize(outputRank);

  // Prepare dims for every input. Prepend 1s if the input's shape has smaller
  // rank, so that all the shapes have the same rank.
  LiteralIndexExpr one(1);
  for (int64_t i = 0; i < numOfInputs; ++i) {
    int64_t r = createIE.getTypeRank(operands[i]);
// Prepend 1s.
#if 1
    DimsExpr dims(outputRank - r, one);
#else
    DimsExpr dims;
    for (int64_t k = 0; k < outputRank - r; ++k)
      dims.emplace_back(one);
#endif
    // Get from the input.
    for (int64_t k = 0; k < r; ++k)
      dims.emplace_back(createIE.getShapeAsDim(operands[i], k));
    inputsDims.emplace_back(dims);
  }
  // Handle the additional operand here.
  if (additionalOperRank>0) {
    DimsExpr dims(outputRank - additionalOperRank, one);
    for (int64_t k = 0; k < additionalOperRank; ++k)
      dims.emplace_back((*additionalOperand)[k]);
    inputsDims.emplace_back(dims);
    numOfInputs++;
  }

  // Initialize the output with the first operand.
  dimsExpr = inputsDims[0];

  // Note on IndexExpr. When we are not allowed to generate code, QuestionMark
  // stands for anything but a literal. When we are allowed to generate code,
  // there should be no more QuestionMarks as we are allowed to generate
  // affine/symbols/dims/non-affine expressions. Since this code predominantly
  // runs when we can gen code (as it actually does gen max ops), we should
  // use !isLiteral() for anything that is runtime. The comments were left
  // unchanged.

  //  Now compute each broadcasted dimension for the output. Folding over the
  //  other operands along the current dimension index.
  for (int64_t i = 1; i < numOfInputs; ++i) {
    for (int64_t j = 0; j < outputRank; ++j) {
      // Set the output dimension based on the two dimension values.
      // Dimension value can be one of 1, QuestionMark, LiteralNot1.
      IndexExpr currentDimExpr = dimsExpr[j];
      IndexExpr nextDimExpr = inputsDims[i][j];
      // Case: 1 - *.
      if (currentDimExpr.isLiteralAndIdenticalTo(1)) {
        if (!isUniBroadcasting && !isNoBroadcasting)
          dimsExpr[j] = nextDimExpr;
        continue;
      }
      // Case: LiteralNot1 - *.
      if (currentDimExpr.isLiteralAndDifferentThan(1)) {
        // LiteralNot1 - LiteralNot1 => keep unchanged with verifying.
        if (nextDimExpr.isLiteralAndDifferentThan(1) &&
            !currentDimExpr.isLiteralAndIdenticalTo(nextDimExpr))
          return failure();
        // Case: LiteralNot1 - (QuestionMark or 1) => Keep unchanged without
        // verifying.
        continue;
      }
      // Case: QuestionMark - 1 => keep unchanged.
      if (!currentDimExpr.isLiteral() &&
          nextDimExpr.isLiteralAndIdenticalTo(1)) {
        continue;
      }
      // Case QuestionMark - LiteralNot1 => set to LiteralNot1 without
      // verifying.
      if (!currentDimExpr.isLiteral() &&
          nextDimExpr.isLiteralAndDifferentThan(1)) {
        dimsExpr[j] = nextDimExpr;
        continue;
      }
      // Case: QuestionMark - QuestionMark
      if (!isUniBroadcasting) {
        dimsExpr[j] = IndexExpr::max(currentDimExpr, nextDimExpr);
      }
    }
  }
  // Set the final output.
  ONNXOpShapeHelper<OP>::setOutputDims(dimsExpr);
  return success();
}

template <class OP>
LogicalResult NewONNXOpBroadcastedShapeHelper<OP>::GetAccessExprs(Value operand,
    uint64_t operandIndex, const SmallVectorImpl<IndexExpr> &outputAccessExprs,
    SmallVectorImpl<IndexExpr> &operandAccessExprs) {
  if (isNoBroadcasting || (isUniBroadcasting && operandIndex == 0)) {
    for (IndexExpr ie : outputAccessExprs)
      operandAccessExprs.emplace_back(ie);
    return success();
  }

  auto operandRank = operand.getType().cast<ShapedType>().getRank();
  for (decltype(operandRank) i = 0; i < operandRank; ++i) {
    // Shape helper may pretend 1s, thus adjust dimension index accordingly.
    auto dimIndex = outputRank - operandRank + i;
    SymbolIndexExpr dim(inputsDims[operandIndex][dimIndex]);

    // Compute access index based on broadcasting rules.
    // If all other operand dims are 1, just use the output access index.
    // Otherwise, emit a select op.
    bool allOtherInputDimsAreOne = true;
    for (unsigned int i = 0; i < inputsDims.size(); ++i) {
      if (i == operandIndex)
        continue;
      IndexExpr dim = inputsDims[i][dimIndex];
      if (!dim.isLiteralAndIdenticalTo(1)) {
        allOtherInputDimsAreOne = false;
        break;
      }
    }
    if (allOtherInputDimsAreOne) {
      operandAccessExprs.emplace_back(outputAccessExprs[dimIndex]);
    } else {
      operandAccessExprs.emplace_back(
          IndexExpr::select(dim > 1, outputAccessExprs[dimIndex], 0));
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Template instantiation (last).
//===----------------------------------------------------------------------===//

=======
>>>>>>> shapehelper-reorg-v2
template struct NewONNXOpShapeHelper<Operation>;

} // namespace onnx_mlir
