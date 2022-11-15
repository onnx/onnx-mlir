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
    : op(op), operands(operands), ieBuilder(ieBuilder), scope(scope),
      outputsDims(), ownScope(scope == nullptr) {
  assert(op && "Expecting a valid operation pointer");
  assert(ieBuilder && "Expecting a valid index expression builder");
  if (ownScope)
    scope = new IndexExprScope(nullptr, op->getLoc());
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
  uint64_t rank = ieBuilder.getShapeRank(operands[0]);
  DimsExpr outputDims;
  for (uint64_t i = 0; i < rank; ++i)
    outputDims.emplace_back(ieBuilder.getShapeAsDim(operands[0], i));
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir
