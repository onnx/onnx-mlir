/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ConcatFromSequence.cpp - ONNX Operations ----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect ConcatFromSequence operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXConcatFromSequenceOp::verify() {
  ONNXConcatFromSequenceOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(operandAdaptor.getInputSequence()))
    return success(); // Won't be able to do any checking at this stage.

  Value inputSequence = operandAdaptor.getInputSequence();
  assert(mlir::isa<SeqType>(inputSequence.getType()) &&
         "Incorrect type for a sequence");
  auto seqType = mlir::cast<SeqType>(inputSequence.getType());
  auto elemType = mlir::cast<ShapedType>(seqType.getElementType());
  int64_t rank = elemType.getShape().size();
  int64_t axisIndex = getAxis();
  int64_t newAxisIndex = getNewAxis();

  // axis attribute must be in the range [-r,r-1], where r = rank(inputs).
  // When `new_axis` is 1, accepted range is [-r-1,r].
  if (newAxisIndex == 1) {
    if (axisIndex < (-rank - 1) || axisIndex > rank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axisIndex,
          onnx_mlir::Diagnostic::Range<int64_t>(-rank - 1, rank));
  } else {
    if (axisIndex < -rank || axisIndex >= rank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axisIndex,
          onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

// TODO
