/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ReverseSequence.cpp - ONNX Operations -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect ReverseSequence operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXReverseSequenceOpShapeHelper::computeShape() {
  // Get info about input data operand.
  ONNXReverseSequenceOpAdaptor operandAdaptor(operands);
  Value input = operandAdaptor.getInput();
  return setOutputDimsFromOperand(input);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXReverseSequenceOp::verify() {
  ONNXReverseSequenceOpAdaptor operandAdaptor =
      ONNXReverseSequenceOpAdaptor(*this);

  auto sequence_lensTy = mlir::dyn_cast<RankedTensorType>(
      operandAdaptor.getSequenceLens().getType());
  auto inputTy =
      mlir::dyn_cast<RankedTensorType>(operandAdaptor.getInput().getType());

  // sequence_lens should be 1D tensor
  if (sequence_lensTy) {
    if (sequence_lensTy.getRank() != 1)
      return emitOpError("sequence_lens of ReverseSequnce should be 1D tensor");
  }

  if (inputTy) {
    if (inputTy.getRank() < 2)
      return emitOpError(
          "input of Reversesequence should be 2D or higher rank tensor");
  }

  if (sequence_lensTy && inputTy) {
    int64_t batchAxis = getBatchAxis();
    if (!sequence_lensTy.isDynamicDim(0) && !inputTy.isDynamicDim(batchAxis)) {
      if (sequence_lensTy.getShape()[0] != inputTy.getShape()[batchAxis]) {
        return emitOpError("Length of sequence_lens should match the sizeof  "
                           "batch axis of the input");
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXReverseSequenceOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getInput().getType()).getElementType();
  ONNXReverseSequenceOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXReverseSequenceOp>;
} // namespace onnx_mlir
