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
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXConcatFromSequenceOp::resultTypeInference() {
  // The output is a tensor whose element type matches the sequence elements.
  // At import time (before full shape inference) derive only the scalar dtype.
  Type scalarType = Builder(getContext()).getF32Type();
  if (auto seqType = mlir::dyn_cast<SeqType>(getInputSequence().getType())) {
    if (auto shapedElem = mlir::dyn_cast<ShapedType>(seqType.getElementType()))
      scalarType = shapedElem.getElementType();
  }
  return {UnrankedTensorType::get(scalarType)};
}

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

LogicalResult ONNXConcatFromSequenceOp::inferShapes(
    std::function<void(Region &)> /*doShapeInference*/) {
  auto seqType = mlir::dyn_cast<SeqType>(getInputSequence().getType());
  if (!seqType)
    return success();

  // Element type of the sequence must be a ranked tensor to infer the output.
  auto elemType = mlir::dyn_cast<RankedTensorType>(seqType.getElementType());
  if (!elemType)
    return success();

  // Sequence length must be statically known.
  const int64_t seqLen = seqType.getLength();
  if (seqLen == ShapedType::kDynamic)
    return success();
  if (seqLen < 0)
    return success();

  const int64_t rank = elemType.getRank();
  const int64_t axis = getAxis();
  const int64_t newAxis = getNewAxis();

  // Normalize negative axis.
  const int64_t axisNorm =
      axis < 0 ? axis + rank + (newAxis == 1 ? 1 : 0) : axis;

  SmallVector<int64_t> outShape;
  if (newAxis == 0) {
    // Concatenate along existing axis: output rank == elem rank.
    // Output axis dim = seqLen * elem_axis_dim (or dynamic if elem is dynamic).
    for (int64_t i = 0; i < rank; ++i) {
      if (i == axisNorm) {
        int64_t d = elemType.getDimSize(i);
        outShape.push_back(
            d == ShapedType::kDynamic ? ShapedType::kDynamic : seqLen * d);
      } else {
        outShape.push_back(elemType.getDimSize(i));
      }
    }
  } else {
    // Stack along a new axis: output rank == elem rank + 1.
    // New axis has size seqLen; all other dims come from the element type.
    for (int64_t i = 0; i < rank + 1; ++i) {
      if (i == axisNorm)
        outShape.push_back(seqLen);
      else
        outShape.push_back(elemType.getDimSize(i < axisNorm ? i : i - 1));
    }
  }

  getResult().setType(
      RankedTensorType::get(outShape, elemType.getElementType()));
  return success();
}
