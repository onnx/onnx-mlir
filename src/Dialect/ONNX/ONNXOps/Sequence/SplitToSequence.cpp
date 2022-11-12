/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ SplitToSequence.cpp - ONNX Operations -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect SplitToSequence operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXSplitToSequenceOp::verify() {
  Value inputValue = input();
  if (!hasShapeAndRank(inputValue))
    return success(); // Won't be able to do any checking at this stage.

  auto inputType = inputValue.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();

  int64_t axisIndex = axis();
  // axis attribute must be in the range [-r,r-1], where r = rank(input).
  if (axisIndex < -inputRank || axisIndex >= inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));
  if (axisIndex < 0)
    axisIndex += inputRank;

  Value splitValue = split();
  if (isFromNone(splitValue)) {
    // since split is not specified, check the keepdims attribute
    int64_t keep = keepdims();
    // keepdims must be 0 or 1
    if (keep < 0 || keep > 1)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "keepdims", keep,
          onnx_mlir::Diagnostic::Range<int64_t>(0, 1));
    return success();
  }
  auto splitType = splitValue.getType().cast<ShapedType>();
  ArrayRef<int64_t> splitShape = splitType.getShape();
  int64_t splitRank = splitShape.size();
  if (splitRank > 1)
    return emitOpError() << ": split has rank " << splitRank << " > 1";
  if (ElementsAttr entries = getElementAttributeFromONNXValue(splitValue)) {
    if (splitRank == 0) {
      auto scalar = getScalarValue<int64_t>(entries, splitType);
      if (scalar <= 0)
        return emitOpError() << ": split scalar " << scalar << " <= 0";
    } else {
      int64_t sum = 0;
      for (auto entry : entries.getValues<IntegerAttr>()) {
        int64_t i = entry.getInt();
        if (i < 0)
          return emitOpError() << ": split tensor has entry " << i << " < 0";
        sum += i;
      }
      int64_t dimSize = inputShape[axisIndex];
      if (!ShapedType::isDynamic(dimSize) && dimSize != sum)
        return emitOpError() << ": split tensor entries sum to " << sum
                             << " != axis dimension size " << dimSize;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXSplitToSequenceOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Value inputValue = input();
  if (!hasShapeAndRank(inputValue))
    return success(); // Cannot infer output shape if input shape isn't known.

  // NOTE: all the asserts below are conditions checked in verify()

  auto inputType = inputValue.getType().cast<ShapedType>();
  ArrayRef<int64_t> shape = inputType.getShape();
  int64_t rank = shape.size();
  int64_t axisIndex = axis();
  assert((-rank <= axisIndex && axisIndex < rank) && "axis out of range");
  if (axisIndex < 0)
    axisIndex += rank;
  int64_t dimSize = shape[axisIndex];

  // start with length unknown and dims == shape with unknown dimension size
  // for axis (-1 is ShapedType::kDynamicSize), and edit it as needed below
  int64_t length = -1;
  SmallVector<int64_t, 4> dims(shape.begin(), shape.end());
  dims[axisIndex] = -1;

  Value splitValue = split();
  if (isFromNone(splitValue)) {
    // since split is not specified, check the keepdims attribute
    int64_t keep = keepdims();
    assert(0 <= keep && keep <= 1 && "keepdims out of range");
    length = dimSize;
    if (keep == 1) {
      // if dimSize is zero we can choose any value here, 1 is fine
      dims[axisIndex] = 1;
    } else {
      dims.erase(dims.begin() + axisIndex);
    }
  } else {
    auto splitType = splitValue.getType().cast<ShapedType>();
    ArrayRef<int64_t> splitShape = splitType.getShape();
    int64_t splitRank = splitShape.size();
    assert(splitRank <= 1 && "invalid split tensor rank");
    if (ElementsAttr entries = getElementAttributeFromONNXValue(splitValue)) {
      if (splitRank == 0) {
        auto scalar = getScalarValue<int64_t>(entries, splitType);
        assert(scalar > 0 && "invalid split scalar");
        if (!ShapedType::isDynamic(dimSize)) {
          length = dimSize / scalar;
          if ((dimSize % scalar) == 0)
            dims[axisIndex] = scalar;
        }
      } else {
        auto values = entries.getValues<IntegerAttr>();
        length = values.size();
        if (length > 0) {
          // in the (unlikely?) case that all entries are the same, we infer
          // that's the dimension size for axis
          int64_t first = values[0].getInt();
          assert(first >= 0 && "invalid split tensor entry");
          if (llvm::all_of(values, [first](IntegerAttr value) {
                return value.getInt() == first;
              }))
            dims[axisIndex] = first;
        }
      }
    } else if (splitRank == 1 && !ShapedType::isDynamic(splitShape[0])) {
      length = splitShape[0];
      // corner case: if the input dimension size for axis is zero, any tensors
      // in the output sequence must also be zero if the sequence is non-empty
      if (length > 0 && dimSize == 0)
        dims[axisIndex] = 0;
      // if length and dimSize are both zero, we can choose any value,
      // leaving it be -1 is fine
    }
  }
  getResult().setType(SeqType::get(
      RankedTensorType::get(dims, inputType.getElementType()), length));
  return success();
}
