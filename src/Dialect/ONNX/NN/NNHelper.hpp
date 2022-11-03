/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ NNHelper.hpp - ONNX Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides specific helpers for Conv and Pool operations.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support function for verifiers.
//===----------------------------------------------------------------------===//

namespace {

// For ops without filter, pass nullptr in filterOperand.
template <class T>
static LogicalResult verifyKernelShape(T *op, Value filterOperand,
    Optional<ArrayAttr> kernelShapeOpt, int64_t spatialRank) {
  if (filterOperand && !hasShapeAndRank(filterOperand)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  // 1) Get shape of filter. Shape is not guaranteed to be compile time
  // constant.
  ArrayRef<int64_t> filterShape =
      filterOperand ? filterOperand.getType().cast<ShapedType>().getShape()
                    : ArrayRef<int64_t>();
  // 2) Get kernel_shape attribute
  if (!kernelShapeOpt.has_value()) {
    assert(
        filterOperand && "ops without filter have mandatory kernel_shape arg");
    // Don't have a kernel shape explicitly, still make sure that the filter
    // shape are fine if known. If size is negative, ok since this is runtime.
    // If positive, ok since it must be strictly positive. If zero, that is
    // bad.
    for (int i = 0; i < spatialRank; ++i)
      if (filterShape[2 + i] == 0)
        return op->emitError("Bad spatial filter size: cannot be zero");
    return success();
  }
  // 3) Verify that we have the right number.
  if ((int64_t)ArrayAttrSize(kernelShapeOpt) != spatialRank)
    return op->emitError(
        "kernel_shape length incompatible with spatial dimensions");
  // 4) Verify that they are all positive.
  for (int i = 0; i < spatialRank; ++i) {
    auto attrSize = ArrayAttrIntVal(kernelShapeOpt, i);
    if (attrSize < 1)
      return op->emitError("Bad kernel_shape value: must be strictly positive");
    if (filterOperand) {
      // Has a shape from filter, make sure its consistent.
      auto filterSize = filterShape[2 + i];
      if (filterSize >= 0 && filterSize != attrSize)
        return op->emitError(
            "Bad kernel_shape value: does not match filter sizes");
    }
  }
  return success();
}

template <class T>
static LogicalResult verifyStrides(T *op, int64_t spatialRank) {
  // 1) Get strides attribute.
  auto strides = op->strides();
  if (!strides.has_value())
    return success();
  // 2) Verify that we have the right number.
  if ((int64_t)ArrayAttrSize(strides) != spatialRank)
    return op->emitError("strides length incompatible with spatial dimensions");
  // 3) Verify that they are all positive.
  for (int i = 0; i < spatialRank; ++i) {
    auto attrSize = ArrayAttrIntVal(strides, i);
    if (attrSize < 1)
      return op->emitError("Bad stride value: must be strictly positive");
  }
  return success();
}

template <class T>
static LogicalResult verifyDilations(T *op, int64_t spatialRank) {
  // 1) Get dilation attribute.
  auto dilations = op->dilations();
  if (!dilations.has_value())
    return success();
  // 2) Verify that we have the right number.
  if ((int64_t)ArrayAttrSize(dilations) != spatialRank)
    return op->emitError(
        "dilations length incompatible with spatial dimensions");
  // 3) Verify that they are all positive.
  for (int i = 0; i < spatialRank; ++i) {
    auto attrSize = ArrayAttrIntVal(dilations, i);
    if (attrSize < 1)
      return op->emitError("Bad dilation value: must be strictly positive");
  }
  return success();
}

template <class T>
static LogicalResult verifyPadding(T *op, int64_t spatialRank) {
  // Verify auto pad field.
  auto autoPad = op->auto_pad();
  if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ||
      autoPad == "VALID" || autoPad == "NOTSET") {
    // Ok, known auto pad value.
  } else {
    return op->emitError("Unknown auto pad option");
  }
  // Verify pad values, if defined.
  auto pads = op->pads();
  if (!pads.has_value())
    return success();
  // Verify that we have the right number of pad values.
  if ((int32_t)ArrayAttrSize(pads) != 2 * spatialRank)
    return op->emitError("pads length incompatible with spatial dimensions");
  // Verify the values of the pads.
  if (autoPad == "NOTSET") {
    for (int i = 0; i < 2 * spatialRank; ++i)
      if (ArrayAttrIntVal(pads, i) < 0)
        return op->emitError("Bad pad value: must be nonnegative");
  } else {
    for (int i = 0; i < 2 * spatialRank; ++i)
      if (ArrayAttrIntVal(pads, i) != 0)
        return op->emitError("Bad pad value: nonzero pad value only allowed "
                             "with NOTSET option");
  }
  return success();
}

} // namespace
