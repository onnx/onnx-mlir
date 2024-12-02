/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ OneHotEncoder.cpp - ONNX Operations ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect OneHotEncoder operation.
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
LogicalResult ONNXOneHotEncoderOpShapeHelper::computeShape() {
  ONNXOneHotEncoderOp oneHotOp = llvm::cast<ONNXOneHotEncoderOp>(op);
  ONNXOneHotEncoderOpAdaptor operandAdaptor(operands);
  Value X = operandAdaptor.getX();
  ShapedType inputType = mlir::dyn_cast<RankedTensorType>(X.getType());
  assert(inputType && "expected ranked type");

  // If the input is a tensor of float, int32, or double,
  // the data will be cast to integers and
  // the cats_int64s category list will be used for the lookups.
  int64_t outDim;
  if (inputType.getElementType().isIntOrFloat()) {
    outDim = ArrayAttrSize(oneHotOp.getCatsInt64s());
  } else {
    outDim = ArrayAttrSize(oneHotOp.getCatsStrings());
  }

  // Encoded output data, having one more dimension than X
  // total category count will determine the size of the extra dimension
  DimsExpr outputDims;
  createIE->getShapeAsDims(X, outputDims);
  outputDims.emplace_back(LitIE(outDim));

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}
} // namespace onnx_mlir
//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXOneHotEncoderOp::verify() {
  ONNXOneHotEncoderOpAdaptor operandAdaptor = ONNXOneHotEncoderOpAdaptor(*this);

  // get operands
  auto input = operandAdaptor.getX();
  if (!hasShapeAndRank(input))
    return success();

  auto inputType = mlir::cast<ShapedType>(input.getType());
  if (!inputType)
    return success();

  // If the input is a tensor of float, int32, or double,
  // the data will be cast to integers and
  // the cats_int64s category list will be used for the lookups.
  if (inputType.getElementType().isIntOrFloat()) {
    if (!operandAdaptor.getCatsInt64s()) {
      return emitOpError("input is a tensor of float, int32, or double, "
                         "but no cats_int64s attribute");
    }
  } else {
    if (!operandAdaptor.getCatsStrings()) {
      return emitOpError("input is not a tensor of float, int32, or double, "
                         "but no cats_strings attribute");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXOneHotEncoderOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()))
    return success();

  ONNXOneHotEncoderOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(FloatType::getF32(getContext()));
  return success();
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXOneHotEncoderOp>;
} // namespace onnx_mlir
