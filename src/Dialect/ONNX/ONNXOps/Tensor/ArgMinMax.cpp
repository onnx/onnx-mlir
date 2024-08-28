/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ArgMinMax.cpp - ONNX Operations -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect ArgMin/Max operations.
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

template <typename OP_TYPE>
LogicalResult ONNXArgMinMaxOpShapeHelper<OP_TYPE>::computeShape() {
  // Get info about input data operand.
  OP_TYPE argOp = llvm::cast<OP_TYPE>(op);
  typename OP_TYPE::Adaptor operandAdaptor(operands);
  Value data = operandAdaptor.getData();
  int64_t dataRank = mlir::cast<ShapedType>(data.getType()).getRank();
  int64_t axisValue = argOp.getAxis();

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  assert(-dataRank <= axisValue && axisValue < dataRank && "axis out of range");

  // Negative axis means values are counted from the opposite side.
  if (axisValue < 0) {
    axisValue = dataRank + axisValue;
    auto builder = Builder(op->getContext());
    argOp.setAxisAttr(
        IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
            APInt(64, /*value=*/axisValue, /*isSigned=*/true)));
  }

  // The keepdims is a required attribute and should have default value of 1.
  int64_t keepdims = argOp.getKeepdims();
  bool isKeepdims = (keepdims == 1);

  // Compute outputDims
  DimsExpr outputDims;
  int64_t reducedRank = isKeepdims ? dataRank : dataRank - 1;
  outputDims.resize(reducedRank);
  for (int64_t i = 0; i < reducedRank; i++) {
    if (isKeepdims)
      outputDims[i] =
          (i != axisValue) ? createIE->getShapeAsDim(data, i) : LitIE(1);
    else
      outputDims[i] = (i < axisValue) ? createIE->getShapeAsDim(data, i)
                                      : createIE->getShapeAsDim(data, i + 1);
  }
  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// ONNXArgMaxOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXArgMaxOp::verify() {
  ONNXArgMaxOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(getOperation()))
    return success();

  int64_t rank = mlir::cast<ShapedType>(getData().getType()).getRank();
  int64_t axisIndex = getAxis();

  // axis value must be in the range [-rank, rank-1].
  if (axisIndex < -rank || axisIndex >= rank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));

  return success();
}

LogicalResult ONNXArgMaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getData()))
    return success();

  // ONNX spec specifies the reduced type as an int64
  Type elementType = IntegerType::get(getContext(), 64);
  ONNXArgMaxOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// ONNXArgMinOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXArgMinOp::verify() {
  ONNXArgMinOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(getOperation()))
    return success();

  int64_t rank = mlir::cast<ShapedType>(getData().getType()).getRank();
  int64_t axisIndex = getAxis();

  // axis value must be in the range [-rank, rank-1].
  if (axisIndex < -rank || axisIndex >= rank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));

  return success();
}

LogicalResult ONNXArgMinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getData()))
    return success();

  // ONNX spec specifies the reduced type as an int64
  Type elementType = IntegerType::get(getContext(), 64);
  ONNXArgMinOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct ONNXArgMinMaxOpShapeHelper<ONNXArgMaxOp>;
template struct ONNXArgMinMaxOpShapeHelper<ONNXArgMinOp>;

} // namespace onnx_mlir
