/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ DynamicQuantizeLinear.cpp - ONNX Operations -------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect DynamicQuantizeLinear
// operation.
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
LogicalResult ONNXDynamicQuantizeLinearOpShapeHelper::computeShape() {
  ONNXDynamicQuantizeLinearOpAdaptor operandAdaptor(
      operands, op->getAttrDictionary());

  // Dim of y are the same as x.
  DimsExpr outputDims;
  createIE->getShapeAsDims(operandAdaptor.getX(), outputDims);
  setOutputDims(outputDims, 0);

  // y_scale and y_zero_point are scalar outputs...
  outputDims.clear();
  setOutputDims(outputDims, 1);
  setOutputDims(outputDims, 2);

  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXDynamicQuantizeLinearOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto inTy = mlir::dyn_cast<RankedTensorType>(getX().getType());
  if (!inTy)
    return success();

  IntegerType ui8Type =
      IntegerType::get(getContext(), 8, IntegerType::Unsigned);
  FloatType f32Type = FloatType::getF32(getContext());

  ONNXDynamicQuantizeLinearOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateTypes(
      {/*y*/ ui8Type, /*scale*/ f32Type, /*zero point*/ ui8Type});
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct ONNXNonSpecificOpShapeHelper<ONNXDynamicQuantizeLinearOp>;

} // namespace onnx_mlir
