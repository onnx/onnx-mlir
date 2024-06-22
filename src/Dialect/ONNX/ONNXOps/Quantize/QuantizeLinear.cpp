/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ QuantizeLinear.cpp.cpp - ONNX Operations ----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect QuantizeLinear operation.
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
LogicalResult ONNXQuantizeLinearOpShapeHelper::computeShape() {
  ONNXQuantizeLinearOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  DimsExpr outputDims;
  createIE->getShapeAsDims(operandAdaptor.getX(), outputDims);
  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXQuantizeLinearOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto inTy = mlir::dyn_cast<RankedTensorType>(getX().getType());
  if (!inTy) {
    return success();
  }

  Type elementType;
  Value zero = getYZeroPoint();
  if (isNoneValue(zero)) {
    // If zero point type isn't provided, output type defaults to ui8.
    elementType = IntegerType::get(getContext(), 8, IntegerType::Unsigned);
  } else {
    // If zero point is provided, output type is same as zero point type.
    elementType = mlir::cast<ShapedType>(zero.getType()).getElementType();
  }

  ONNXQuantizeLinearOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateTypes(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct ONNXNonSpecificOpShapeHelper<ONNXQuantizeLinearOp>;

} // namespace onnx_mlir
