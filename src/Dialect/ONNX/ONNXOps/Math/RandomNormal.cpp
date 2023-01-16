/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ RandomNormal.cpp - ONNX Operations ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect RandomNormal operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXRandomNormalOpShapeHelper::computeShape() {
  ONNXRandomNormalOp randomOp = llvm::cast<ONNXRandomNormalOp>(op);

  DimsExpr outputDims;
  createIE->getIntFromArrayAsLiterals(randomOp.shape(), outputDims);
  if (!IndexExpr::isNonNegativeLiteral(outputDims))
    return op->emitError("Random normal tensor has dynamic dimension.");
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

LogicalResult ONNXRandomNormalOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto elementTypeID = dtype();
  Type elementType = FloatType::getF32(getContext());
  if (elementTypeID == 0)
    elementType = FloatType::getF16(getContext());
  else if (elementTypeID == 2)
    elementType = FloatType::getF64(getContext());

  ONNXRandomNormalOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXRandomNormalOp>;
} // namespace onnx_mlir
