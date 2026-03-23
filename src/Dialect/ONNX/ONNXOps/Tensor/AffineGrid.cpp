/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ AffineGrid.cpp - ONNX Operations ------------------===//
//
// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
// =============================================================================
//
// This file provides definition of ONNX dialect AffineGrid operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXAffineGridOpShapeHelper::computeShape() {
  ONNXAffineGridOpAdaptor operandAdaptor(operands);
  Value sizeVal = operandAdaptor.getSize();

  // size = (N, C, H, W) for 2D or (N, C, D, H, W) for 3D.
  // output shape:
  //   2D: output = (N, H, W, 2)
  //   3D: output = (N, D, H, W, 3)
  const int64_t sizeLen = cast<ShapedType>(sizeVal.getType()).getDimSize(0);
  const bool is2D = (sizeLen == 4);
  assert(sizeLen == 4 || sizeLen == 5);

  DimsExpr outputDims;
  outputDims.emplace_back(createIE->getIntFromArrayAsSymbol(sizeVal, 0));
  outputDims.emplace_back(createIE->getIntFromArrayAsSymbol(sizeVal, 2));
  outputDims.emplace_back(createIE->getIntFromArrayAsSymbol(sizeVal, 3));

  if (is2D) {
    outputDims.emplace_back(LitIE(2));
  } else {
    outputDims.emplace_back(createIE->getIntFromArrayAsSymbol(sizeVal, 4));
    outputDims.emplace_back(LitIE(3));
  }

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

LogicalResult ONNXAffineGridOp::inferShapes(
    std::function<void(Region &)> /*doShapeInference*/) {
  const auto sizeType = dyn_cast<ShapedType>(getSize().getType());
  if (!sizeType || !sizeType.hasStaticShape())
    return success();

  const Type elementType = getElementTypeOrSelf(getTheta().getType());
  ONNXAffineGridOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXAffineGridOp>;
} // namespace onnx_mlir
