/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Reshape.cpp - ZHigh Operations
//---------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

LogicalResult ZHighReshapeOpShapeHelper::computeShape() {
  ZHighReshapeOpAdaptor operandAdaptor(operands);

  // Shape has the dimensions of the output.
  DimsExpr outputDims;
  createIE->getIntFromArrayAsSymbols(operandAdaptor.getShape(), outputDims);
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

// Builder

LogicalResult ZHighReshapeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Value source = getSource();
  if (!hasRankedType(source))
    return success();
  // Output type has the same type as the input/source type.
  RankedTensorType sourceType =
      mlir::dyn_cast<RankedTensorType>(source.getType());
  Type elementType = sourceType.getElementType();
  // Get encoding
  StringAttr layout = getLayoutAttr();
  ZTensorEncodingAttr::DataLayout dataLayout;
  Attribute encoding;
  if (layout) {
    // Operation has an optional output layout, use it.
    dataLayout = convertStringAttrToZTensorDataLayout(layout);
    encoding = ZTensorEncodingAttr::get(this->getContext(), dataLayout);
  } else {
    // Operation does not have an optional output layout, reuse it from input.
    encoding = sourceType.getEncoding();
  }

  ZHighReshapeOpShapeHelper shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(elementType, encoding);
}

} // namespace zhigh
} // namespace onnx_mlir
