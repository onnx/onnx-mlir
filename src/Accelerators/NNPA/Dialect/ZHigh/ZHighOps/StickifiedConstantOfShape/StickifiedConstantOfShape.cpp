/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- StickifiedConstantOfShape.cpp - ZHigh Operations ------------------===//
//
// Copyright 2023- The IBM Research Authors.
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
// Custom builders
//===----------------------------------------------------------------------===//

void ZHighStickifiedConstantOfShapeOp::build(OpBuilder &builder,
    OperationState &state, Value shape, FloatAttr value, StringAttr layout) {
  Type resType = builder.getNoneType();
  ShapedType shapeType = shape.getType().cast<ShapedType>();
  Type elementType = builder.getF32Type();

  if (shapeType.hasRank()) {
    int64_t rank = shapeType.getShape()[0];
    ZTensorEncodingAttr::DataLayout dataLayout;
    if (layout)
      dataLayout = convertStringAttrToZTensorDataLayout(layout);
    else {
      dataLayout = getZTensorDataLayoutByRank(rank);
      // Create a layout attribute.
      layout = convertZTensorDataLayoutToStringAttr(builder, dataLayout);
    }
    SmallVector<int64_t, 4> resShape(rank, ShapedType::kDynamic);
    resType = RankedTensorType::get(resShape, elementType,
        ZTensorEncodingAttr::get(builder.getContext(), dataLayout));
  } else {
    resType = UnrankedTensorType::get(elementType);
  }
  build(builder, state, resType, shape, value, layout);
}

} // namespace zhigh
} // namespace onnx_mlir
