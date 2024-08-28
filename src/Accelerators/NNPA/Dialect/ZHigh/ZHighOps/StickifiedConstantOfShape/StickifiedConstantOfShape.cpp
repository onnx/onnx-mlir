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
  ShapedType shapeType = mlir::cast<ShapedType>(shape.getType());
  Type elementType = builder.getF16Type();

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

//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickifiedConstantOfShapeOpShapeHelper::computeShape() {
  auto stickOp = llvm::dyn_cast<ZHighStickifiedConstantOfShapeOp>(op);
  ZHighStickifiedConstantOfShapeOp::Adaptor operandAdaptor(operands);
  Value shape = operandAdaptor.getShape();
  StringAttr layout = stickOp.getLayoutAttr();

  if (!hasRankedType(shape))
    return success();

  auto shapeType = mlir::cast<ShapedType>(shape.getType());
  int64_t rank = shapeType.getShape()[0];

  // Output dims of result.
  DimsExpr outputDims;
  outputDims.resize(rank);
  for (int64_t i = 0; i < rank; i++) {
    IndexExpr dim = createIE->getIntFromArrayAsSymbol(shape, i, rank);
    outputDims[i] = dim;
  }

  // Direct stickify from NCHW to NHWC.
  if (isNHWCLayout(layout)) {
    assert((rank == 4) && "Stickify input must have rank 4");
    // NCHW -> NHWC
    IndexExpr C = outputDims[1];
    outputDims[1] = outputDims[2];
    outputDims[2] = outputDims[3];
    outputDims[3] = C;
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickifiedConstantOfShapeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Value shape = getShape();
  if (!hasRankedType(shape))
    return success();

  auto shapeType = mlir::cast<RankedTensorType>(shape.getType());
  StringAttr layout = getLayoutAttr();
  int64_t rank = shapeType.getShape()[0];

  ZTensorEncodingAttr::DataLayout dataLayout;
  if (layout)
    dataLayout = convertStringAttrToZTensorDataLayout(layout);
  else
    dataLayout = getZTensorDataLayoutByRank(rank);
  auto encoding = ZTensorEncodingAttr::get(this->getContext(), dataLayout);

  ZHighStickifiedConstantOfShapeOpShapeHelper shapeHelper(getOperation());
  Type elementType =
      mlir::cast<ShapedType>(getResult().getType()).getElementType();
  return shapeHelper.computeShapeAndUpdateType(elementType, encoding);
}

} // namespace zhigh
} // namespace onnx_mlir
