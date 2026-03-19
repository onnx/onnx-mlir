/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- Clip.cpp - ONNX Operations ---------------===//
//
// (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
// =============================================================================
//
// This file provides definition of ONNX dialect Clip operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXClipOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Folder
//===----------------------------------------------------------------------===//

OpFoldResult ONNXClipOp::fold(FoldAdaptor adaptor) {
  // FoldAdaptor returns non-null attributes only for operands that are
  // constants (e.g. produced by ONNXConstantOp). NoneType operands fold to
  // UnitAttr, which the ElementsAttr cast below rejects.
  auto minAttr = mlir::dyn_cast_or_null<ElementsAttr>(adaptor.getMin());
  auto maxAttr = mlir::dyn_cast_or_null<ElementsAttr>(adaptor.getMax());
  if (!minAttr || !maxAttr)
    return nullptr;

  if (minAttr.getNumElements() != 1 || maxAttr.getNumElements() != 1)
    return nullptr;

  auto resultType = mlir::dyn_cast<RankedTensorType>(getOutput().getType());
  if (!resultType || !resultType.hasStaticShape())
    return nullptr;

  // Per the ONNX spec: when min >= max, all output values equal max.
  // This folder does not preserve NaN values, the whole operation is replaced
  // by a constant
  Type elemType = resultType.getElementType();
  if (mlir::isa<FloatType>(elemType)) {
    const auto minVal = minAttr.getValues<APFloat>()[0];
    const auto maxVal = maxAttr.getValues<APFloat>()[0];
    if (maxVal <= minVal) {
      return DenseElementsAttr::get(resultType, maxVal);
    }
  } else if (auto intType = mlir::dyn_cast<IntegerType>(elemType)) {
    const auto minVal = minAttr.getValues<APInt>()[0];
    const auto maxVal = maxAttr.getValues<APInt>()[0];
    const bool minGeMax =
        intType.isUnsigned() ? minVal.uge(maxVal) : minVal.sge(maxVal);
    if (minGeMax) {
      return DenseElementsAttr::get(resultType, maxVal);
    }
  }

  return nullptr;
}
