/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- ShapeTransform.cpp - ONNX Operations ---------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXShapeTransformOpShapeHelper::computeShape() {
  ONNXShapeTransformOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  Value input = operandAdaptor.getInput();
  AffineMap indexMap = operandAdaptor.getIndexMap();

  auto inputType = mlir::cast<ShapedType>(input.getType());
  Type elementType = inputType.getElementType();
  ArrayRef<int64_t> inputDims = inputType.getShape();
  int64_t outputRank = indexMap.getNumResults();

  // Use the given affine_map to compute output's shape.
  // IndexExpr does not support construction with an existing affine_map, so
  // compute the output's shape manually, and put it inside IndexExpr as result.
  //
  // Note that, affine_map is for index access, but what we want to compute here
  // is the upper bound for output dimensions.
  //
  // We will borrow memref normalization in MLIR to obtain the upper bound, as
  // follows:
  // - construct a MemRefType using the input shape and affine_map
  // - normalize the MemRefType.
  // TODO: support dynamic shape.
  MemRefType affineMemRefType =
      MemRefType::get(inputDims, elementType, AffineMapAttr::get(indexMap));
  MemRefType flatMemRefType = affine::normalizeMemRefType(affineMemRefType);
  assert((flatMemRefType.getRank() == outputRank) && "Normalization failed");

  DimsExpr outputIEs(outputRank);
  for (int64_t i = 0; i < outputRank; ++i) {
    LiteralIndexExpr dim(flatMemRefType.getShape()[i]);
    outputIEs[i] = dim;
  }

  setOutputDims(outputIEs);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXShapeTransformOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Operation *op = getOperation();
  // If any input is not ranked tensor, do nothing.
  if (!hasShapeAndRank(op))
    return success();
  // Input and output have the same element type and encoding.
  auto inputType = mlir::cast<RankedTensorType>(getOperand().getType());
  ONNXShapeTransformOpShapeHelper shapeHelper(op, {});
  return shapeHelper.computeShapeAndUpdateTypes(
      inputType.getElementType(), inputType.getEncoding());
}

LogicalResult ONNXShapeTransformOp::verify() {
  ONNXShapeTransformOpAdaptor operandAdaptor(*this);

  // Get operands.
  Value input = operandAdaptor.getInput();
  Value output = getOutput();
  AffineMap indexMap = operandAdaptor.getIndexMap();

  // Does not support affine_map with symbols.
  // All inputs of affine_map must be from dim.
  if (indexMap.getNumSymbols() != 0)
    return emitError("Does not support affine_map with symbols");

  // Only support static shape at this moment.
  auto inputType = mlir::dyn_cast<ShapedType>(input.getType());
  if (inputType && !inputType.hasStaticShape())
    return emitError("Does not support input with dynamic shape");

  // If input and output have static shape, check that the same number of
  // elements are the same.
  if (auto outputType = mlir::dyn_cast<ShapedType>(output.getType()))
    if (outputType.hasStaticShape()) {
      uint64_t elementsInput = 1;
      for (uint64_t d : inputType.getShape())
        elementsInput *= d;
      uint64_t elementsOutput = 1;
      for (uint64_t d : outputType.getShape())
        elementsOutput *= d;
      if (elementsInput != elementsOutput)
        return emitError(
            "The number of elements in the input and output mismatched");
    }

  return success();
}
