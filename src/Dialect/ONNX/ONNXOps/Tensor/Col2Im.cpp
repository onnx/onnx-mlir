/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Col2Im.cpp - ONNX Operations ----------------===//
//
// This file provides definition of ONNX dialect Col2Im operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

namespace {
// Verify that an optional spatial-rank-sized attribute, if present, has
// exactly `expected` elements. Model errors like this must be caught here
// rather than surfacing as an out-of-bounds ArrayAttr access deep in the
// ONNXToKrnl lowering.
LogicalResult checkSpatialAttrLength(ONNXCol2ImOp op,
    std::optional<ArrayAttr> attr, int64_t expected, StringRef name) {
  if (attr.has_value() && static_cast<int64_t>(attr->size()) != expected)
    return op.emitOpError() << "'" << name << "' has " << attr->size()
                            << " elements, expected " << expected;
  return success();
}
} // namespace

LogicalResult ONNXCol2ImOp::verify() {
  auto imageShapeType = mlir::dyn_cast<ShapedType>(getImageShape().getType());
  auto blockShapeType = mlir::dyn_cast<ShapedType>(getBlockShape().getType());
  // Defer to shape inference: spatial rank is not yet known.
  if (!imageShapeType || !blockShapeType || !imageShapeType.hasStaticShape() ||
      !blockShapeType.hasStaticShape())
    return success();

  int64_t spatialRank = imageShapeType.getShape()[0];
  if (blockShapeType.getShape()[0] != spatialRank)
    return emitOpError("image_shape and block_shape must have the same "
                       "number of elements");

  if (failed(checkSpatialAttrLength(
          *this, getDilations(), spatialRank, "dilations")) ||
      failed(checkSpatialAttrLength(
          *this, getStrides(), spatialRank, "strides")) ||
      failed(checkSpatialAttrLength(*this, getPads(), 2 * spatialRank, "pads")))
    return failure();

  if (std::optional<ArrayAttr> strides = getStrides())
    for (int64_t i = 0; i < spatialRank; ++i)
      if (mlir::cast<IntegerAttr>((*strides)[i]).getInt() <= 0)
        return emitOpError("strides must be strictly positive");

  return success();
}

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXCol2ImOpShapeHelper::computeShape() {
  ONNXCol2ImOpAdaptor operandAdaptor(operands);
  Value input = operandAdaptor.getInput();
  Value imageShape = operandAdaptor.getImageShape();
  Value blockShape = operandAdaptor.getBlockShape();
  if (!hasShapeAndRank(input))
    return failure();

  // Number of spatial dims is given by the (static) size of the image_shape
  // 1D tensor.
  int64_t spatialRank = createIE->getArraySize(imageShape, /*static*/ true);
  assert(spatialRank == createIE->getArraySize(blockShape, /*static*/ true) &&
         "image_shape and block_shape must have the same size");

  // Compute the product of the block_shape values.
  IndexExpr blockProd = LiteralIndexExpr(1);
  for (int64_t i = 0; i < spatialRank; ++i)
    blockProd = blockProd * createIE->getIntFromArrayAsSymbol(blockShape, i);

  // Output has shape [N, C, d_1, ..., d_n].
  DimsExpr outputDims;
  outputDims.emplace_back(createIE->getShapeAsDim(input, 0)); // N
  DimIndexExpr channelXBlock(createIE->getShapeAsDim(input, 1));
  outputDims.emplace_back(channelXBlock.floorDiv(blockProd)); // C
  for (int64_t i = 0; i < spatialRank; ++i)
    outputDims.emplace_back(createIE->getIntFromArrayAsDim(imageShape, i));

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXCol2ImOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no input shape exists.
  if (!hasShapeAndRank(getInput()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getInput().getType()).getElementType();
  ONNXCol2ImOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXCol2ImOp>;
} // namespace onnx_mlir
