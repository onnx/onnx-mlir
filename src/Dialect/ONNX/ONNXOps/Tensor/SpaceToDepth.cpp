/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ SpaceToDepth.cpp - ONNX Operations ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect SpaceToDepth operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXSpaceToDepthOpShapeHelper::computeShape(
    ONNXSpaceToDepthOpAdaptor operandAdaptor) {
  // Get info about input data operand and blocksize.
  Value input = operandAdaptor.input();
  int64_t blocksize = op->blocksize();
  assert(input.getType().isa<ShapedType>() && "Input should have a shape");
  assert(blocksize > 0 && "blocksize should be strictly positive");

  int64_t inputRank = input.getType().cast<ShapedType>().getShape().size();
  assert(inputRank == 4 && "Unexpected input tensor rank");

  // Compute outputDims.
  // The input tensor has format [N,C,H,W], where N is the batch axis, C is the
  // channel or depth, H is the height and W is the width. The output tensor has
  // shape [N, C * blocksize * blocksize, H/blocksize, W/blocksize].
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  MemRefBoundsIndexCapture inputBounds(input);
  DimIndexExpr N(inputBounds.getDim(0));
  DimIndexExpr C(inputBounds.getDim(1));
  DimIndexExpr H(inputBounds.getDim(2));
  DimIndexExpr W(inputBounds.getDim(3));

  outputDims[0] = N;
  outputDims[1] = C * blocksize * blocksize;
  outputDims[2] = H.floorDiv(blocksize);
  outputDims[3] = W.floorDiv(blocksize);

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXSpaceToDepthOp::verify() {
  ONNXSpaceToDepthOpAdaptor operandAdaptor(*this);

  // Check input.
  Value input = operandAdaptor.input();
  if (!hasShapeAndRank(input)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  auto inputType = input.getType().cast<ShapedType>();
  auto inputShape = inputType.getShape();
  if (inputShape.size() != 4)
    return emitOpError("Input should have a rank of four");

  // Check blocksize.
  int64_t blocksize = operandAdaptor.blocksize();
  if (blocksize < 0)
    return emitOpError("Blocksize should be non negative");

  int64_t H = inputShape[2];
  int64_t W = inputShape[3];

  if (!ShapedType::isDynamic(H) && H % blocksize != 0)
    return emitOpError(
        "The input tensor height must be divisible by the block size");
  if (!ShapedType::isDynamic(W) && W % blocksize != 0)
    return emitOpError(
        "The input tensor width must be divisible by the block size");

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXSpaceToDepthOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no input shape exists.
  if (!input().getType().isa<RankedTensorType>())
    return success();

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXSpaceToDepthOpShapeHelper,
      ONNXSpaceToDepthOp, ONNXSpaceToDepthOpAdaptor>(*this, elementType);
}
