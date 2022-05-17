/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ DepthToSpace.cpp - Shape Inference for DepthToSpace Op --------===//
//
// This file implements shape inference for the ONNX DepthToSpace Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXDepthToSpaceOpShapeHelper::computeShape(
    ONNXDepthToSpaceOpAdaptor operandAdaptor) {
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
  // shape [N, C / (blocksize * blocksize), H * blocksize, W * blocksize].
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  MemRefBoundsIndexCapture inputBounds(input);
  DimIndexExpr N(inputBounds.getDim(0));
  DimIndexExpr C(inputBounds.getDim(1));
  DimIndexExpr H(inputBounds.getDim(2));
  DimIndexExpr W(inputBounds.getDim(3));

  outputDims[0] = N;
  outputDims[1] = C.floorDiv(blocksize * blocksize);
  outputDims[2] = H * blocksize;
  outputDims[3] = W * blocksize;

  // Save the final result.
  dimsForOutput() = outputDims;
  return success();
}

} // namespace onnx_mlir
