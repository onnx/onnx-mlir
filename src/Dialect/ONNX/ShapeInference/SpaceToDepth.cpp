/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SpaceToDepth.cpp - Shape Inference for SpaceToDepth Op -------===//
//
// This file implements shape inference for the ONNX SpaceToDepth Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXSpaceToDepthOpShapeHelper::ONNXSpaceToDepthOpShapeHelper(
    ONNXSpaceToDepthOp *newOp)
    : ONNXOpShapeHelper<ONNXSpaceToDepthOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXSpaceToDepthOpShapeHelper::ONNXSpaceToDepthOpShapeHelper(
    ONNXSpaceToDepthOp *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXSpaceToDepthOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

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
  dimsForOutput(0) = outputDims;
  return success();
}

} // namespace onnx_mlir
