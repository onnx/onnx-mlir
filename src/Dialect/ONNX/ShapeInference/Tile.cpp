/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Tile.cpp - Shape Inference for Tile Op ----------------===//
//
// This file implements shape inference for the ONNX Tile Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXTileOpShapeHelper::ONNXTileOpShapeHelper(ONNXTileOp *newOp)
    : ONNXOpShapeHelper<ONNXTileOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXTileOpShapeHelper::ONNXTileOpShapeHelper(ONNXTileOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXTileOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXTileOpShapeHelper::computeShape(
    ONNXTileOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Get info about input data operand.
  Value input = operandAdaptor.input();
  // TOFIX: need to check is_a<ShapedType>?
  int64_t inputRank = input.getType().cast<ShapedType>().getShape().size();
  Value repeats = operandAdaptor.repeats();

  // Compute outputDims
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  MemRefBoundsIndexCapture inputBounds(input);
  ArrayValueIndexCapture repeatsCapture(repeats, fGetDenseVal, fLoadVal);
  for (auto i = 0; i < inputRank; i++) {
    DimIndexExpr dimInput(inputBounds.getDim(i));
    SymbolIndexExpr repeatsValue(repeatsCapture.getSymbol(i));
    IndexExpr dimOutput = dimInput * repeatsValue;
    outputDims[i] = dimOutput;
  }
  dimsForOutput(0) = outputDims;
  return success();
}

} // namespace onnx_mlir
