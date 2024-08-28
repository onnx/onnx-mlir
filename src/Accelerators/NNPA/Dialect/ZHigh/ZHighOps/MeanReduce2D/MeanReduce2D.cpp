/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ MenReduce2D.cpp - ZHigh Operations ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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

LogicalResult ZHighMeanReduce2DOpShapeHelper::computeShape() {
  ZHighMeanReduce2DOp::Adaptor operandAdaptor(operands);
  Value input = operandAdaptor.getInput();

  // Output dims of result.
  DimsExpr outputDims;

  // Get operands and bounds.
  SmallVector<IndexExpr, 4> inputDims;
  createIE->getShapeAsDims(input, inputDims);
  int64_t rank = inputDims.size();
  assert((rank == 4) && "ZHighMeanReduce2D's input must have rank 4");

  // Input is NHWC, and H and W are reduction dimensions.
  outputDims.emplace_back(inputDims[0]);
  outputDims.emplace_back(LitIE(1));
  outputDims.emplace_back(LitIE(1));
  outputDims.emplace_back(inputDims[3]);

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighMeanReduce2DOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasRankedType(getInput()))
    return success();

  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  ZHighMeanReduce2DOpShapeHelper shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(
      inputType.getElementType(), inputType.getEncoding());

  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
