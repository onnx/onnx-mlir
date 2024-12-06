/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ GRU.cpp - ZHigh Operations ------------------------===//
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

LogicalResult ZHighGRUOpShapeHelper::computeShape() {
  ZHighGRUOp gruOp = llvm::dyn_cast<ZHighGRUOp>(op);
  ZHighGRUOp::Adaptor operandAdaptor(operands);
  // Get operands.
  // X: [S, B, I]
  Value X = operandAdaptor.getInput();
  // R: [D, H, H]
  Value R = operandAdaptor.getHiddenWeights();

  // Return all timesteps or only the final step;
  bool isAllTimesteps = (gruOp.getReturnAllSteps() == -1) ? true : false;

  // Get bounds
  SmallVector<IndexExpr, 4> XDims, RDims;
  createIE->getShapeAsDims(X, XDims);
  createIE->getShapeAsDims(R, RDims);
  IndexExpr S = XDims[0];
  IndexExpr B = XDims[1];
  IndexExpr I = XDims[2];
  IndexExpr D = RDims[0];
  IndexExpr H = RDims[1];

  // Shape for hn_ouput : [S, D, B, H] if return all  timesteps. [1, D, B, H] if
  // return the final step only.
  DimsExpr hnOutputDims;
  if (isAllTimesteps)
    hnOutputDims.emplace_back(S);
  else
    hnOutputDims.emplace_back(LitIE(1));
  hnOutputDims.emplace_back(D);
  hnOutputDims.emplace_back(B);
  hnOutputDims.emplace_back(H);

  // Shape for cf_ouput : [1, B, H]
  DimsExpr cfOutputDims;
  cfOutputDims.emplace_back(LitIE(1));
  cfOutputDims.emplace_back(B);
  cfOutputDims.emplace_back(H);

  // Shape for optional values.
  // Initialized value: [D, B, H]
  h0Shape.emplace_back(D);
  h0Shape.emplace_back(B);
  h0Shape.emplace_back(H);
  // Bias value: [D, 3*H]
  biasShape.emplace_back(D);
  biasShape.emplace_back(H * 3);

  // Keep all original dimensions.
  allOriginalDims.emplace_back(D);
  allOriginalDims.emplace_back(S);
  allOriginalDims.emplace_back(B);
  allOriginalDims.emplace_back(I);
  allOriginalDims.emplace_back(H);

  // Save the final results.
  setOutputDims(hnOutputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

LogicalResult ZHighGRUOp::verify() {
  ZHighGRUOpAdaptor operandAdaptor(*this);
  // Get operands.
  Value W = operandAdaptor.getInputWeights();
  Value R = operandAdaptor.getHiddenWeights();
  Value WB = operandAdaptor.getInputBias();
  Value RB = operandAdaptor.getHiddenBias();

  // Hidden size attribute.
  int64_t hiddenSize = getHiddenSize();

  // Verify hidden size in W.
  if (hasRankedType(W)) {
    int64_t dim2 = mlir::cast<RankedTensorType>(W.getType()).getShape()[2];
    if (!ShapedType::isDynamic(dim2) && (dim2 != hiddenSize * 3))
      return failure();
  }

  // Verify hidden size in R.
  if (hasRankedType(R)) {
    int64_t dim1 = mlir::cast<RankedTensorType>(R.getType()).getShape()[1];
    int64_t dim2 = mlir::cast<RankedTensorType>(R.getType()).getShape()[2];
    if (!ShapedType::isDynamic(dim1) && (dim1 != hiddenSize))
      return failure();
    if (!ShapedType::isDynamic(dim2) && (dim2 != hiddenSize * 3))
      return failure();
  }

  // Verify hidden size in WB.
  if (!mlir::isa<NoneType>(WB.getType()) && hasRankedType(WB)) {
    int64_t dim1 = mlir::cast<RankedTensorType>(WB.getType()).getShape()[1];
    if (!ShapedType::isDynamic(dim1) && (dim1 != hiddenSize * 3))
      return failure();
  }

  // Verify hidden size in RB.
  if (!mlir::isa<NoneType>(RB.getType()) && hasRankedType(RB)) {
    int64_t dim1 = mlir::cast<RankedTensorType>(RB.getType()).getShape()[1];
    if (!ShapedType::isDynamic(dim1) && (dim1 != hiddenSize * 3))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighGRUOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasRankedType(getInput()) || !hasRankedType(getHiddenWeights()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getResult().getType()).getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::_4DS);
  ZHighGRUOpShapeHelper shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(elementType, encoding);
}

} // namespace zhigh
} // namespace onnx_mlir
