// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.

// =============================================================================
//
// This file provides definition of ONNX dialect Attention operation.
//
//===----------------------------------------------------------------------===//

#include "ONNXOps.hpp"

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXAttentionOpShapeHelper::computeShape() {
  auto attentionOp = cast<ONNXAttentionOp>(op);

  int64_t rank = createIE->getShapedTypeRank(attentionOp.getQ());
  DimsExpr qShape;
  createIE->getShapeAsDims(attentionOp.getQ(), qShape);
  DimsExpr kShape;
  createIE->getShapeAsDims(attentionOp.getK(), kShape);
  DimsExpr vShape;
  createIE->getShapeAsDims(attentionOp.getV(), vShape);

  auto qNumHeads = attentionOp.getQNumHeads();
  auto kvNumHeads = attentionOp.getKvNumHeads();

  if (rank == 4) {
    DimsExpr outputDims = qShape;
    outputDims[3] = vShape[3];
    setOutputDims(outputDims, 0);
  } else if (rank == 3) {
    assert(qNumHeads && kvNumHeads &&
           "*_num_heads attributes must be present with 3D inputs");
    DimsExpr outputDims = qShape;
    outputDims[2] = LitIE(*qNumHeads * (vShape[2].getLiteral() / *kvNumHeads));
    setOutputDims(outputDims, 0);
  } else {
    return failure();
  }

  if (attentionOp.getOperands().size() != 6)
    return success();

  if (isNoneValue(attentionOp.getPastKey()) ||
      isNoneValue(attentionOp.getPastValue()) ||
      isNoneValue(attentionOp.getPresentKey()) ||
      isNoneValue(attentionOp.getPresentValue()))
    return success();

  if (!hasShapeAndRank(attentionOp.getPastKey()) ||
      !hasShapeAndRank(attentionOp.getPastValue()))
    return success();

  DimsExpr pastKShape;
  createIE->getShapeAsDims(attentionOp.getPastKey(), pastKShape);
  DimsExpr pastVShape;
  createIE->getShapeAsDims(attentionOp.getPastValue(), pastVShape);

  if (pastKShape.size() != 4 || pastVShape.size() != 4)
    return failure();

  auto totalSeqLen = pastKShape[2] + kShape[2];

  DimsExpr presentKeyDims = kShape;
  presentKeyDims[2] = totalSeqLen;
  setOutputDims(presentKeyDims, 1);

  DimsExpr presentValueDims = vShape;
  presentValueDims[2] = totalSeqLen;
  setOutputDims(presentValueDims, 2);

  if (attentionOp.getQkMatmulOutputMode()) {
    DimsExpr qkOutputDims = qShape;
    qkOutputDims[3] = totalSeqLen;
    setOutputDims(presentValueDims, 3);
  }

  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXAttentionOp::inferShapes(
    std::function<void(Region &)> /*shapeInferenceFunc*/) {

  // Cannot infer the output shape if the q, k, v operands' shapes are not yet
  // known.
  for (size_t i = 0; i < 3; i++)
    if (!hasShapeAndRank(this->getOperand(i)))
      return success();

  Type elementType = mlir::cast<ShapedType>(getQ().getType()).getElementType();
  ONNXAttentionOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
  return success();
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXAttentionOp>;
} // namespace onnx_mlir