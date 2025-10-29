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

  // Need past_key/value inputs to infer shapes for present_key/value outputs
  if (attentionOp->getNumOperands() < 6)
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

LogicalResult ONNXAttentionOp::verify() {
  const int64_t numIn = this->getNumOperands();
  const int64_t numOut = this->getNumResults();

  // If presentK and presentV are outputs, then we must pass pastK and pastV as
  // inputs
  if (numOut >= 3) {
    Value presentK = this->getResult(1);
    Value presentV = this->getResult(2);
    if (!isNoneValue(presentK) || !isNoneValue(presentV)) {
      if (numIn < 6)
        return emitOpError("inputs 'pastK' and 'pastV' are needed for outputs "
                           "'presentK' and 'presentV'");

      Value pastK = this->getOperand(4);
      Value pastV = this->getOperand(5);
      if (isNoneValue(pastK) || isNoneValue(pastV))
        return emitOpError("inputs 'pastK' and 'pastV' are needed for outputs "
                           "'presentK' and 'presentV'");
    }
  }

  ONNXAttentionOpAdaptor adaptor(*this);

  Value q = adaptor.getQ();
  if (!hasShapeAndRank(q))
    return success(); // Won't be able to do any more checking at this stage.

  auto qType = mlir::cast<ShapedType>(q.getType());
  int64_t qRank = qType.getShape().size();
  if (qRank != 3 && qRank != 4)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), q, qRank, "3 or 4");

  // check q_num_heads is present for 3D input
  auto qNumHeads = adaptor.getQNumHeads();
  if (qRank == 3 && !qNumHeads)
    return emitOpError("attribute 'q_num_heads' must be provided when input "
                       "'q' is a 3D tensor.");

  Value k = adaptor.getK();
  Value v = adaptor.getV();
  if (!hasShapeAndRank(k) || !hasShapeAndRank(v))
    return success(); // Won't be able to do any more checking at this stage.

  auto kType = mlir::cast<ShapedType>(k.getType());
  int64_t kRank = kType.getShape().size();
  if (kRank != 3 && kRank != 4)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), k, kRank, "3 or 4");

  auto vType = mlir::cast<ShapedType>(v.getType());
  int64_t vRank = vType.getShape().size();
  if (vRank != 3 && vRank != 4)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), v, vRank, "3 or 4");

  // check kv_num_heads is present for 3D inputs
  auto kvNumHeads = adaptor.getKvNumHeads();
  if ((kRank == 3 || vRank == 3) && !kvNumHeads)
    return emitOpError("attribute 'kv_num_heads' must be provided when inputs "
                       "'k' or 'v' are 3D tensors.");

  auto divisibleByNumHeads =
      [&](ShapedType &type, std::optional<int64_t> numHeads, Value &operand) {
        if (type.hasStaticShape()) {
          auto shape = type.getShape();
          if (type.getRank() == 3 && numHeads && shape[2] % *numHeads != 0)
            return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
                *this->getOperation(), operand, 2, shape[2],
                "divisible by " + std::to_string(*numHeads));
        }
        return success();
      };

  auto qTypeDivisibleByQNumHeads = divisibleByNumHeads(qType, qNumHeads, q);
  if (!succeeded(qTypeDivisibleByQNumHeads))
    return qTypeDivisibleByQNumHeads;

  auto kTypeDivisibleByKVNumHeads = divisibleByNumHeads(kType, kvNumHeads, k);
  if (!succeeded(kTypeDivisibleByKVNumHeads))
    return kTypeDivisibleByKVNumHeads;

  auto vTypeDivisibleByKVNumHeads = divisibleByNumHeads(vType, kvNumHeads, v);
  if (!succeeded(vTypeDivisibleByKVNumHeads))
    return vTypeDivisibleByKVNumHeads;

  return success();
}

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