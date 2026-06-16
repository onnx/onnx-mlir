/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Attention.cpp - ONNX Operations ----------------------===//
//
// Copyright 2024-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Attention operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

LogicalResult ONNXAttentionOpShapeHelper::computeShape() {
  ONNXAttentionOp attentionOp = mlir::dyn_cast<ONNXAttentionOp>(op);
  ONNXAttentionOpAdaptor operandAdaptor(operands, attentionOp);

  Value Q = operandAdaptor.getQ();
  Value K = operandAdaptor.getK();
  Value V = operandAdaptor.getV();
  Value pastKey = operandAdaptor.getPastKey();
  Value pastValue = operandAdaptor.getPastValue();

  bool hasPastKey = !isNoneValue(pastKey);
  bool hasPastValue = !isNoneValue(pastValue);

  int64_t inputRank = mlir::cast<ShapedType>(Q.getType()).getShape().size();
  bool is3DInput = (inputRank == 3);

  // Q shape: (batch_size, q_sequence_length, q_hidden_size) for 3D
  //       or (batch_size, q_num_heads, q_sequence_length, head_size) for 4D
  IndexExpr batchSize = createIE->getShapeAsDim(Q, 0);

  // Output Y: (batch_size, q_sequence_length, v_hidden_size) for 3D input
  //        or (batch_size, q_num_heads, q_sequence_length, v_head_size) for 4D
  DimsExpr yDims;
  if (is3DInput) {
    IndexExpr qSeqLen = createIE->getShapeAsDim(Q, 1);
    IndexExpr vHiddenSize = createIE->getShapeAsDim(V, 2);
    yDims = {batchSize, qSeqLen, vHiddenSize};
  } else {
    IndexExpr qNumHeads = createIE->getShapeAsDim(Q, 1);
    IndexExpr qSeqLen = createIE->getShapeAsDim(Q, 2);
    IndexExpr vHeadSize = createIE->getShapeAsDim(V, 3);
    yDims = {batchSize, qNumHeads, qSeqLen, vHeadSize};
  }
  setOutputDims(yDims, 0);

  // present_key and present_value outputs.
  // They are present only when past_key/past_value are used (cache update
  // inside the op). Shape: (batch_size, kv_num_heads, total_seq_len, head_size)
  // where total_seq_len = past_seq_len + kv_seq_len.
  if (hasPastKey && !isNoneValue(attentionOp.getPresentKey())) {
    // past_key: (batch_size, kv_num_heads, past_seq_len, head_size)
    IndexExpr kvNumHeads = createIE->getShapeAsDim(pastKey, 1);
    IndexExpr pastSeqLen = createIE->getShapeAsDim(pastKey, 2);
    IndexExpr headSize = createIE->getShapeAsDim(pastKey, 3);

    IndexExpr kvSeqLen;
    if (is3DInput) {
      kvSeqLen = createIE->getShapeAsDim(K, 1);
    } else {
      kvSeqLen = createIE->getShapeAsDim(K, 2);
    }
    IndexExpr totalSeqLen = pastSeqLen + kvSeqLen;

    DimsExpr presentKeyDims = {batchSize, kvNumHeads, totalSeqLen, headSize};
    setOutputDims(presentKeyDims, 1);
  } else {
    DimsExpr emptyDims;
    setOutputDims(emptyDims, 1);
  }

  if (hasPastValue && !isNoneValue(attentionOp.getPresentValue())) {
    IndexExpr kvNumHeads = createIE->getShapeAsDim(pastValue, 1);
    IndexExpr pastSeqLen = createIE->getShapeAsDim(pastValue, 2);
    IndexExpr vHeadSize = createIE->getShapeAsDim(pastValue, 3);

    IndexExpr kvSeqLen;
    if (is3DInput) {
      kvSeqLen = createIE->getShapeAsDim(K, 1);
    } else {
      kvSeqLen = createIE->getShapeAsDim(K, 2);
    }
    IndexExpr totalSeqLen = pastSeqLen + kvSeqLen;

    DimsExpr presentValueDims = {batchSize, kvNumHeads, totalSeqLen, vHeadSize};
    setOutputDims(presentValueDims, 2);
  } else {
    DimsExpr emptyDims;
    setOutputDims(emptyDims, 2);
  }

  // qk_matmul_output: optional, depends on qk_matmul_output_mode.
  // Mode 0: not output (None).
  // Mode 1/2/3: shape (batch_size, q_num_heads, q_seq_len, total_kv_seq_len).
  int64_t qkMode = attentionOp.getQkMatmulOutputMode();
  if (qkMode != 0 && !isNoneValue(attentionOp.getQkMatmulOutput())) {
    IndexExpr qNumHeads, qSeqLen;
    if (is3DInput) {
      auto qNumHeadsAttr = attentionOp.getQNumHeads();
      if (!qNumHeadsAttr.has_value())
        return op->emitError(
            "q_num_heads attribute required for 3D input when "
            "qk_matmul_output_mode != 0");
      qNumHeads = LitIE(qNumHeadsAttr.value());
      qSeqLen = createIE->getShapeAsDim(Q, 1);
    } else {
      qNumHeads = createIE->getShapeAsDim(Q, 1);
      qSeqLen = createIE->getShapeAsDim(Q, 2);
    }

    // total_kv_seq_len depends on whether past_key is present.
    IndexExpr totalKvSeqLen;
    if (is3DInput) {
      totalKvSeqLen = createIE->getShapeAsDim(K, 1);
    } else {
      totalKvSeqLen = createIE->getShapeAsDim(K, 2);
    }
    if (hasPastKey) {
      IndexExpr pastSeqLen = createIE->getShapeAsDim(pastKey, 2);
      totalKvSeqLen = pastSeqLen + totalKvSeqLen;
    }

    DimsExpr qkDims = {batchSize, qNumHeads, qSeqLen, totalKvSeqLen};
    setOutputDims(qkDims, 3);
  } else {
    DimsExpr emptyDims;
    setOutputDims(emptyDims, 3);
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
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getQ()) || !hasShapeAndRank(getK()) ||
      !hasShapeAndRank(getV()))
    return success();

  Value pastKey = getPastKey();
  Value pastValue = getPastValue();
  if (!isNoneValue(pastKey) && !hasShapeAndRank(pastKey))
    return success();
  if (!isNoneValue(pastValue) && !hasShapeAndRank(pastValue))
    return success();

  Type elementType = mlir::cast<ShapedType>(getQ().getType()).getElementType();
  ONNXAttentionOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
