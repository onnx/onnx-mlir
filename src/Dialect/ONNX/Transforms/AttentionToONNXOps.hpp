/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ AttentionToONNXOps.hpp - Decompose ONNX Attention Op --------===//
//
// Copyright 2024-2026 The IBM Research Authors.
//
// =============================================================================
//
// Shared logic to decompose onnx.Attention into a sequence of basic ONNX ops
// (Reshape, Transpose, MatMul, Softmax, Add, Mul, Less, Where, etc.). Used by
// ONNXToKrnl conversion, or Accelarator lowering.
// This decompose can not be done in the onnx-decompose pass because
// the decision is backend dependent.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ATTENTION_TO_ONNX_OPS_H
#define ONNX_MLIR_ATTENTION_TO_ONNX_OPS_H

#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

// Decompose `attentionOp` into basic ONNX ops and replace it, dispatching
// between the "fixed-size KV cache" pattern (attn_mask/past_key/past_value
// all None, nonpad_kv_seqlen given) and the "growing KV cache" pattern
// (cache update, if any, happens via past_key/past_value concatenation
// inside the op). The choice can be overridden by the `--kv-cache` compiler
// option (`onnx_mlir::kvCache`, values "fixed"/"growing"); any other
// non-empty value is reported as an op error.
//
// `Q`/`K`/`V`/`attnMask`/`pastKey`/`pastValue`/`nonpadKvSeqlen` are the
// op's operands as the caller sees them (already remapped by a conversion
// adaptor if applicable); `attentionOp` is only used for its attributes
// (is_causal, q_num_heads, scale, ...) and location.
mlir::LogicalResult lowerONNXAttentionOp(mlir::ONNXAttentionOp attentionOp,
    mlir::Value Q, mlir::Value K, mlir::Value V, mlir::Value attnMask,
    mlir::Value pastKey, mlir::Value pastValue, mlir::Value nonpadKvSeqlen,
    mlir::PatternRewriter &rewriter);

} // namespace onnx_mlir
#endif
