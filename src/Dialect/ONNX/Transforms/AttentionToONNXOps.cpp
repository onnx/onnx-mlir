/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ AttentionToONNXOps.cpp - Decompose ONNX Attention Op --------===//
//
// Copyright 2024-2026 The IBM Research Authors.
//
// =============================================================================
//
// See AttentionToONNXOps.hpp for an overview. This file decomposes
// onnx.Attention into basic ONNX operations (MatMul, Transpose, Softmax,
// Add, Mul, Less, Where, etc.).
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/Transforms/AttentionToONNXOps.hpp"

#include <cmath>

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Build a rank-1 (shape {1}) ONNX constant holding a single value of
// `elementType`, converted into that type's own float semantics. Used to
// build the "keep" (0.0) and "mask out" (large negative) values for the
// additive masks below.
Value createScalarFloatConstant(
    MultiDialectBuilder<OnnxBuilder> &create, Type elementType, double value) {
  auto floatType = mlir::cast<FloatType>(elementType);
  APFloat f(value);
  bool losesInfo;
  f.convert(
      floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
  auto tensorType = RankedTensorType::get({1}, elementType);
  return create.onnx.constant(DenseElementsAttr::get(tensorType, {f}));
}

// f16's max finite magnitude (~65504) is small enough that -1e9 would
// overflow to -inf, which can turn into NaN once multiple masks are summed
// and then passed through Softmax's max-subtraction stabilization on a
// fully-masked row. Use a smaller-magnitude value for f16 to stay safely
// finite through that arithmetic.
double negMaskValueFor(Type elementType) {
  return elementType.isF16() ? -1.0e4 : -1.0e9;
}

// Computes the effective multiplicative scale applied to Q@K^T: the `scale`
// attribute if present, else the spec's documented default of
// 1/sqrt(head_size) (head_size = Q's last dim after reshaping to 4D).
// Returns false (and leaves `scaleValue` unset) if `scale` is absent and
// head_size is not statically known, since the default can't be computed.
bool getEffectiveScale(
    ONNXAttentionOp attentionOp, ShapedType qShape4D, float &scaleValue) {
  auto scaleOpt = attentionOp.getScale();
  if (scaleOpt) {
    scaleValue = scaleOpt->convertToFloat();
    return true;
  }
  int64_t headSize = qShape4D.getShape()[3];
  if (ShapedType::isDynamic(headSize))
    return false;
  scaleValue = 1.0f / std::sqrt(static_cast<float>(headSize));
  return true;
}

// Reshape a (batch, seq, hidden) tensor into (batch, numHeads, seq,
// headSize), per the spec: split hidden into (numHeads, headSize) as
// trailing dims via Reshape to (batch, seq, numHeads, headSize), THEN move
// numHeads before seq via Transpose([0, 2, 1, 3]). A single Reshape
// straight to (batch, numHeads, seq, headSize) is NOT equivalent -- it
// would reinterpret the flat buffer without moving any data, scrambling
// which (seq, head) pair each element belongs to.
Value reshapeToMultiHead4D(MultiDialectBuilder<OnnxBuilder> &create, Value v,
    int64_t batchSize, int64_t seqLen, int64_t numHeads, int64_t headSize,
    Type elementType) {
  SmallVector<int64_t> intermediateShape = {
      batchSize, seqLen, numHeads, headSize};
  Type intermediateType = RankedTensorType::get(intermediateShape, elementType);
  Value shapeConst = create.onnx.constantInt64(intermediateShape);
  Value intermediate = create.onnx.reshape(intermediateType, v, shapeConst);
  SmallVector<int64_t> perm = {0, 2, 1, 3};
  return create.onnx.transposeInt64(intermediate, perm);
}

// Inverse of reshapeToMultiHead4D: (batch, numHeads, seq, headSize) ->
// (batch, seq, numHeads * headSize). Transpose back to (batch, seq,
// numHeads, headSize) BEFORE flattening the trailing two dims with
// Reshape, for the same reason as above.
Value reshapeFromMultiHead4D(MultiDialectBuilder<OnnxBuilder> &create, Value v,
    int64_t batchSize, int64_t seqLen, int64_t numHeads, int64_t headSize,
    Type elementType) {
  SmallVector<int64_t> perm = {0, 2, 1, 3};
  Value transposed = create.onnx.transposeInt64(v, perm);
  SmallVector<int64_t> finalShape = {batchSize, seqLen, numHeads * headSize};
  Type finalType = RankedTensorType::get(finalShape, elementType);
  Value shapeConst = create.onnx.constantInt64(finalShape);
  return create.onnx.reshape(finalType, transposed, shapeConst);
}

// Build a rank-1 int64 constant [0, 1, ..., n-1], then reshape it to
// `shape` (which must have exactly n elements total). Used to build the
// query/key position-index tensors used by the mask construction below.
Value createReshapedRange(MultiDialectBuilder<OnnxBuilder> &create,
    Type i64Type, int64_t n, ArrayRef<int64_t> shape) {
  SmallVector<int64_t> vals(n);
  for (int64_t i = 0; i < n; ++i)
    vals[i] = i;
  Value range1D = create.onnx.constantInt64(vals);
  Type reshapedType = RankedTensorType::get(shape, i64Type);
  Value shapeConst = create.onnx.constantInt64(shape);
  return create.onnx.reshape(reshapedType, range1D, shapeConst);
}

// Build a (qSeqLen, kvSeqLen) additive causal-mask constant for the
// "scalar offset" causal case used by lowerGrowingSizeKVCacheAttention:
// query position i (0-indexed in the current Q chunk) may attend kv
// position j (0-indexed in the full, already-concatenated K) iff
// j <= i + offset, where offset is the number of KV positions that
// preceded this Q chunk (0 with no past_key, or past_key's sequence length
// when continuing a cache). Unlike lowerFixedSizeKVCacheAttention's causal
// mask, offset here is the same for every batch element, so the whole
// (qSeqLen, kvSeqLen) mask is a compile-time constant -- no Less/Where ops
// needed.
Value createScalarOffsetCausalMaskConstant(
    MultiDialectBuilder<OnnxBuilder> &create, Type elementType, int64_t qSeqLen,
    int64_t kvSeqLen, int64_t offset) {
  auto floatType = mlir::cast<FloatType>(elementType);
  APFloat zeroF(0.0), negF(negMaskValueFor(elementType));
  bool losesInfo;
  zeroF.convert(
      floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
  negF.convert(
      floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
  SmallVector<APFloat> vals;
  vals.reserve(qSeqLen * kvSeqLen);
  for (int64_t i = 0; i < qSeqLen; ++i)
    for (int64_t j = 0; j < kvSeqLen; ++j)
      vals.push_back(j <= i + offset ? zeroF : negF);
  auto causalType = RankedTensorType::get({qSeqLen, kvSeqLen}, elementType);
  return create.onnx.constant(
      DenseElementsAttr::get(causalType, ArrayRef<APFloat>(vals)));
}

// Lower the "fixed-size KV cache" input pattern of onnx.Attention:
// (Q, K, V, attn_mask = None, past_key = None, past_value = None,
//  nonpad_kv_seqlen = <given>).
//
// Here K/V are already the full, fixed-size padded KV cache (cache update
// happens outside this op, per the ONNX Attention spec), and
// `nonpad_kv_seqlen` (shape (batch_size,), i64) gives, per batch, how many
// leading positions of that cache are valid (non-padding). Since attn_mask
// is always None in this pattern, this function synthesizes the additive
// mask itself from `is_causal` and `nonpad_kv_seqlen`, instead of just
// passing `none` through like lowerGrowingSizeKVCacheAttention does.
//
// This function is intentionally self-contained (its own 3D->4D reshape,
// its own reshape-back) rather than sharing code with
// lowerGrowingSizeKVCacheAttention below, so the KV-cache-specific logic
// stays easy to find and maintain on its own.
//
// Scope limits (not handled here, same as the generic path):
// - kv_num_heads vs q_num_heads (GQA/MQA): reshapes K/V using q_num_heads,
//   same simplification the generic path already makes. Rejected with an
//   explicit error by checkSupportedAttentionAttributes() below rather than
//   silently mis-lowered when kv_num_heads != q_num_heads.
// - softcap, softmax_precision, qk_matmul_output_mode: not implemented;
//   also rejected with an explicit error by
//   checkSupportedAttentionAttributes() when set to a non-default value.
// - Requires K's sequence length (the fixed cache size) and Q's sequence
//   length (the number of new query tokens) to be statically known;
//   otherwise returns failure().
LogicalResult lowerFixedSizeKVCacheAttention(ONNXAttentionOp attentionOp,
    Value Q, Value K, Value V, Value nonpadKvSeqlen,
    PatternRewriter &rewriter) {
  Location loc = attentionOp.getLoc();
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

  ShapedType qType = mlir::cast<ShapedType>(Q.getType());
  int64_t inputRank = qType.getShape().size();
  bool is3DInput = (inputRank == 3);
  Type elementType = qType.getElementType();

  Value Q_reshaped = Q;
  Value K_reshaped = K;
  Value V_reshaped = V;

  if (is3DInput) {
    auto qNumHeadsAttr = attentionOp.getQNumHeads();
    int64_t qNumHeads = qNumHeadsAttr.has_value() ? qNumHeadsAttr.value() : 1;

    ArrayRef<int64_t> qShape = qType.getShape();
    int64_t batchSize = qShape[0];
    int64_t qSeqLen = qShape[1];
    int64_t qHiddenSize = qShape[2];
    if (ShapedType::isDynamic(qHiddenSize))
      return failure();
    int64_t headSize = qHiddenSize / qNumHeads;

    Q_reshaped = reshapeToMultiHead4D(
        create, Q, batchSize, qSeqLen, qNumHeads, headSize, elementType);

    ShapedType kType = mlir::cast<ShapedType>(K.getType());
    ArrayRef<int64_t> kShape = kType.getShape();
    int64_t kSeqLen = kShape[1];
    int64_t kHiddenSize = kShape[2];
    if (ShapedType::isDynamic(kHiddenSize))
      return failure();
    K_reshaped = reshapeToMultiHead4D(create, K, batchSize, kSeqLen, qNumHeads,
        kHiddenSize / qNumHeads, elementType);

    ShapedType vType = mlir::cast<ShapedType>(V.getType());
    ArrayRef<int64_t> vShape = vType.getShape();
    int64_t vHiddenSize = vShape[2];
    if (ShapedType::isDynamic(vHiddenSize))
      return failure();
    V_reshaped = reshapeToMultiHead4D(create, V, batchSize, kSeqLen, qNumHeads,
        vHiddenSize / qNumHeads, elementType);
  }

  ShapedType qShape4D = mlir::cast<ShapedType>(Q_reshaped.getType());
  ShapedType kShape4D = mlir::cast<ShapedType>(K_reshaped.getType());
  int64_t batchDim = qShape4D.getShape()[0];
  int64_t qSeqLen = qShape4D.getShape()[2];
  int64_t kvSeqLen = kShape4D.getShape()[2];
  // "Fixed-size KV cache" means the cache's (padded) sequence length is
  // statically known; qSeqLen (the number of new query tokens) must also be
  // static so the position-index constants below can be built.
  if (ShapedType::isDynamic(kvSeqLen) || ShapedType::isDynamic(qSeqLen))
    return failure();

  Value zeroConst = createScalarFloatConstant(create, elementType, 0.0);
  Value negConst = createScalarFloatConstant(
      create, elementType, negMaskValueFor(elementType));

  Type i64Type = rewriter.getI64Type();
  Value kvPositions =
      createReshapedRange(create, i64Type, kvSeqLen, {1, 1, 1, kvSeqLen});

  Type nonpadReshapedType = RankedTensorType::get({batchDim, 1, 1, 1}, i64Type);
  Value nonpadReshapeShape = create.onnx.constantInt64({-1, 1, 1, 1});
  Value nonpadReshaped = create.onnx.reshape(
      nonpadReshapedType, nonpadKvSeqlen, nonpadReshapeShape);

  // Padding mask: 0.0 where kv position j < nonpad_kv_seqlen[b], else
  // negMaskValue. Always built: needed whenever is_causal == 0, and
  // harmless-but-redundant when is_causal == 1 (see note below).
  SmallVector<int64_t> paddingMaskShape = {batchDim, 1, 1, kvSeqLen};
  Type boolPaddingType =
      RankedTensorType::get(paddingMaskShape, rewriter.getI1Type());
  Value validKv = ONNXLessOp::create(
      rewriter, loc, boolPaddingType, kvPositions, nonpadReshaped);
  Type paddingMaskType = RankedTensorType::get(paddingMaskShape, elementType);
  Value attnMaskFinal =
      create.onnx.where(paddingMaskType, validKv, zeroConst, negConst);

  bool isCausal = attentionOp.getIsCausal() != 0;
  if (isCausal) {
    // Alignment: query position i (0-indexed within this qSeqLen-token
    // chunk) is actually at absolute sequence position
    // nonpad_kv_seqlen[b] - qSeqLen + i. So kv position j is allowed iff
    // j <= nonpad_kv_seqlen[b] - qSeqLen + i, i.e. (j - i) <=
    // nonpad_kv_seqlen[b] - qSeqLen. Note that at i = qSeqLen - 1 (the
    // newest query) this bound is j <= nonpad_kv_seqlen[b] - 1 <
    // nonpad_kv_seqlen[b], so the causal mask alone already implies the
    // padding mask for every i; both are still added for robustness rather
    // than relying on that implication.
    Value qPositions =
        createReshapedRange(create, i64Type, qSeqLen, {1, 1, qSeqLen, 1});
    SmallVector<int64_t> diffShape = {1, 1, qSeqLen, kvSeqLen};
    Type diffType = RankedTensorType::get(diffShape, i64Type);
    Value diff =
        ONNXSubOp::create(rewriter, loc, diffType, kvPositions, qPositions);

    Value qSeqLenConst = create.onnx.constantInt64({qSeqLen});
    Type offsetType = RankedTensorType::get({batchDim, 1, 1, 1}, i64Type);
    Value offset = ONNXSubOp::create(
        rewriter, loc, offsetType, nonpadReshaped, qSeqLenConst);

    SmallVector<int64_t> causalMaskShape = {batchDim, 1, qSeqLen, kvSeqLen};
    Type boolCausalType =
        RankedTensorType::get(causalMaskShape, rewriter.getI1Type());
    Value causalValid =
        ONNXLessOrEqualOp::create(rewriter, loc, boolCausalType, diff, offset);
    Type causalMaskType = RankedTensorType::get(causalMaskShape, elementType);
    Value causalMask =
        create.onnx.where(causalMaskType, causalValid, zeroConst, negConst);

    attnMaskFinal = create.onnx.add(attnMaskFinal, causalMask);
  }

  // Transpose K: (B, num_heads, kv_seq_len, head_size) -> (B, num_heads,
  // head_size, kv_seq_len).
  SmallVector<int64_t> kTransposePerm = {0, 1, 3, 2};
  Value K_transposed = create.onnx.transposeInt64(K_reshaped, kTransposePerm);

  ShapedType kTransposedShape = mlir::cast<ShapedType>(K_transposed.getType());
  SmallVector<int64_t> qkShape = {qShape4D.getShape()[0],
      qShape4D.getShape()[1], qShape4D.getShape()[2],
      kTransposedShape.getShape()[3]};
  Type qkType = RankedTensorType::get(qkShape, elementType);
  Value qk = create.onnx.matmul(qkType, Q_reshaped, K_transposed);

  Value qk_scaled = qk;
  float scaleValue;
  if (!getEffectiveScale(attentionOp, qShape4D, scaleValue))
    return failure();
  if (scaleValue != 1.0f) {
    Value scaleConstant =
        createScalarFloatConstant(create, elementType, scaleValue);
    qk_scaled = create.onnx.mul(qk, scaleConstant);
  }

  Value qk_masked = create.onnx.add(qk_scaled, attnMaskFinal);

  Value probs =
      ONNXSoftmaxOp::create(rewriter, loc, qk_masked.getType(), qk_masked,
          IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true), -1));

  ShapedType vShape4D = mlir::cast<ShapedType>(V_reshaped.getType());
  SmallVector<int64_t> outputShape = {qShape4D.getShape()[0],
      qShape4D.getShape()[1], qShape4D.getShape()[2], vShape4D.getShape()[3]};
  Type outputType4D = RankedTensorType::get(outputShape, elementType);
  Value result = create.onnx.matmul(outputType4D, probs, V_reshaped);

  Value result_final = result;
  if (is3DInput) {
    ShapedType resultType = mlir::cast<ShapedType>(result.getType());
    ArrayRef<int64_t> resultShape = resultType.getShape();
    int64_t batchSize = resultShape[0];
    int64_t numHeads = resultShape[1];
    int64_t qSeqLenOut = resultShape[2];
    int64_t headSize = resultShape[3];
    result_final = reshapeFromMultiHead4D(
        create, result, batchSize, qSeqLenOut, numHeads, headSize, elementType);
  }

  Value noneVal = create.onnx.none();
  SmallVector<Value, 4> replacementValues;
  replacementValues.push_back(result_final); // Result 0: Y
  replacementValues.push_back(noneVal);      // Result 1: present_key
  replacementValues.push_back(noneVal);      // Result 2: present_value
  replacementValues.push_back(noneVal);      // Result 3: qk_matmul_output

  rewriter.replaceOp(attentionOp, replacementValues);
  return success();
}

// Lower the "growing KV cache" (original/generic) input pattern of
// onnx.Attention: cache update happens inside the op, so K/V are only the
// incoming tokens for this step, and past_key/past_value (when given) are
// concatenated onto them; attn_mask, when given, is added as-is (no masking
// is synthesized here, unlike lowerFixedSizeKVCacheAttention above).
LogicalResult lowerGrowingSizeKVCacheAttention(ONNXAttentionOp attentionOp,
    Value Q, Value K, Value V, Value attnMask, Value pastKey, Value pastValue,
    PatternRewriter &rewriter) {
  Location loc = attentionOp.getLoc();
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

  bool hasPastKey = !isNoneValue(pastKey);
  bool hasPastValue = !isNoneValue(pastValue);

  // Get input rank to determine if it's 3D or 4D
  ShapedType qType = mlir::cast<ShapedType>(Q.getType());
  int64_t inputRank = qType.getShape().size();
  bool is3DInput = (inputRank == 3);
  Type elementType = qType.getElementType();

  Value Q_reshaped = Q;
  Value K_reshaped = K;
  Value V_reshaped = V;

  if (is3DInput) {
    // For 3D inputs, reshape to 4D for attention computation
    auto qNumHeadsAttr = attentionOp.getQNumHeads();
    int64_t qNumHeads = 1;
    if (qNumHeadsAttr.has_value()) {
      qNumHeads = qNumHeadsAttr.value();
    }

    ArrayRef<int64_t> qShape = qType.getShape();
    int64_t batchSize = qShape[0];
    int64_t qSeqLen = qShape[1];
    int64_t qHiddenSize = qShape[2];

    if (ShapedType::isDynamic(qHiddenSize)) {
      return failure();
    }

    int64_t headSize = qHiddenSize / qNumHeads;

    // Reshape Q: (B, S, H) -> (B, qNumHeads, S, H/qNumHeads)
    Q_reshaped = reshapeToMultiHead4D(
        create, Q, batchSize, qSeqLen, qNumHeads, headSize, elementType);

    // Reshape K: (B, S', H) -> (B, qNumHeads, S', H/qNumHeads)
    ShapedType kType = mlir::cast<ShapedType>(K.getType());
    ArrayRef<int64_t> kShape = kType.getShape();
    int64_t kSeqLen = kShape[1];
    int64_t kHiddenSize = kShape[2];

    if (ShapedType::isDynamic(kHiddenSize)) {
      return failure();
    }

    K_reshaped = reshapeToMultiHead4D(create, K, batchSize, kSeqLen, qNumHeads,
        kHiddenSize / qNumHeads, elementType);

    // Reshape V: (B, S', V_H) -> (B, qNumHeads, S', V_H/qNumHeads)
    ShapedType vType = mlir::cast<ShapedType>(V.getType());
    ArrayRef<int64_t> vShape = vType.getShape();
    int64_t vHiddenSize = vShape[2];

    if (ShapedType::isDynamic(vHiddenSize)) {
      return failure();
    }

    V_reshaped = reshapeToMultiHead4D(create, V, batchSize, kSeqLen, qNumHeads,
        vHiddenSize / qNumHeads, elementType);
  }

  // Concatenate past_key with K if present
  if (hasPastKey) {
    ShapedType kShape = mlir::cast<ShapedType>(K_reshaped.getType());
    ShapedType pastKeyShape = mlir::cast<ShapedType>(pastKey.getType());
    int64_t newKSeqLen =
        ShapedType::isDynamic(kShape.getShape()[2]) ||
                ShapedType::isDynamic(pastKeyShape.getShape()[2])
            ? ShapedType::kDynamic
            : (kShape.getShape()[2] + pastKeyShape.getShape()[2]);
    SmallVector<int64_t> kConcatShape = {kShape.getShape()[0],
        kShape.getShape()[1], newKSeqLen, kShape.getShape()[3]};
    Type kConcatType = RankedTensorType::get(kConcatShape, elementType);
    K_reshaped = create.onnx.concat(kConcatType, {pastKey, K_reshaped}, 2);
  }

  // Concatenate past_value with V if present
  if (hasPastValue) {
    ShapedType vShape = mlir::cast<ShapedType>(V_reshaped.getType());
    ShapedType pastValueShape = mlir::cast<ShapedType>(pastValue.getType());
    int64_t newVSeqLen =
        ShapedType::isDynamic(vShape.getShape()[2]) ||
                ShapedType::isDynamic(pastValueShape.getShape()[2])
            ? ShapedType::kDynamic
            : (vShape.getShape()[2] + pastValueShape.getShape()[2]);
    SmallVector<int64_t> vConcatShape = {vShape.getShape()[0],
        vShape.getShape()[1], newVSeqLen, vShape.getShape()[3]};
    Type vConcatType = RankedTensorType::get(vConcatShape, elementType);
    V_reshaped = create.onnx.concat(vConcatType, {pastValue, V_reshaped}, 2);
  }

  // Step 1: Transpose K: (B, num_heads, seq_len, head_size) -> (B,
  // num_heads, head_size, seq_len)
  SmallVector<int64_t> kTransposePerm = {0, 1, 3, 2};
  Value K_transposed = create.onnx.transposeInt64(K_reshaped, kTransposePerm);

  // Step 2: MatMul(Q, K^T)
  ShapedType qShape4D = mlir::cast<ShapedType>(Q_reshaped.getType());
  ShapedType kTransposedShape = mlir::cast<ShapedType>(K_transposed.getType());
  SmallVector<int64_t> qkShape = {qShape4D.getShape()[0],
      qShape4D.getShape()[1], qShape4D.getShape()[2],
      kTransposedShape.getShape()[3]};
  Type qkType = RankedTensorType::get(qkShape, elementType);
  Value qk = create.onnx.matmul(qkType, Q_reshaped, K_transposed);

  // Step 3: Apply scaling if needed
  Value qk_scaled = qk;
  float scaleValue;
  if (!getEffectiveScale(attentionOp, qShape4D, scaleValue))
    return failure();
  if (scaleValue != 1.0f) {
    Value scaleConstant =
        createScalarFloatConstant(create, elementType, scaleValue);
    qk_scaled = create.onnx.mul(qk, scaleConstant);
  }

  // Step 4: Add attention mask if present, then a causal mask on top if
  // is_causal is set (the two combine additively, matching the ONNX
  // reference implementation).
  Value qk_masked = qk_scaled;
  if (!isNoneValue(attnMask)) {
    // attn_mask's kv dimension is allowed by the spec to be shorter than
    // K/V's actual (padded) sequence length when nonpad_kv_seqlen is also
    // given, which this function does not implement (only
    // lowerFixedSizeKVCacheAttention above handles that case, where
    // attn_mask itself is None). Reject the mismatch explicitly instead of
    // letting the Add below hit a broadcast-shape assertion.
    ShapedType attnMaskType = mlir::cast<ShapedType>(attnMask.getType());
    int64_t maskKvLen = attnMaskType.getShape().back();
    int64_t kvSeqLenForMask =
        mlir::cast<ShapedType>(K_reshaped.getType()).getShape()[2];
    if (!ShapedType::isDynamic(maskKvLen) &&
        !ShapedType::isDynamic(kvSeqLenForMask) && maskKvLen != 1 &&
        maskKvLen != kvSeqLenForMask)
      return attentionOp.emitOpError(
          "unsupported: attn_mask's kv dimension (" +
          std::to_string(maskKvLen) +
          ") does not match K/V's sequence length (" +
          std::to_string(kvSeqLenForMask) +
          "); a shorter attn_mask combined with nonpad_kv_seqlen is not "
          "implemented");

    // attn_mask is either a boolean mask (True = take part in attention) or
    // a float bias of Q/K/V's element type added directly to the scores.
    // Convert the boolean case to an additive bias; anything else (an
    // integer type other than i1) is not implemented.
    Value additiveMask = attnMask;
    Type maskElemType = attnMaskType.getElementType();
    if (maskElemType.isInteger(1)) {
      Value zeroConst = createScalarFloatConstant(create, elementType, 0.0);
      Value negConst = createScalarFloatConstant(
          create, elementType, negMaskValueFor(elementType));
      Type additiveMaskType =
          RankedTensorType::get(attnMaskType.getShape(), elementType);
      additiveMask =
          create.onnx.where(additiveMaskType, attnMask, zeroConst, negConst);
    } else if (maskElemType != elementType) {
      return attentionOp.emitOpError(
          "unsupported: attn_mask element type must be either i1 (boolean) "
          "or the same float type as Q/K/V; other integer attn_mask types "
          "are not implemented");
    }
    qk_masked = create.onnx.add(qk_scaled, additiveMask);
  }
  if (attentionOp.getIsCausal() != 0) {
    // The causal frontier is anchored to the end of the (possibly
    // past_key-extended) KV sequence: query i (0-indexed in this Q chunk)
    // may attend kv position j (0-indexed in the full, post-concat K) iff
    // j <= i + offset, where offset is the number of KV positions already
    // in the cache before this chunk (0 with no past_key, else past_key's
    // sequence length).
    int64_t qSeqLenForCausal = qShape4D.getShape()[2];
    int64_t kvSeqLenForCausal =
        mlir::cast<ShapedType>(K_reshaped.getType()).getShape()[2];
    if (ShapedType::isDynamic(qSeqLenForCausal) ||
        ShapedType::isDynamic(kvSeqLenForCausal))
      return failure();
    int64_t offset = 0;
    if (hasPastKey) {
      int64_t pastKeySeqLen =
          mlir::cast<ShapedType>(pastKey.getType()).getShape()[2];
      if (ShapedType::isDynamic(pastKeySeqLen))
        return failure();
      offset = pastKeySeqLen;
    }
    Value causalMask = createScalarOffsetCausalMaskConstant(
        create, elementType, qSeqLenForCausal, kvSeqLenForCausal, offset);
    qk_masked = create.onnx.add(qk_masked, causalMask);
  }

  // Step 5: Apply softmax over the last axis
  Value probs =
      ONNXSoftmaxOp::create(rewriter, loc, qk_masked.getType(), qk_masked,
          IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true), -1));

  // Step 6: MatMul(softmax(...), V)
  ShapedType vShape4D = mlir::cast<ShapedType>(V_reshaped.getType());
  SmallVector<int64_t> outputShape = {qShape4D.getShape()[0],
      qShape4D.getShape()[1], qShape4D.getShape()[2], vShape4D.getShape()[3]};
  Type outputType4D = RankedTensorType::get(outputShape, elementType);
  Value result = create.onnx.matmul(outputType4D, probs, V_reshaped);

  // Step 7: Reshape back to 3D if input was 3D
  Value result_final = result;
  if (is3DInput) {
    ShapedType resultType = mlir::cast<ShapedType>(result.getType());
    ArrayRef<int64_t> resultShape = resultType.getShape();
    int64_t batchSize = resultShape[0];
    int64_t numHeads = resultShape[1];
    int64_t qSeqLen = resultShape[2];
    int64_t headSize = resultShape[3];
    result_final = reshapeFromMultiHead4D(
        create, result, batchSize, qSeqLen, numHeads, headSize, elementType);
  }

  // Create the none value for optional outputs
  Value noneVal = create.onnx.none();

  // Replace all 4 outputs of the AttentionOp
  SmallVector<Value, 4> replacementValues;
  replacementValues.push_back(result_final); // Result 0: Y

  // Result 1: present_key - return concatenated K if past_key was used, else
  // none Note: K_reshaped is either the original K (if no past) or K
  // concatenated with past_key (if past was used)
  replacementValues.push_back(hasPastKey ? K_reshaped : noneVal);

  // Result 2: present_value - return concatenated V if past_value was used,
  // else none Note: V_reshaped is either the original V (if no past) or V
  // concatenated with past_value (if past was used)
  replacementValues.push_back(hasPastValue ? V_reshaped : noneVal);

  // Result 3: qk_matmul_output - not computed in this lowering
  replacementValues.push_back(noneVal);

  rewriter.replaceOp(attentionOp, replacementValues);
  return success();
}

// Reject attribute combinations that neither lowering function above
// correctly implements, instead of silently ignoring them and producing
// wrong results. Both paths share these gaps (see the "Scope limits" note
// on lowerFixedSizeKVCacheAttention above).
LogicalResult checkSupportedAttentionAttributes(ONNXAttentionOp attentionOp) {
  // GQA/MQA (kv_num_heads < q_num_heads): both lowering functions reshape
  // K/V using q_num_heads, which is only correct when kv_num_heads equals
  // q_num_heads (plain MHA).
  auto kvNumHeadsAttr = attentionOp.getKvNumHeads();
  if (kvNumHeadsAttr.has_value()) {
    auto qNumHeadsAttr = attentionOp.getQNumHeads();
    int64_t qNumHeads = qNumHeadsAttr.has_value() ? qNumHeadsAttr.value() : 1;
    if (kvNumHeadsAttr.value() != qNumHeads)
      return attentionOp.emitOpError(
          "unsupported: kv_num_heads (" +
          std::to_string(kvNumHeadsAttr.value()) + ") != q_num_heads (" +
          std::to_string(qNumHeads) +
          "); grouped/multi-query attention (GQA/MQA) is not implemented");
  }

  // For already-4D inputs there is no kv_num_heads/q_num_heads attribute to
  // check: the head count is just dim 1 of Q/K's shape. Catch a GQA/MQA
  // mismatch there too, instead of letting it reach a lowering that assumes
  // Q and K share the same head count and crashes on the shape mismatch.
  auto qType = mlir::dyn_cast<ShapedType>(attentionOp.getQ().getType());
  auto kType = mlir::dyn_cast<ShapedType>(attentionOp.getK().getType());
  if (qType && kType && qType.getRank() == 4 && kType.getRank() == 4) {
    int64_t qHeads = qType.getShape()[1];
    int64_t kHeads = kType.getShape()[1];
    if (!ShapedType::isDynamic(qHeads) && !ShapedType::isDynamic(kHeads) &&
        qHeads != kHeads)
      return attentionOp.emitOpError(
          "unsupported: K/V's num_heads (" + std::to_string(kHeads) +
          ") != Q's num_heads (" + std::to_string(qHeads) +
          "); grouped/multi-query attention (GQA/MQA) is not implemented");
  }

  if (attentionOp.getSoftcap().convertToFloat() != 0.0f)
    return attentionOp.emitOpError(
        "unsupported: non-zero softcap attribute is not implemented");

  if (attentionOp.getQkMatmulOutputMode() != 0)
    return attentionOp.emitOpError(
        "unsupported: qk_matmul_output_mode != 0 is not implemented "
        "(qk_matmul_output is never computed by this lowering)");

  if (attentionOp.getSoftmaxPrecision().has_value())
    return attentionOp.emitOpError(
        "unsupported: softmax_precision attribute is not implemented");

  return success();
}

} // namespace

LogicalResult lowerONNXAttentionOp(ONNXAttentionOp attentionOp, Value Q,
    Value K, Value V, Value attnMask, Value pastKey, Value pastValue,
    Value nonpadKvSeqlen, PatternRewriter &rewriter) {
  if (failed(checkSupportedAttentionAttributes(attentionOp)))
    return failure();

  bool hasPastKey = !isNoneValue(pastKey);
  bool hasPastValue = !isNoneValue(pastValue);

  // Fixed-size KV cache pattern: the whole (padded) KV cache is passed
  // directly as K/V (no past_key/past_value to concatenate), and
  // nonpad_kv_seqlen says how many leading positions per batch are valid.
  // attn_mask is always None in this pattern.
  bool isFixedPattern = isNoneValue(attnMask) && !hasPastKey && !hasPastValue &&
                        !isNoneValue(nonpadKvSeqlen);

  // The --kv-cache option, when set, overrides the pattern-based choice
  // above. Default ("") means: use the pattern-based choice as-is.
  bool useFixed = isFixedPattern;
  if (kvCache != KVCacheType::Undefined) {
    if (kvCache == KVCacheType::Fixed) {
      if (!useFixed)
        return attentionOp.emitOpError(
            "Unaccepted --kv-cache option value; since the input of the op "
            "is not for fixed cache");
      // else: useFixed stays true, proceed.
    } else if (kvCache == KVCacheType::Grow) {
      // Will have more implementation in future
      useFixed = false;
    } else {
      return attentionOp.emitOpError(
          "invalid --kv-cache option value; expected 'Fixed' or 'Grow'");
    }
  }

  if (useFixed)
    return lowerFixedSizeKVCacheAttention(
        attentionOp, Q, K, V, nonpadKvSeqlen, rewriter);
  return lowerGrowingSizeKVCacheAttention(
      attentionOp, Q, K, V, attnMask, pastKey, pastValue, rewriter);
}

} // namespace onnx_mlir
