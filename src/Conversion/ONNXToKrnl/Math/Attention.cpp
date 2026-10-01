/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Attention.cpp - Lowering Attention Op ----------------===//
//
// Copyright 2024-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Attention Operator to Krnl dialect by decomposing
// it into basic ONNX operations (MatMul, Transpose, Softmax, Add, Mul, etc.).
// The resulting ONNX operations are then lowered to Krnl by their respective
// lowering patterns.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// Build a rank-1 (shape {1}) ONNX constant holding a single value of
// `elementType`, converted into that type's own float semantics. Used to
// build the "keep" (0.0) and "mask out" (large negative) values for the
// additive masks below.
static Value createScalarFloatConstant(
    MultiDialectBuilder<OnnxBuilder> &create, Type elementType, double value) {
  auto floatType = mlir::cast<FloatType>(elementType);
  APFloat f(value);
  bool losesInfo;
  f.convert(
      floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
  auto tensorType = RankedTensorType::get({1}, elementType);
  return create.onnx.constant(DenseElementsAttr::get(tensorType, {f}));
}

// Build a rank-1 int64 constant [0, 1, ..., n-1], then reshape it to
// `shape` (which must have exactly n elements total). Used to build the
// query/key position-index tensors used by the mask construction below.
static Value createReshapedRange(MultiDialectBuilder<OnnxBuilder> &create,
    Type i64Type, int64_t n, ArrayRef<int64_t> shape) {
  SmallVector<int64_t> vals(n);
  for (int64_t i = 0; i < n; ++i)
    vals[i] = i;
  Value range1D = create.onnx.constantInt64(vals);
  Type reshapedType = RankedTensorType::get(shape, i64Type);
  Value shapeConst = create.onnx.constantInt64(shape);
  return create.onnx.reshape(reshapedType, range1D, shapeConst);
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
// passing `none` through like the generic path below does.
//
// This function is intentionally self-contained (its own 3D->4D reshape,
// its own reshape-back) rather than sharing code with the generic
// `matchAndRewrite` below, so the KV-cache-specific logic stays easy to
// find and maintain on its own.
//
// Scope limits (not handled here, same as the generic path):
// - kv_num_heads vs q_num_heads (GQA/MQA): reshapes K/V using q_num_heads,
//   same simplification the generic path already makes.
// - softcap, softmax_precision, qk_matmul_output_mode are ignored.
// - Requires K's sequence length (the fixed cache size) and Q's sequence
//   length (the number of new query tokens) to be statically known;
//   otherwise returns failure().
static LogicalResult lowerFixedSizeKVCacheAttention(ONNXAttentionOp attentionOp,
    ONNXAttentionOpAdaptor adaptor, ConversionPatternRewriter &rewriter) {
  Location loc = attentionOp.getLoc();
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

  Value Q = adaptor.getQ();
  Value K = adaptor.getK();
  Value V = adaptor.getV();
  Value nonpadKvSeqlen = adaptor.getNonpadKvSeqlen();

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

    SmallVector<int64_t> qNewShape = {batchSize, qNumHeads, qSeqLen, headSize};
    Type qNewType = RankedTensorType::get(qNewShape, elementType);
    Value reshapeShapeQ = create.onnx.constantInt64(qNewShape);
    Q_reshaped = create.onnx.reshape(qNewType, Q, reshapeShapeQ);

    ShapedType kType = mlir::cast<ShapedType>(K.getType());
    ArrayRef<int64_t> kShape = kType.getShape();
    int64_t kSeqLen = kShape[1];
    int64_t kHiddenSize = kShape[2];
    if (ShapedType::isDynamic(kHiddenSize))
      return failure();
    SmallVector<int64_t> kNewShape = {
        batchSize, qNumHeads, kSeqLen, kHiddenSize / qNumHeads};
    Type kNewType = RankedTensorType::get(kNewShape, elementType);
    Value reshapeShapeK = create.onnx.constantInt64(kNewShape);
    K_reshaped = create.onnx.reshape(kNewType, K, reshapeShapeK);

    ShapedType vType = mlir::cast<ShapedType>(V.getType());
    ArrayRef<int64_t> vShape = vType.getShape();
    int64_t vHiddenSize = vShape[2];
    if (ShapedType::isDynamic(vHiddenSize))
      return failure();
    SmallVector<int64_t> vNewShape = {
        batchSize, qNumHeads, kSeqLen, vHiddenSize / qNumHeads};
    Type vNewType = RankedTensorType::get(vNewShape, elementType);
    Value reshapeShapeV = create.onnx.constantInt64(vNewShape);
    V_reshaped = create.onnx.reshape(vNewType, V, reshapeShapeV);
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

  // f16's max finite magnitude (~65504) is small enough that -1e9 would
  // overflow to -inf, which can turn into NaN once the causal and padding
  // masks are summed and then passed through Softmax's max-subtraction
  // stabilization on a fully-masked row. Use a smaller-magnitude value for
  // f16 to stay safely finite through that arithmetic.
  double negMaskValue = elementType.isF16() ? -1.0e4 : -1.0e9;
  Value zeroConst = createScalarFloatConstant(create, elementType, 0.0);
  Value negConst = createScalarFloatConstant(create, elementType, negMaskValue);

  Type i64Type = rewriter.getI64Type();
  Value kvPositions =
      createReshapedRange(create, i64Type, kvSeqLen, {1, 1, 1, kvSeqLen});

  Type nonpadReshapedType =
      RankedTensorType::get({batchDim, 1, 1, 1}, i64Type);
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
  Type paddingMaskType =
      RankedTensorType::get(paddingMaskShape, elementType);
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
    Value causalValid = ONNXLessOrEqualOp::create(
        rewriter, loc, boolCausalType, diff, offset);
    Type causalMaskType =
        RankedTensorType::get(causalMaskShape, elementType);
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
  auto scaleOpt = attentionOp.getScale();
  if (scaleOpt) {
    float scaleValue = scaleOpt->convertToFloat();
    if (scaleValue != 1.0f) {
      Value scaleConstant = create.onnx.constantFloat32({scaleValue});
      qk_scaled = create.onnx.mul(qk, scaleConstant);
    }
  }

  Value qk_masked = create.onnx.add(qk_scaled, attnMaskFinal);

  Value probs = ONNXSoftmaxOp::create(rewriter, loc, qk_masked.getType(),
      qk_masked, rewriter.getI64IntegerAttr(-1));

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
    SmallVector<int64_t> finalShape = {
        batchSize, qSeqLenOut, numHeads * headSize};
    Type finalType = RankedTensorType::get(finalShape, elementType);
    Value reshapeShapeFinal = create.onnx.constantInt64(finalShape);
    result_final = create.onnx.reshape(finalType, result, reshapeShapeFinal);
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
static LogicalResult lowerGrowingSizeKVCacheAttention(
    ONNXAttentionOp attentionOp, ONNXAttentionOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  Location loc = attentionOp.getLoc();
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

  Value Q = adaptor.getQ();
  Value K = adaptor.getK();
  Value V = adaptor.getV();
  Value attnMask = adaptor.getAttnMask();
  Value pastKey = adaptor.getPastKey();
  Value pastValue = adaptor.getPastValue();
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
    SmallVector<int64_t> qNewShape = {batchSize, qNumHeads, qSeqLen, headSize};
    Type qNewType = RankedTensorType::get(qNewShape, elementType);
    Value reshapeShapeQ = create.onnx.constantInt64(qNewShape);
    Q_reshaped = create.onnx.reshape(qNewType, Q, reshapeShapeQ);

    // Reshape K: (B, S', H) -> (B, qNumHeads, S', H/qNumHeads)
    ShapedType kType = mlir::cast<ShapedType>(K.getType());
    ArrayRef<int64_t> kShape = kType.getShape();
    int64_t kSeqLen = kShape[1];
    int64_t kHiddenSize = kShape[2];

    if (ShapedType::isDynamic(kHiddenSize)) {
      return failure();
    }

    SmallVector<int64_t> kNewShape = {
        batchSize, qNumHeads, kSeqLen, kHiddenSize / qNumHeads};
    Type kNewType = RankedTensorType::get(kNewShape, elementType);
    Value reshapeShapeK = create.onnx.constantInt64(kNewShape);
    K_reshaped = create.onnx.reshape(kNewType, K, reshapeShapeK);

    // Reshape V: (B, S', V_H) -> (B, qNumHeads, S', V_H/qNumHeads)
    ShapedType vType = mlir::cast<ShapedType>(V.getType());
    ArrayRef<int64_t> vShape = vType.getShape();
    int64_t vHiddenSize = vShape[2];

    if (ShapedType::isDynamic(vHiddenSize)) {
      return failure();
    }

    SmallVector<int64_t> vNewShape = {
        batchSize, qNumHeads, kSeqLen, vHiddenSize / qNumHeads};
    Type vNewType = RankedTensorType::get(vNewShape, elementType);
    Value reshapeShapeV = create.onnx.constantInt64(vNewShape);
    V_reshaped = create.onnx.reshape(vNewType, V, reshapeShapeV);
  }

  // Concatenate past_key with K if present
  if (hasPastKey) {
    ShapedType kShape = mlir::cast<ShapedType>(K_reshaped.getType());
    ShapedType pastKeyShape = mlir::cast<ShapedType>(pastKey.getType());
    int64_t newKSeqLen = ShapedType::isDynamic(kShape.getShape()[2]) ||
                                  ShapedType::isDynamic(
                                      pastKeyShape.getShape()[2])
                              ? ShapedType::kDynamic
                              : (kShape.getShape()[2] +
                                    pastKeyShape.getShape()[2]);
    SmallVector<int64_t> kConcatShape = {kShape.getShape()[0],
        kShape.getShape()[1], newKSeqLen, kShape.getShape()[3]};
    Type kConcatType = RankedTensorType::get(kConcatShape, elementType);
    K_reshaped = create.onnx.concat(kConcatType, {pastKey, K_reshaped}, 2);
  }

  // Concatenate past_value with V if present
  if (hasPastValue) {
    ShapedType vShape = mlir::cast<ShapedType>(V_reshaped.getType());
    ShapedType pastValueShape = mlir::cast<ShapedType>(pastValue.getType());
    int64_t newVSeqLen = ShapedType::isDynamic(vShape.getShape()[2]) ||
                                  ShapedType::isDynamic(
                                      pastValueShape.getShape()[2])
                              ? ShapedType::kDynamic
                              : (vShape.getShape()[2] +
                                    pastValueShape.getShape()[2]);
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
  auto scaleOpt = attentionOp.getScale();
  if (scaleOpt) {
    float scaleValue = scaleOpt->convertToFloat();
    if (scaleValue != 1.0f) {
      Value scaleConstant = create.onnx.constantFloat32({scaleValue});
      qk_scaled = create.onnx.mul(qk, scaleConstant);
    }
  }

  // Step 4: Add attention mask if present
  Value qk_masked = qk_scaled;
  if (!isNoneValue(attnMask)) {
    qk_masked = create.onnx.add(qk_scaled, attnMask);
  }

  // Step 5: Apply softmax over the last axis
  Value probs = ONNXSoftmaxOp::create(rewriter, loc, qk_masked.getType(),
      qk_masked, rewriter.getI64IntegerAttr(-1));

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

    SmallVector<int64_t> finalShape = {batchSize, qSeqLen, numHeads * headSize};
    Type finalType = RankedTensorType::get(finalShape, elementType);
    Value reshapeShapeFinal = create.onnx.constantInt64(finalShape);
    result_final = create.onnx.reshape(finalType, result, reshapeShapeFinal);
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

struct ONNXAttentionOpLowering : public OpConversionPattern<ONNXAttentionOp> {
  ONNXAttentionOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXAttentionOp>(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXAttentionOp attentionOp,
      ONNXAttentionOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Value attnMask = adaptor.getAttnMask();
    Value pastKey = adaptor.getPastKey();
    Value pastValue = adaptor.getPastValue();
    Value nonpadKvSeqlen = adaptor.getNonpadKvSeqlen();
    bool hasPastKey = !isNoneValue(pastKey);
    bool hasPastValue = !isNoneValue(pastValue);

    // Fixed-size KV cache pattern: the whole (padded) KV cache is passed
    // directly as K/V (no past_key/past_value to concatenate), and
    // nonpad_kv_seqlen says how many leading positions per batch are valid.
    // attn_mask is always None in this pattern.
    bool isFixedPattern = isNoneValue(attnMask) && !hasPastKey &&
                           !hasPastValue && !isNoneValue(nonpadKvSeqlen);

    // The --kv-cache option, when set, overrides the pattern-based choice
    // above. Default ("") means: use the pattern-based choice as-is.
    bool useFixed = isFixedPattern;
    if (!kvCache.empty()) {
      if (kvCache == "fixed") {
        useFixed = true;
      } else if (kvCache == "growing") {
        useFixed = false;
      } else {
        return attentionOp.emitOpError(
            "invalid --kv-cache option value '" + kvCache +
            "'; expected 'fixed' or 'growing'");
      }
    }

    if (useFixed)
      return lowerFixedSizeKVCacheAttention(attentionOp, adaptor, rewriter);
    return lowerGrowingSizeKVCacheAttention(attentionOp, adaptor, rewriter);
  }
};

void populateLoweringONNXAttentionOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXAttentionOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
