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

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXAttentionOpLowering : public OpConversionPattern<ONNXAttentionOp> {
  ONNXAttentionOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXAttentionOp>(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXAttentionOp attentionOp,
      ONNXAttentionOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
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
      SmallVector<int64_t> qNewShape = {
          batchSize, qNumHeads, qSeqLen, headSize};
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
    ShapedType kTransposedShape =
        mlir::cast<ShapedType>(K_transposed.getType());
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
    Value probs = rewriter.create<ONNXSoftmaxOp>(
        loc, qk_masked.getType(), qk_masked, rewriter.getI64IntegerAttr(-1));

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

      SmallVector<int64_t> finalShape = {
          batchSize, qSeqLen, numHeads * headSize};
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
};

void populateLoweringONNXAttentionOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXAttentionOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
