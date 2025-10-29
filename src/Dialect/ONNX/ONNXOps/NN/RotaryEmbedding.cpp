// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.

// =============================================================================
//
// This file provides definition of ONNX dialect RotaryEmbedding operation.
//
//===----------------------------------------------------------------------===//

#include "ONNXOps.hpp"

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXRotaryEmbeddingOpShapeHelper::computeShape() {
  ONNXRotaryEmbeddingOpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getX());
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXRotaryEmbeddingOp::verify() {
  ONNXRotaryEmbeddingOpAdaptor adaptor(*this);
  Value input = adaptor.getX();
  if (!hasShapeAndRank(input))
    return success(); // Won't be able to do any checking at this stage.

  auto inputType = mlir::cast<ShapedType>(input.getType());
  int64_t inputRank = inputType.getRank();

  if (inputRank != 3 && inputRank != 4)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), input, inputRank, "3 or 4");

  auto numHeads = adaptor.getNumHeads();
  if (inputRank == 3 && !numHeads)
    return emitOpError(
        "attribute 'num_heads' must be provided when input is a 3D tensor.");

  // Check hidden_size divisible by num_heads
  if (inputType.hasStaticShape()) {
    auto inputShape = inputType.getShape();
    if (inputRank == 3 && numHeads && inputShape[2] % *numHeads != 0)
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), input, 2, inputShape[2],
          "divisible by " + std::to_string(*numHeads));
  }

  Value cosCache = adaptor.getCosCache();
  Value sinCache = adaptor.getCosCache();
  if (!hasShapeAndRank(cosCache) || !hasShapeAndRank(sinCache))
    return success(); // Won't be able to do any more checking at this stage.

  auto cosCacheType = mlir::cast<ShapedType>(cosCache.getType());
  auto sinCacheType = mlir::cast<ShapedType>(sinCache.getType());
  if (cosCacheType.getRank() != 2 && cosCacheType.getRank() != 3)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), cosCache, cosCacheType.getRank(), "2 or 3");
  if (sinCacheType.getRank() != 2 && sinCacheType.getRank() != 3)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), sinCache, sinCacheType.getRank(), "2 or 3");

  if (!cosCacheType.hasStaticShape() || !sinCacheType.hasStaticShape())
    return success(); // Won't be able to do any more checking at this stage.

  auto cosCacheShape = cosCacheType.getShape();
  auto sinCacheShape = sinCacheType.getShape();

  // Last dim of cos/sin caches must be equal to rotary_embedding_dim / 2
  auto rotaryEmbeddingDim = adaptor.getRotaryEmbeddingDim();
  if (rotaryEmbeddingDim) {
    size_t lastIndex = cosCacheShape.size() - 1;
    if (cosCacheShape[lastIndex] != rotaryEmbeddingDim / 2)
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), cosCache, lastIndex, cosCacheShape[lastIndex],
          std::to_string(rotaryEmbeddingDim / 2));
    lastIndex = sinCacheShape.size() - 1;
    if (sinCacheShape[lastIndex] == rotaryEmbeddingDim / 2)
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), sinCache, lastIndex, sinCacheShape[lastIndex],
          std::to_string(rotaryEmbeddingDim / 2));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRotaryEmbeddingOp::inferShapes(
    std::function<void(Region &)> /*shapeInferenceFunc*/) {
  // Cannot infer the output shape if the data input's shape is not yet known.
  if (!hasShapeAndRank(getOperation()->getOperand(0)))
    return success();

  Type elementType = mlir::cast<ShapedType>(getX().getType()).getElementType();
  ONNXRotaryEmbeddingOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXRotaryEmbeddingOp>;
} // namespace onnx_mlir