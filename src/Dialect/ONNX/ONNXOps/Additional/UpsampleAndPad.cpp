/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- UpsampleAndPad.cpp - UpsampleAndPad Op -----------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file provides canonicalization patterns for the ONNX UpsampleAndPad
// operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXUpsampleAndPadOpShapeHelper::computeShape() {
  // Get operation and adaptor.
  auto upsampleAndPadOp = mlir::dyn_cast<ONNXUpsampleAndPadOp>(op);
  ONNXUpsampleAndPadOpAdaptor operandAdaptor(operands, upsampleAndPadOp);

  // Get input tensor.
  Value X = operandAdaptor.getX();
  if (!hasShapeAndRank(X))
    return failure();

  uint64_t rank = createIE->getShapedTypeRank(X);

  // Get strides and pads attributes.
  std::optional<ArrayAttr> stridesAttrOpt = operandAdaptor.getStrides();
  std::optional<ArrayAttr> padsAttrOpt = operandAdaptor.getPads();

  // Determine k (number of dimensions to process).
  uint64_t k = 0;
  SmallVector<int64_t, 4> stridesVec;
  SmallVector<int64_t, 8> padsVec;

  if (stridesAttrOpt.has_value()) {
    ArrayAttr stridesAttr = stridesAttrOpt.value();
    k = stridesAttr.size();
    for (auto attr : stridesAttr)
      stridesVec.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }

  if (padsAttrOpt.has_value()) {
    ArrayAttr padsAttr = padsAttrOpt.value();
    if (k == 0) {
      // If strides not specified, infer k from pads.
      k = padsAttr.size() / 2;
    }
    for (auto attr : padsAttr)
      padsVec.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }

  // If both are not specified, process all dimensions (k = rank).
  if (k == 0)
    k = rank;

  // Fill in default values if not specified.
  if (stridesVec.empty())
    stridesVec.assign(k, 1);
  if (padsVec.empty())
    padsVec.assign(2 * k, 0);

  // Get input dimensions as IndexExpr.
  DimsExpr inputDims;
  createIE->getShapeAsDims(X, inputDims);

  // Compute output dimensions using IndexExpr.
  DimsExpr outputDims;
  outputDims.resize(rank);

  // First (rank - k) dimensions remain unchanged.
  for (uint64_t i = 0; i < rank - k; ++i) {
    outputDims[i] = inputDims[i];
  }

  // Last k dimensions: apply upsampling and padding.
  for (uint64_t i = 0; i < k; ++i) {
    uint64_t dimIdx = rank - k + i;

    // Upsample: upsampled_size = (input_size - 1) * stride + 1.
    IndexExpr inputDim = inputDims[dimIdx];
    LiteralIndexExpr stride(stridesVec[i]);
    LiteralIndexExpr one(1);

    IndexExpr upsampledDim = (inputDim - one) * stride + one;

    // Add padding: output_size = upsampled_size + pad_begin + pad_end.
    LiteralIndexExpr bothPad(padsVec[i] + padsVec[k + i]);

    outputDims[dimIdx] = upsampledDim + bothPad;
  }

  // Save output dimensions.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXUpsampleAndPadOp::verify() {
  ONNXUpsampleAndPadOpAdaptor operandAdaptor(*this);

  // Check input.
  Value X = operandAdaptor.getX();
  if (!hasShapeAndRank(X)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  auto inputType = mlir::cast<ShapedType>(X.getType());
  int64_t rank = inputType.getRank();

  // Get strides and pads.
  std::optional<ArrayAttr> stridesAttrOpt = operandAdaptor.getStrides();
  std::optional<ArrayAttr> padsAttrOpt = operandAdaptor.getPads();

  // Determine k.
  int64_t k = 0;
  if (stridesAttrOpt.has_value()) {
    k = stridesAttrOpt.value().size();
  } else if (padsAttrOpt.has_value()) {
    k = padsAttrOpt.value().size() / 2;
  } else {
    k = rank; // Default: process all dimensions.
  }

  // Check k <= rank.
  if (k > rank)
    return emitOpError("strides dimensions (")
           << k << ") cannot exceed input rank (" << rank << ")";

  // Check strides values if specified.
  if (stridesAttrOpt.has_value()) {
    ArrayAttr stridesAttr = stridesAttrOpt.value();
    for (int64_t i = 0; i < k; ++i) {
      int64_t stride = mlir::cast<IntegerAttr>(stridesAttr[i]).getInt();
      if (stride <= 0)
        return emitOpError("stride[")
               << i << "] = " << stride << " must be positive";
    }
  }

  // Check pads size if specified.
  if (padsAttrOpt.has_value()) {
    ArrayAttr padsAttr = padsAttrOpt.value();
    if (padsAttr.size() != static_cast<size_t>(2 * k))
      return emitOpError("pads size (")
             << padsAttr.size() << ") must be exactly 2 * strides size ("
             << 2 * k << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXUpsampleAndPadOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no input shape exists.
  if (!hasShapeAndRank(getX()))
    return success();

  ONNXUpsampleAndPadOpShapeHelper shapeHelper(getOperation(), {});
  // ElementType should be from the output.
  ShapedType resultType =
      mlir::cast<ShapedType>(getOperation()->getResult(0).getType());
  Type elementType = resultType.getElementType();
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXUpsampleAndPadOp>;
} // namespace onnx_mlir
