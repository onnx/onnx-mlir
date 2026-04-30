/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Im2Col.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Im2Col operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

LogicalResult ONNXIm2ColOpShapeHelper::computeShape() {
  // Get the operation and adaptor.
  ONNXIm2ColOpAdaptor operandAdaptor(operands);
  ONNXIm2ColOp im2colOp = llvm::cast<ONNXIm2ColOp>(op);

  // Get input tensor X.
  Value X = operandAdaptor.getX();
  assert(isRankedShapedType(X.getType()) && "Expected ranked input");

  // Get kernel shape (required attribute, validated in verifier).
  ArrayAttr kernelShapeAttr = im2colOp.getKernelShapeAttr();
  spatialRank = kernelShapeAttr.size();

  // Extract kernel shape values (validation done in verifier).
  kernelShape.clear();
  for (auto attr : kernelShapeAttr)
    kernelShape.push_back(mlir::cast<IntegerAttr>(attr).getInt());

  // Process dilations (default: all 1s, validation done in verifier).
  dilations.clear();
  if (im2colOp.getDilations().has_value()) {
    ArrayAttr dilationsAttr = im2colOp.getDilationsAttr();
    for (auto attr : dilationsAttr)
      dilations.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  } else {
    dilations.assign(spatialRank, 1);
  }

  // Process strides (default: all 1s, validation done in verifier).
  strides.clear();
  if (im2colOp.getStrides().has_value()) {
    ArrayAttr stridesAttr = im2colOp.getStridesAttr();
    for (auto attr : stridesAttr)
      strides.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  } else {
    strides.assign(spatialRank, 1);
  }

  // Process pads.
  pads.clear();
  StringRef autoPad = im2colOp.getAutoPad();
  IndexExpr zeroIE = LitIE(0);

  if (autoPad == "NOTSET") {
    // Use explicit pads or default to 0 (validation done in verifier).
    if (im2colOp.getPads().has_value()) {
      ArrayAttr padsAttr = im2colOp.getPadsAttr();
      // Store all pads: [begin_0, ..., begin_N, end_0, ..., end_N].
      for (int64_t i = 0; i < 2 * spatialRank; ++i) {
        int64_t val = mlir::cast<IntegerAttr>(padsAttr[i]).getInt();
        pads.push_back(LitIE(val));
      }
    } else {
      // Default: no padding.
      pads.assign(2 * spatialRank, zeroIE);
    }
  } else if (autoPad == "VALID") {
    // For VALID, pads are all 0.
    pads.assign(2 * spatialRank, zeroIE);
  } else {
    // SAME_UPPER or SAME_LOWER: compute pads to make output size = ceil(input
    // / stride). Formula: total_pad = max(0, (output_size - 1) * stride +
    // kernel_size - input_size) where output_size = ceil(input_size /
    // stride).
    llvm::SmallVector<IndexExpr, 4> padEnds;
    for (int64_t i = 0; i < spatialRank; ++i) {
      IndexExpr inputDim = createIE->getShapeAsDim(X, 2 + i);
      int64_t stride = strides[i];
      int64_t kernel = kernelShape[i];
      int64_t dilation = dilations[i];
      int64_t effectiveKernel = (kernel - 1) * dilation + 1;

      // output_size = ceil(input_size / stride).
      IndexExpr outputDim = inputDim.ceilDiv(stride);
      // total_pad = max(0, (output_size - 1) * stride + effectiveKernel -
      // input_size).
      IndexExpr totalPad = IndexExpr::max(
          zeroIE, (outputDim - 1) * stride + effectiveKernel - inputDim);

      // For SAME_UPPER: pad_begin = floor(total_pad / 2).
      // For SAME_LOWER: pad_begin = total_pad - floor(total_pad / 2).
      IndexExpr padBegin, padEnd;
      if (autoPad == "SAME_UPPER") {
        padBegin = totalPad.floorDiv(2);
        padEnd = totalPad - padBegin;
      } else { // SAME_LOWER
        padEnd = totalPad.floorDiv(2);
        padBegin = totalPad - padEnd;
      }
      pads.push_back(padBegin);
      padEnds.push_back(padEnd);
    }
    // Append end pads.
    pads.insert(pads.end(), padEnds.begin(), padEnds.end());
  }

  // Compute output spatial dimensions.
  // Formula: output_dim = floor((input_dim + pad_being + pad_end -
  // dilation*(kernel-1) - 1) / stride) + 1.
  outputSpatialDims.clear();
  for (int64_t i = 0; i < spatialRank; ++i) {
    IndexExpr inputDim = createIE->getShapeAsDim(X, 2 + i);
    int64_t stride = strides[i];
    int64_t kernel = kernelShape[i];
    int64_t dilation = dilations[i];

    // Effective kernel size with dilation.
    int64_t effectiveKernel = (kernel - 1) * dilation + 1;

    // output_dim = floor((input_dim + padBegin + padEnd - effectiveKernel) /
    // stride) + 1.
    IndexExpr padBegin = pads[i];
    IndexExpr padEnd = pads[spatialRank + i];
    IndexExpr numerator = inputDim + padBegin + padEnd - effectiveKernel;
    IndexExpr outputDim = numerator.floorDiv(stride) + 1;
    outputSpatialDims.push_back(outputDim);
  }

  // Compute output shape: [N, CI * prod(kernel_shape),
  // prod(output_spatial_dims)].
  IndexExpr N = createIE->getShapeAsDim(X, 0);
  IndexExpr CI = createIE->getShapeAsDim(X, 1);

  // Compute numRows = CI * prod(kernel_shape).
  IndexExpr numRows = CI;
  for (int64_t k : kernelShape) {
    numRows = numRows * k;
  }

  // Compute numCols = prod(outputSpatialDims).
  IndexExpr numCols = LitIE(1);
  for (const IndexExpr &dim : outputSpatialDims) {
    numCols = numCols * dim;
  }

  // Set output dimensions: [N, CI*KH*KW, OH*OW].
  DimsExpr outputDims = {N, numRows, numCols};
  setOutputDims(outputDims);

  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXIm2ColOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Get input type.
  if (!hasShapeAndRank(getX()))
    return success();

  // Create shape helper and compute shape.
  ONNXIm2ColOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      mlir::cast<ShapedType>(getX().getType()).getElementType());
}

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

LogicalResult ONNXIm2ColOp::verify() {
  ONNXIm2ColOpAdaptor operandAdaptor(*this);

  // Verify kernel_shape is provided (it's a required attribute in TableGen).
  ArrayAttr kernelShapeAttr = getKernelShapeAttr();
  if (!kernelShapeAttr || kernelShapeAttr.empty())
    return emitError(
        "kernel_shape attribute is required and must not be empty");

  int64_t spatialRank = kernelShapeAttr.size();
  if (spatialRank <= 0)
    return emitError("spatial rank is required to be greater than 0");

  // Verify kernel_shape values are positive.
  for (int64_t i = 0; i < spatialRank; ++i) {
    int64_t val = mlir::cast<IntegerAttr>(kernelShapeAttr[i]).getInt();
    if (val <= 0)
      return emitError("Kernel shape values must be positive");
  }

  // Verify input rank if input has shape.
  Value X = operandAdaptor.getX();
  if (isRankedShapedType(X.getType())) {
    ArrayRef<int64_t> xShape = getShape(X.getType());
    int64_t xRank = xShape.size();

    // Input must be at least rank 3 (N, CI, spatial_dims...).
    if (xRank < 3)
      return emitError("Im2Col input must have rank >= 3");

    // Verify spatial rank matches input.
    if (xRank != spatialRank + 2)
      return emitError("Input rank must equal kernel_shape size + 2");
  }

  // Verify dilations if provided.
  if (getDilations().has_value()) {
    ArrayAttr dilationsAttr = getDilationsAttr();
    if ((int64_t)dilationsAttr.size() != spatialRank)
      return emitError("Dilations size must match spatial rank");
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t val = mlir::cast<IntegerAttr>(dilationsAttr[i]).getInt();
      if (val < 1)
        return emitError("Dilation values must be >= 1");
    }
  }

  // Verify strides if provided.
  if (getStrides().has_value()) {
    ArrayAttr stridesAttr = getStridesAttr();
    if ((int64_t)stridesAttr.size() != spatialRank)
      return emitError("Strides size must match spatial rank");
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t val = mlir::cast<IntegerAttr>(stridesAttr[i]).getInt();
      if (val < 1)
        return emitError("Stride values must be >= 1");
    }
  }

  // Verify pads if provided and auto_pad is NOTSET.
  StringRef autoPad = getAutoPad();
  if (autoPad != "NOTSET" && autoPad != "SAME_UPPER" &&
      autoPad != "SAME_LOWER" && autoPad != "VALID")
    return emitError(
        "auto_pad must be NOTSET, SAME_UPPER, SAME_LOWER, or VALID");

  if (autoPad == "NOTSET" && getPads().has_value()) {
    ArrayAttr padsAttr = getPadsAttr();
    if ((int64_t)padsAttr.size() != 2 * spatialRank)
      return emitError("Pads size must be 2 * spatial rank");
    for (int64_t i = 0; i < (int64_t)padsAttr.size(); ++i) {
      int64_t val = mlir::cast<IntegerAttr>(padsAttr[i]).getInt();
      if (val < 0)
        return emitError("Pad values must be non-negative");
    }
  }

  return success();
}
