// IMPLEMENT YOUR SHAPE INFERENCE HERE
// This file contains templates - safe to edit and customize
// Move to: src/Dialect/ONNX/ONNXOps/Additional/XFEShapeInference.cpp
// and add it to the CMakeLists.txt in src/Dialect/ONNX/

#include "XFEShapeInference.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Helper function for computing channel-last spatial output dimensions
// Reused by Conv and Pooling operations
//===----------------------------------------------------------------------===//

// Compute output spatial dimension for channel-last operations
// Formula: output = floor((input + pad_before + pad_after - ((kernel - 1) *
// dilation + 1)) / stride) + 1
static int64_t computeChannelLastSpatialDim(int64_t inputDim, int64_t kernelDim,
    int64_t padBefore, int64_t padAfter, int64_t stride, int64_t dilation) {
  if (inputDim == ShapedType::kDynamic || kernelDim == ShapedType::kDynamic)
    return ShapedType::kDynamic;

  int64_t effectiveKernel = (kernelDim - 1) * dilation + 1;
  return ((inputDim + padBefore + padAfter - effectiveKernel) / stride) + 1;
}

//===----------------------------------------------------------------------===//
// XFE Shape Inference Implementations
//===----------------------------------------------------------------------===//

LogicalResult XFEMatMulBiasOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  // Cast to specific op type to access operation-specific methods
  auto matmulOp = dyn_cast<XFEMatMulBiasOp>(op);
  if (!matmulOp)
    return failure();

  // Get A and B inputs (bias C doesn't affect output shape)
  Value A = matmulOp.getA();
  Value B = matmulOp.getB();

  // Cannot infer shape if inputs don't have shape and rank
  if (!hasShapeAndRank(A) || !hasShapeAndRank(B))
    return success();

  // Reuse the existing MatMul shape helper by creating a temporary helper
  // The helper computes shape following ONNX MatMul semantics
  Type elementType = mlir::cast<ShapedType>(A.getType()).getElementType();

  // Use ONNXMatMulOpShapeHelper to compute the shape
  // We pass A and B as operands (bias doesn't affect shape)
  ONNXMatMulOpShapeHelper shapeHelper(op, {A, B});
  if (failed(shapeHelper.computeShapeAndUpdateType(elementType)))
    return failure();

  return success();
}

LogicalResult XFEConvOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  // Cast to specific op type
  auto convOp = dyn_cast<XFEConvOp>(op);
  if (!convOp)
    return failure();

  // Get inputs: X (channel-last), W (OHWI), optional B
  Value X = convOp.getX();
  Value W = convOp.getW();

  // Cannot infer shape if inputs don't have shape and rank
  if (!hasShapeAndRank(X) || !hasShapeAndRank(W))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto wType = mlir::cast<ShapedType>(W.getType());
  auto xShape = xType.getShape();
  auto wShape = wType.getShape();

  // X is channel-last (NHWC): [N, spatial_dims..., C_in]
  // W is OHWI: [C_out, spatial_dims..., C_in/group]
  // Require at least 3D tensors and matching ranks
  if (xShape.size() < 3 || wShape.size() < 3 || xShape.size() != wShape.size())
    return op->emitError("ConvChannelLast requires matching rank tensors with "
                         "at least 3 dimensions");

  int64_t rank = xShape.size();
  int64_t numSpatialDims = rank - 2; // exclude batch and channel
  int64_t N = xShape[0];             // batch
  int64_t C_out = wShape[0]; // output channels (first dimension in OHWI)

  // Get attributes
  auto stridesAttr = convOp.getStrides();
  auto padsAttr = convOp.getPads();
  auto dilationsAttr = convOp.getDilations();

  // Default values (all 1s for strides/dilations, 0s for pads)
  SmallVector<int64_t, 4> strides(numSpatialDims, 1);
  SmallVector<int64_t, 8> pads(
      numSpatialDims * 2, 0); // begin and end pads for each dim
  SmallVector<int64_t, 4> dilations(numSpatialDims, 1);

  // Parse strides
  if (stridesAttr.has_value()) {
    auto stridesArray = stridesAttr.value();
    for (size_t i = 0; i < std::min(stridesArray.size(), strides.size()); ++i) {
      strides[i] = mlir::cast<IntegerAttr>(stridesArray[i]).getInt();
    }
  }

  // Parse pads [begin_0, begin_1, ..., end_0, end_1, ...]
  if (padsAttr.has_value()) {
    auto padsArray = padsAttr.value();
    for (size_t i = 0; i < std::min(padsArray.size(), pads.size()); ++i) {
      pads[i] = mlir::cast<IntegerAttr>(padsArray[i]).getInt();
    }
  }

  // Parse dilations
  if (dilationsAttr.has_value()) {
    auto dilationsArray = dilationsAttr.value();
    for (size_t i = 0; i < std::min(dilationsArray.size(), dilations.size());
         ++i) {
      dilations[i] = mlir::cast<IntegerAttr>(dilationsArray[i]).getInt();
    }
  }

  // Compute output spatial dimensions
  SmallVector<int64_t, 6> outputShape;
  outputShape.push_back(N); // batch

  for (int64_t i = 0; i < numSpatialDims; ++i) {
    int64_t inputDim = xShape[i + 1]; // spatial dimension from input (NHWC)
    int64_t kernelDim =
        wShape[i + 1]; // kernel size from weight (OHWI: skip O, then H,W,...)
    int64_t padBegin = pads[i];
    int64_t padEnd = pads[numSpatialDims + i];
    int64_t stride = strides[i];
    int64_t dilation = dilations[i];

    int64_t outputDim = computeChannelLastSpatialDim(
        inputDim, kernelDim, padBegin, padEnd, stride, dilation);
    outputShape.push_back(outputDim);
  }

  outputShape.push_back(C_out); // output channels

  // Set the result type
  // CRITICAL: Preserve existing element type if already set (e.g., quantized
  // types). Only fall back to input element type if result is unranked
  Type elementType = xType.getElementType();
  if (auto existingType = dyn_cast<ShapedType>(convOp.getResult().getType())) {
    elementType = existingType.getElementType();
  }
  auto resultType = RankedTensorType::get(outputShape, elementType);
  convOp.getResult().setType(resultType);

  return success();
}

LogicalResult XFEConvTransposeOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  // Cast to specific op type
  auto convTransposeOp = dyn_cast<XFEConvTransposeOp>(op);
  if (!convTransposeOp)
    return failure();

  // Get inputs: X (channel-last), W (OHWI), optional B
  Value X = convTransposeOp.getX();
  Value W = convTransposeOp.getW();

  // Cannot infer shape if inputs don't have shape and rank
  if (!hasShapeAndRank(X) || !hasShapeAndRank(W))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto wType = mlir::cast<ShapedType>(W.getType());
  auto xShape = xType.getShape();
  auto wShape = wType.getShape();

  // X is channel-last (NHWC): [N, spatial_dims..., C_in]
  // W is OHWI: [C_out, spatial_dims..., C_in/group]
  if (xShape.size() < 3 || wShape.size() < 3 || xShape.size() != wShape.size())
    return op->emitError(
        "ConvTransposeChannelLast requires matching rank tensors with "
        "at least 3 dimensions");

  int64_t rank = xShape.size();
  int64_t numSpatialDims = rank - 2; // exclude batch and channel
  int64_t N = xShape[0];             // batch
  int64_t C_out = wShape[0]; // output channels (first dimension in OHWI)

  // Get attributes
  auto stridesAttr = convTransposeOp.getStrides();
  auto padsAttr = convTransposeOp.getPads();
  auto dilationsAttr = convTransposeOp.getDilations();
  auto outputPaddingAttr = convTransposeOp.getOutputPadding();

  // Default values
  SmallVector<int64_t, 4> strides(numSpatialDims, 1);
  SmallVector<int64_t, 8> pads(numSpatialDims * 2, 0);
  SmallVector<int64_t, 4> dilations(numSpatialDims, 1);
  SmallVector<int64_t, 4> outputPadding(numSpatialDims, 0);

  // Parse attributes
  if (stridesAttr.has_value()) {
    auto stridesArray = stridesAttr.value();
    for (size_t i = 0; i < std::min(stridesArray.size(), strides.size()); ++i) {
      strides[i] = mlir::cast<IntegerAttr>(stridesArray[i]).getInt();
    }
  }

  if (padsAttr.has_value()) {
    auto padsArray = padsAttr.value();
    for (size_t i = 0; i < std::min(padsArray.size(), pads.size()); ++i) {
      pads[i] = mlir::cast<IntegerAttr>(padsArray[i]).getInt();
    }
  }

  if (dilationsAttr.has_value()) {
    auto dilationsArray = dilationsAttr.value();
    for (size_t i = 0; i < std::min(dilationsArray.size(), dilations.size());
         ++i) {
      dilations[i] = mlir::cast<IntegerAttr>(dilationsArray[i]).getInt();
    }
  }

  if (outputPaddingAttr.has_value()) {
    auto outputPaddingArray = outputPaddingAttr.value();
    for (size_t i = 0;
         i < std::min(outputPaddingArray.size(), outputPadding.size()); ++i) {
      outputPadding[i] =
          mlir::cast<IntegerAttr>(outputPaddingArray[i]).getInt();
    }
  }

  // Compute output spatial dimensions for ConvTranspose
  // Formula: output_dim = (input_dim - 1) * stride - 2 * pad + (kernel - 1) *
  // dilation + 1 + output_padding
  SmallVector<int64_t, 6> outputShape;
  outputShape.push_back(N); // batch

  for (int64_t i = 0; i < numSpatialDims; ++i) {
    int64_t inputDim = xShape[i + 1]; // spatial dimension from input (NHWC)
    int64_t kernelDim =
        wShape[i + 1]; // kernel size from weight (OHWI: skip O, then H,W,...)
    int64_t padBegin = pads[i];
    int64_t padEnd = pads[numSpatialDims + i];
    int64_t stride = strides[i];
    int64_t dilation = dilations[i];
    int64_t outPad = outputPadding[i];

    int64_t outputDim = ShapedType::kDynamic;
    if (inputDim != ShapedType::kDynamic && kernelDim != ShapedType::kDynamic) {
      int64_t effectiveKernel = (kernelDim - 1) * dilation + 1;
      outputDim = (inputDim - 1) * stride - padBegin - padEnd +
                  effectiveKernel + outPad;
    }
    outputShape.push_back(outputDim);
  }

  outputShape.push_back(C_out); // output channels

  // Set the result type
  // CRITICAL: Preserve existing element type if already set (e.g., quantized
  // types)
  Type elementType = xType.getElementType();
  if (auto existingType =
          dyn_cast<ShapedType>(convTransposeOp.getResult().getType())) {
    elementType = existingType.getElementType();
  }
  auto resultType = RankedTensorType::get(outputShape, elementType);
  convTransposeOp.getResult().setType(resultType);

  return success();
}

LogicalResult XFEAveragePoolOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto poolOp = dyn_cast<XFEAveragePoolOp>(op);
  if (!poolOp)
    return failure();

  Value X = poolOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto xShape = xType.getShape(); // [N, spatial_dims..., C]

  if (xShape.size() < 3)
    return op->emitError(
        "AveragePoolChannelLast requires at least 3D input tensor");

  int64_t rank = xShape.size();
  int64_t numSpatialDims = rank - 2; // exclude batch and channel
  int64_t N = xShape[0];             // batch
  int64_t C = xShape[rank - 1];      // channels

  // Get attributes
  auto kernelShapeAttr = poolOp.getKernelShape();
  auto stridesAttr = poolOp.getStrides();
  auto padsAttr = poolOp.getPads();

  // Extract kernel shape (required attribute)
  if (!kernelShapeAttr.has_value() ||
      static_cast<int64_t>(kernelShapeAttr->size()) < numSpatialDims)
    return op->emitError(
        "kernel_shape attribute required with matching spatial dimensions");

  SmallVector<int64_t, 4> kernels;
  for (int64_t i = 0; i < numSpatialDims; ++i) {
    kernels.push_back(mlir::cast<IntegerAttr>((*kernelShapeAttr)[i]).getInt());
  }

  // Parse strides (default 1)
  SmallVector<int64_t, 4> strides(numSpatialDims, 1);
  if (stridesAttr.has_value()) {
    for (size_t i = 0; i < std::min(stridesAttr->size(), strides.size()); ++i) {
      strides[i] = mlir::cast<IntegerAttr>((*stridesAttr)[i]).getInt();
    }
  }

  // Parse pads [begin_0, begin_1, ..., end_0, end_1, ...] (default 0)
  SmallVector<int64_t, 8> pads(numSpatialDims * 2, 0);
  if (padsAttr.has_value()) {
    for (size_t i = 0; i < std::min(padsAttr->size(), pads.size()); ++i) {
      pads[i] = mlir::cast<IntegerAttr>((*padsAttr)[i]).getInt();
    }
  }

  // Compute output spatial dimensions (no dilation for average pool)
  SmallVector<int64_t, 6> outputShape;
  outputShape.push_back(N); // batch

  for (int64_t i = 0; i < numSpatialDims; ++i) {
    int64_t inputDim = xShape[i + 1];
    int64_t kernelDim = kernels[i];
    int64_t padBegin = pads[i];
    int64_t padEnd = pads[numSpatialDims + i];
    int64_t stride = strides[i];

    int64_t outputDim = computeChannelLastSpatialDim(
        inputDim, kernelDim, padBegin, padEnd, stride, 1);
    outputShape.push_back(outputDim);
  }

  outputShape.push_back(C); // channels

  // CRITICAL: Preserve existing element type if already set (e.g., quantized
  // types)
  Type elementType = xType.getElementType();
  if (auto existingType = dyn_cast<ShapedType>(poolOp.getResult().getType())) {
    elementType = existingType.getElementType();
  }
  auto resultType = RankedTensorType::get(outputShape, elementType);
  poolOp.getResult().setType(resultType);

  return success();
}

LogicalResult XFEMaxPoolOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto poolOp = dyn_cast<XFEMaxPoolOp>(op);
  if (!poolOp)
    return failure();

  Value X = poolOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto xShape = xType.getShape(); // [N, spatial_dims..., C]

  if (xShape.size() < 3)
    return op->emitError(
        "MaxPoolChannelLast requires at least 3D input tensor");

  int64_t rank = xShape.size();
  int64_t numSpatialDims = rank - 2; // exclude batch and channel
  int64_t N = xShape[0];             // batch
  int64_t C = xShape[rank - 1];      // channels

  // Get attributes
  auto kernelShapeAttr = poolOp.getKernelShape();
  auto stridesAttr = poolOp.getStrides();
  auto padsAttr = poolOp.getPads();
  auto dilationsAttr = poolOp.getDilations();

  // Extract kernel shape (required attribute)
  if (!kernelShapeAttr.has_value() ||
      static_cast<int64_t>(kernelShapeAttr->size()) < numSpatialDims)
    return op->emitError(
        "kernel_shape attribute required with matching spatial dimensions");

  SmallVector<int64_t, 4> kernels;
  for (int64_t i = 0; i < numSpatialDims; ++i) {
    kernels.push_back(mlir::cast<IntegerAttr>((*kernelShapeAttr)[i]).getInt());
  }

  // Parse strides (default 1)
  SmallVector<int64_t, 4> strides(numSpatialDims, 1);
  if (stridesAttr.has_value()) {
    for (size_t i = 0; i < std::min(stridesAttr->size(), strides.size()); ++i) {
      strides[i] = mlir::cast<IntegerAttr>((*stridesAttr)[i]).getInt();
    }
  }

  // Parse pads [begin_0, begin_1, ..., end_0, end_1, ...] (default 0)
  SmallVector<int64_t, 8> pads(numSpatialDims * 2, 0);
  if (padsAttr.has_value()) {
    for (size_t i = 0; i < std::min(padsAttr->size(), pads.size()); ++i) {
      pads[i] = mlir::cast<IntegerAttr>((*padsAttr)[i]).getInt();
    }
  }

  // Parse dilations (default 1) - MaxPool supports dilations
  SmallVector<int64_t, 4> dilations(numSpatialDims, 1);
  if (dilationsAttr.has_value()) {
    for (size_t i = 0; i < std::min(dilationsAttr->size(), dilations.size());
         ++i) {
      dilations[i] = mlir::cast<IntegerAttr>((*dilationsAttr)[i]).getInt();
    }
  }

  // Compute output spatial dimensions with dilations
  SmallVector<int64_t, 6> outputShape;
  outputShape.push_back(N); // batch

  for (int64_t i = 0; i < numSpatialDims; ++i) {
    int64_t inputDim = xShape[i + 1];
    int64_t kernelDim = kernels[i];
    int64_t padBegin = pads[i];
    int64_t padEnd = pads[numSpatialDims + i];
    int64_t stride = strides[i];
    int64_t dilation = dilations[i];

    int64_t outputDim = computeChannelLastSpatialDim(
        inputDim, kernelDim, padBegin, padEnd, stride, dilation);
    outputShape.push_back(outputDim);
  }

  outputShape.push_back(C); // channels

  // CRITICAL: Preserve existing element type if already set (e.g., quantized
  // types)
  Type elementType = xType.getElementType();
  if (auto existingType = dyn_cast<ShapedType>(poolOp.getResult().getType())) {
    elementType = existingType.getElementType();
  }
  auto resultType = RankedTensorType::get(outputShape, elementType);
  poolOp.getResult().setType(resultType);

  return success();
}

LogicalResult XFEGlobalAveragePoolOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto poolOp = dyn_cast<XFEGlobalAveragePoolOp>(op);
  if (!poolOp)
    return failure();

  Value X = poolOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto xShape = xType.getShape(); // [N, spatial_dims..., C]

  if (xShape.size() < 3)
    return op->emitError(
        "GlobalAveragePoolChannelLast requires at least 3D input tensor");

  int64_t rank = xShape.size();
  int64_t numSpatialDims = rank - 2; // exclude batch and channel
  int64_t N = xShape[0];             // batch
  int64_t C = xShape[rank - 1];      // channels

  // Global pooling reduces all spatial dimensions to 1
  // Output shape: [N, 1, 1, ..., 1, C]
  SmallVector<int64_t, 6> outputShape;
  outputShape.push_back(N); // batch
  for (int64_t i = 0; i < numSpatialDims; ++i) {
    outputShape.push_back(1); // each spatial dimension reduced to 1
  }
  outputShape.push_back(C); // channels

  // CRITICAL: Preserve existing element type if already set (e.g., quantized
  // types)
  Type elementType = xType.getElementType();
  if (auto existingType = dyn_cast<ShapedType>(poolOp.getResult().getType())) {
    elementType = existingType.getElementType();
  }
  auto resultType = RankedTensorType::get(outputShape, elementType);
  poolOp.getResult().setType(resultType);

  return success();
}

LogicalResult XFEGlobalMaxPoolOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto poolOp = dyn_cast<XFEGlobalMaxPoolOp>(op);
  if (!poolOp)
    return failure();

  Value X = poolOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto xShape = xType.getShape(); // [N, spatial_dims..., C]

  if (xShape.size() < 3)
    return op->emitError(
        "GlobalMaxPoolChannelLast requires at least 3D input tensor");

  int64_t rank = xShape.size();
  int64_t numSpatialDims = rank - 2; // exclude batch and channel
  int64_t N = xShape[0];             // batch
  int64_t C = xShape[rank - 1];      // channels

  // Global pooling reduces all spatial dimensions to 1
  // Output shape: [N, 1, 1, ..., 1, C]
  SmallVector<int64_t, 6> outputShape;
  outputShape.push_back(N); // batch
  for (int64_t i = 0; i < numSpatialDims; ++i) {
    outputShape.push_back(1); // each spatial dimension reduced to 1
  }
  outputShape.push_back(C); // channels

  // CRITICAL: Preserve existing element type if already set (e.g., quantized
  // types)
  Type elementType = xType.getElementType();
  if (auto existingType = dyn_cast<ShapedType>(poolOp.getResult().getType())) {
    elementType = existingType.getElementType();
  }
  auto resultType = RankedTensorType::get(outputShape, elementType);
  poolOp.getResult().setType(resultType);

  return success();
}

LogicalResult XFEInstanceNormalizationOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto normOp = dyn_cast<XFEInstanceNormalizationOp>(op);
  if (!normOp)
    return failure();

  Value input = normOp.getInput();
  if (!hasShapeAndRank(input))
    return success();

  auto inputType = mlir::cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape(); // [N, spatial_dims..., C]

  if (inputShape.size() < 3)
    return op->emitError(
        "InstanceNormalizationChannelLast requires at least 3D input tensor");

  // Instance normalization preserves the input shape
  // Output shape: same as input
  SmallVector<int64_t, 6> outputShape(inputShape.begin(), inputShape.end());

  Type elementType = inputType.getElementType();
  auto resultType = RankedTensorType::get(outputShape, elementType);
  normOp.getResult().setType(resultType);

  return success();
}

LogicalResult XFEDepthToSpaceOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto d2sOp = dyn_cast<XFEDepthToSpaceOp>(op);
  if (!d2sOp)
    return failure();

  Value input = d2sOp.getInput();
  if (!hasShapeAndRank(input))
    return success();

  auto inputType = mlir::cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape(); // [N, H, W, C]

  if (inputShape.size() != 4)
    return op->emitError("DepthToSpaceChannelLast requires 4D input tensor");

  // Get blocksize attribute
  auto blocksizeAttr = d2sOp.getBlocksize();
  if (!blocksizeAttr.has_value())
    return op->emitError("blocksize attribute is required");

  int64_t blocksize = blocksizeAttr.value();
  if (blocksize <= 0)
    return op->emitError("blocksize must be positive");

  int64_t N = inputShape[0];
  int64_t H = inputShape[1];
  int64_t W = inputShape[2];
  int64_t C = inputShape[3];

  // Check if channels is divisible by blocksize^2
  int64_t blocksize_sq = blocksize * blocksize;
  if (C != ShapedType::kDynamic && C % blocksize_sq != 0)
    return op->emitError("input channels must be divisible by blocksize^2");

  // DepthToSpace: increases spatial dimensions, decreases channel depth
  // Output: [N, H*blocksize, W*blocksize, C/(blocksize^2)]
  int64_t H_out =
      (H == ShapedType::kDynamic) ? ShapedType::kDynamic : H * blocksize;
  int64_t W_out =
      (W == ShapedType::kDynamic) ? ShapedType::kDynamic : W * blocksize;
  int64_t C_out =
      (C == ShapedType::kDynamic) ? ShapedType::kDynamic : C / blocksize_sq;

  SmallVector<int64_t, 4> outputShape = {N, H_out, W_out, C_out};

  Type elementType = inputType.getElementType();
  auto resultType = RankedTensorType::get(outputShape, elementType);
  d2sOp.getResult().setType(resultType);

  return success();
}

LogicalResult XFESpaceToDepthOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto s2dOp = dyn_cast<XFESpaceToDepthOp>(op);
  if (!s2dOp)
    return failure();

  Value input = s2dOp.getInput();
  if (!hasShapeAndRank(input))
    return success();

  auto inputType = mlir::cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape(); // [N, H, W, C]

  if (inputShape.size() != 4)
    return op->emitError("SpaceToDepthChannelLast requires 4D input tensor");

  // Get blocksize attribute
  auto blocksizeAttr = s2dOp.getBlocksize();
  if (!blocksizeAttr.has_value())
    return op->emitError("blocksize attribute is required");

  int64_t blocksize = blocksizeAttr.value();
  if (blocksize <= 0)
    return op->emitError("blocksize must be positive");

  int64_t N = inputShape[0];
  int64_t H = inputShape[1];
  int64_t W = inputShape[2];
  int64_t C = inputShape[3];

  // Check if spatial dimensions are divisible by blocksize
  if (H != ShapedType::kDynamic && H % blocksize != 0)
    return op->emitError("input height must be divisible by blocksize");
  if (W != ShapedType::kDynamic && W % blocksize != 0)
    return op->emitError("input width must be divisible by blocksize");

  // SpaceToDepth: decreases spatial dimensions, increases channel depth
  // Output: [N, H/blocksize, W/blocksize, C*(blocksize^2)]
  int64_t blocksize_sq = blocksize * blocksize;
  int64_t H_out =
      (H == ShapedType::kDynamic) ? ShapedType::kDynamic : H / blocksize;
  int64_t W_out =
      (W == ShapedType::kDynamic) ? ShapedType::kDynamic : W / blocksize;
  int64_t C_out =
      (C == ShapedType::kDynamic) ? ShapedType::kDynamic : C * blocksize_sq;

  SmallVector<int64_t, 4> outputShape = {N, H_out, W_out, C_out};

  Type elementType = inputType.getElementType();
  auto resultType = RankedTensorType::get(outputShape, elementType);
  s2dOp.getResult().setType(resultType);

  return success();
}

LogicalResult XFEResizeOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto resizeOp = dyn_cast<XFEResizeOp>(op);
  if (!resizeOp)
    return failure();

  Value X = resizeOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto xShape = xType.getShape();

  // Helper to check if input is absent (None or empty tensor)
  auto isAbsent = [](Value input) -> bool {
    if (isa<NoneType>(input.getType()))
      return true;
    if (auto shapedType = mlir::dyn_cast<ShapedType>(input.getType())) {
      return shapedType.hasStaticShape() && shapedType.getNumElements() == 0;
    }
    return false;
  };

  Value scales = resizeOp.getScales();
  Value sizes = resizeOp.getSizes();

  bool scalesIsAbsent = isAbsent(scales);
  bool sizesIsAbsent = isAbsent(sizes);

  // Axes attribute is not yet supported for channel-last resize
  if (resizeOp.getAxes().has_value())
    return success(); // Return success but don't infer shape

  SmallVector<int64_t, 6> outputShape;
  int64_t rank = xShape.size();

  if (!scalesIsAbsent) {
    // Output shape determined by scales
    // Try to get constant scales
    DenseElementsAttr scalesAttr;
    if (auto defOp = scales.getDefiningOp<ONNXConstantOp>()) {
      if (auto valueAttr = defOp.getValue()) {
        scalesAttr = mlir::dyn_cast<DenseElementsAttr>(*valueAttr);
      }
    }

    if (scalesAttr) {
      auto scalesValues = scalesAttr.getValues<float>();
      if (static_cast<int64_t>(scalesValues.size()) != rank)
        return op->emitError("scales size must match input rank");

      for (int64_t i = 0; i < rank; ++i) {
        int64_t inputDim = xShape[i];
        float scale = scalesValues[i];
        if (inputDim == ShapedType::kDynamic) {
          outputShape.push_back(ShapedType::kDynamic);
        } else {
          outputShape.push_back(
              static_cast<int64_t>(std::floor(inputDim * scale)));
        }
      }
    } else {
      // Scales are not constant, output shape is dynamic
      for (int64_t i = 0; i < rank; ++i) {
        outputShape.push_back(ShapedType::kDynamic);
      }
    }
  } else if (!sizesIsAbsent) {
    // Output shape determined by sizes
    DenseElementsAttr sizesAttr;
    if (auto defOp = sizes.getDefiningOp<ONNXConstantOp>()) {
      if (auto valueAttr = defOp.getValue()) {
        sizesAttr = mlir::dyn_cast<DenseElementsAttr>(*valueAttr);
      }
    }

    if (sizesAttr) {
      auto sizesValues = sizesAttr.getValues<int64_t>();
      if (static_cast<int64_t>(sizesValues.size()) != rank)
        return op->emitError("sizes size must match input rank");

      for (int64_t i = 0; i < rank; ++i) {
        outputShape.push_back(sizesValues[i]);
      }
    } else {
      // Sizes are not constant, output shape is dynamic
      for (int64_t i = 0; i < rank; ++i) {
        outputShape.push_back(ShapedType::kDynamic);
      }
    }
  } else {
    // Both scales and sizes are absent - cannot infer shape
    return success();
  }

  // Set the result type
  // CRITICAL: Preserve existing element type if already set (e.g., quantized
  // types)
  Type elementType = xType.getElementType();
  if (auto existingType = dyn_cast<ShapedType>(resizeOp.getResult().getType())) {
    elementType = existingType.getElementType();
  }
  auto resultType = RankedTensorType::get(outputShape, elementType);
  resizeOp.getResult().setType(resultType);

  return success();
}

} // namespace mlir
