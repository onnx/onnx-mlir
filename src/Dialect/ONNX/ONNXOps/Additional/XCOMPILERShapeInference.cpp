
#include "XCOMPILERShapeInference.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace mlir {

LogicalResult XCOMPILERFusedEltwiseOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto eltwiseOp = dyn_cast<XCOMPILERFusedEltwiseOp>(op);
  if (!eltwiseOp)
    return failure();

  // Get input A (required) and B (optional)
  Value A = eltwiseOp.getA();
  Value B = eltwiseOp.getB();

  // Cannot infer shape if A doesn't have shape and rank
  if (!hasShapeAndRank(A))
    return success();

  auto aType = mlir::cast<ShapedType>(A.getType());
  ArrayRef<int64_t> aShape = aType.getShape();

  // In a quantization domain (any operand or the result has a quant element
  // type), preserve the result's element type so shape inference never
  // changes types; otherwise default to operand A's element type.
  Type elementType = aType.getElementType();
  if (isInQuantizedDomain(op, eltwiseOp.getResult()))
    elementType = mlir::cast<ShapedType>(eltwiseOp.getResult().getType())
                      .getElementType();

  SmallVector<int64_t> outputShape;

  // Check if B is provided (not None)
  if (B && !mlir::isa<NoneType>(B.getType())) {
    // B is provided - compute broadcasted shape
    if (!hasShapeAndRank(B))
      return success();

    auto bType = mlir::cast<ShapedType>(B.getType());
    ArrayRef<int64_t> bShape = bType.getShape();

    // Compute output shape using NumPy-style broadcasting
    if (!OpTrait::util::getBroadcastedShape(aShape, bShape, outputShape)) {
      return op->emitError("FusedEltwise: incompatible shapes for broadcasting")
             << " A shape: [" << aShape << "], B shape: [" << bShape << "]";
    }
  } else {
    // B is None - output shape is same as A (unary operation)
    outputShape.assign(aShape.begin(), aShape.end());
  }

  auto resultType = RankedTensorType::get(outputShape, elementType);
  eltwiseOp.getResult().setType(resultType);

  return success();
}

LogicalResult XCOMPILERDepthwiseConvOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto convOp = dyn_cast<XCOMPILERDepthwiseConvOp>(op);
  if (!convOp)
    return failure();

  // Get input X: [N, H, W, C] for 2D or [N, D, H, W, C] for 3D (NHWC layout)
  Value X = convOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  ArrayRef<int64_t> xShape = xType.getShape();

  // In a quantization domain (any operand or the result has a quant element
  // type), preserve the result's element type so shape inference never
  // changes types; otherwise default to X's element type.
  Type elementType = xType.getElementType();
  if (isInQuantizedDomain(op, convOp.getResult()))
    elementType =
        mlir::cast<ShapedType>(convOp.getResult().getType()).getElementType();

  // Input must be 4D [N, H, W, C] or 5D [N, D, H, W, C]
  size_t rank = xShape.size();
  if (rank != 4 && rank != 5)
    return op->emitError(
        "DepthwiseConv: input must be 4D [N, H, W, C] or 5D [N, D, H, W, C]");

  bool is3D = (rank == 5);
  size_t numSpatialDims = is3D ? 3 : 2;

  // Extract dimensions based on NHWC layout
  int64_t N = xShape[0];
  int64_t C = xShape[rank - 1]; // Channel is last in NHWC
  SmallVector<int64_t> spatialDims;
  for (size_t i = 1; i < rank - 1; ++i)
    spatialDims.push_back(xShape[i]);

  // Get kernel shape (required attribute)
  auto kernelShapeAttr = op->getAttrOfType<ArrayAttr>("kernel_shape");
  if (!kernelShapeAttr || kernelShapeAttr.size() != numSpatialDims)
    return op->emitError("DepthwiseConv: kernel_shape must have ")
           << numSpatialDims << " elements for " << (is3D ? "3D" : "2D")
           << " convolution";

  SmallVector<int64_t> kernelShape;
  for (auto attr : kernelShapeAttr)
    kernelShape.push_back(mlir::cast<IntegerAttr>(attr).getInt());

  // Get strides (default all 1s)
  SmallVector<int64_t> strides(numSpatialDims, 1);
  if (auto stridesAttr = op->getAttrOfType<ArrayAttr>("strides")) {
    strides.clear();
    for (auto attr : stridesAttr)
      strides.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }

  // Get dilations (default all 1s)
  SmallVector<int64_t> dilations(numSpatialDims, 1);
  if (auto dilationsAttr = op->getAttrOfType<ArrayAttr>("dilations")) {
    dilations.clear();
    for (auto attr : dilationsAttr)
      dilations.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }

  // Get pads (default all 0s) - format: [begin0, begin1, ..., end0, end1, ...]
  SmallVector<int64_t> pads(numSpatialDims * 2, 0);
  if (auto padsAttr = op->getAttrOfType<ArrayAttr>("pads")) {
    pads.clear();
    for (auto attr : padsAttr)
      pads.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }

  // Get auto_pad attribute
  auto autoPad = convOp.getAutoPad();

  // Compute output spatial dimensions
  SmallVector<int64_t> outputSpatialDims;
  for (size_t i = 0; i < numSpatialDims; ++i) {
    int64_t inputDim = spatialDims[i];
    int64_t k = kernelShape[i];
    int64_t s = strides[i];
    int64_t d = dilations[i];

    // Effective kernel size with dilation
    int64_t effectiveK = (k - 1) * d + 1;

    int64_t outputDim;
    if (inputDim == ShapedType::kDynamic) {
      outputDim = ShapedType::kDynamic;
    } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
      // SAME padding: output_size = ceil(input_size / stride)
      outputDim = (inputDim + s - 1) / s;
    } else if (autoPad == "VALID") {
      // VALID: no padding
      outputDim = (inputDim - effectiveK) / s + 1;
    } else {
      // NOTSET or default: use explicit pads
      int64_t padBegin = pads[i];
      int64_t padEnd = pads[i + numSpatialDims];
      outputDim = (inputDim + padBegin + padEnd - effectiveK) / s + 1;
    }
    outputSpatialDims.push_back(outputDim);
  }

  // Build output shape in NHWC layout: [N, spatial..., C]
  SmallVector<int64_t> outputShape;
  outputShape.push_back(N);
  for (int64_t dim : outputSpatialDims)
    outputShape.push_back(dim);
  outputShape.push_back(C); // Channel stays the same for depthwise conv

  auto resultType = RankedTensorType::get(outputShape, elementType);
  convOp.getResult().setType(resultType);

  return success();
}

LogicalResult XCOMPILERRequantizeOpShapeInference(
    Operation *op, std::function<void(Region &)> doShapeInference) {
  auto requantizeOp = dyn_cast<XCOMPILERRequantizeOp>(op);
  if (!requantizeOp)
    return failure();

  // Requantize: output shape == input shape. In a quant domain, preserve
  // the result's element type (Requantize bridges distinct quant grids, so
  // the result carries its own y_scale/y_zero_point). Otherwise default to
  // X's element type.
  Value X = requantizeOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  Type elementType = xType.getElementType();
  if (isInQuantizedDomain(op, requantizeOp.getResult()))
    elementType = mlir::cast<ShapedType>(requantizeOp.getResult().getType())
                      .getElementType();
  auto resultType = RankedTensorType::get(xType.getShape(), elementType);
  requantizeOp.getResult().setType(resultType);
  return success();
}

} // namespace mlir
