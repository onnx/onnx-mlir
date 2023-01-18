/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Pooling.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Pooling operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

#include "src/Dialect/ONNX/ONNXOps/NN/NNHelper.cpp.inc"

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <typename OP_TYPE>
LogicalResult ONNXGenericGlobalPoolOpShapeHelper<OP_TYPE>::computeShape() {
  typename OP_TYPE::Adaptor operandAdaptor(operands);
  DimsExpr xDims, outputDims;
  createIE->getShapeAsDims(operandAdaptor.X(), xDims);
  if (xDims.size() < 3)
    return op->emitError("Data input shape must be at least (NxCxD1)");
  // Keep first two dims.
  outputDims.emplace_back(xDims[0]);
  outputDims.emplace_back(xDims[1]);
  // Spatial dimensions are reduced to 1.
  for (int i = 2; i < (int)xDims.size(); ++i)
    outputDims.emplace_back(LiteralIndexExpr(1));
  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

template <>
LogicalResult ONNXMaxRoiPoolOpShapeHelper::computeShape() {
  ONNXMaxRoiPoolOpAdaptor operandAdaptor(operands, op->getAttrDictionary());

  IndexExpr channel = createIE->getShapeAsDim(operandAdaptor.X(), 1);
  uint64_t roisRank = createIE->getShapedTypeRank(operandAdaptor.rois());
  if (roisRank != 2)
    return op->emitError("rois rank is expected to be 2d");

  // 2d tensor: (num_rois, 5)
  IndexExpr numRois = createIE->getShapeAsDim(operandAdaptor.rois(), 0);
  DimsExpr pooledDims;
  createIE->getIntFromArrayAsLiterals(
      operandAdaptor.pooled_shape(), pooledDims);

  // 4-D tensor : (num_rois, channels, pooled_shape[0], pooled_shape[1]).
  DimsExpr outputDims;
  outputDims.push_back(LiteralIndexExpr(numRois));
  outputDims.push_back(channel);
  outputDims.push_back(pooledDims[0]);
  outputDims.push_back(pooledDims[1]);

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// AveragePool
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXAveragePoolOpShapeHelper::computeShape() {
  ONNXAveragePoolOpAdaptor operandAdaptor = ONNXAveragePoolOpAdaptor(operands);
  ONNXAveragePoolOp poolOp = llvm::cast<ONNXAveragePoolOp>(op);
  return customComputeShape(operandAdaptor.X(), /*W*/ nullptr,
      poolOp.kernel_shape(), poolOp.auto_pad(), poolOp.pads(), poolOp.strides(),
      /*dilation*/ None, /*hasFilter*/ false, poolOp.ceil_mode());
}

} // namespace onnx_mlir

LogicalResult ONNXAveragePoolOp::verify() {
  ONNXAveragePoolOpAdaptor operandAdaptor = ONNXAveragePoolOpAdaptor(*this);

  // Mandatory and unsupported parameters.
  if (!kernel_shape())
    return emitOpError("kernel_shape is a mandatory attribute");
  // Get spatial rank from mandatory kernel_shape parameter.
  int64_t spatialRank = kernel_shape().size();
  if (spatialRank < 1)
    return emitOpError("Spatial rank must be strictly positive");

  // Get operands.
  auto X = operandAdaptor.X();
  if (hasShapeAndRank(X)) {
    auto xShape = X.getType().cast<ShapedType>().getShape();
    if ((int64_t)xShape.size() - 2 != spatialRank)
      return emitOpError("Input and kernel shape rank mismatch");
  }

  // Verify parameters.
  if (failed(verifyKernelShape<ONNXAveragePoolOp>(
          this, nullptr, kernel_shape(), spatialRank)))
    return failure();
  if (failed(verifyStrides<ONNXAveragePoolOp>(this, spatialRank)))
    return failure();
  if (failed(verifyPadding<ONNXAveragePoolOp>(this, spatialRank)))
    return failure();
  return success();
}

LogicalResult ONNXAveragePoolOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(X()))
    return success();

  Type elementType = X().getType().cast<ShapedType>().getElementType();
  ONNXAveragePoolOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// GlobalAveragePool
//===----------------------------------------------------------------------===//

LogicalResult ONNXGlobalAveragePoolOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type elementType = X().getType().cast<ShapedType>().getElementType();
  ONNXGlobalAveragePoolOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// GlobalLpPool
//===----------------------------------------------------------------------===//

LogicalResult ONNXGlobalLpPoolOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type elementType = X().getType().cast<ShapedType>().getElementType();
  ONNXGlobalLpPoolOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// GlobalMaxPool
//===----------------------------------------------------------------------===//

LogicalResult ONNXGlobalMaxPoolOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type elementType = X().getType().cast<ShapedType>().getElementType();
  ONNXGlobalMaxPoolOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// MaxPoolSingleOut
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXMaxPoolSingleOutOpShapeHelper::computeShape() {
  ONNXMaxPoolSingleOutOpAdaptor operandAdaptor =
      ONNXMaxPoolSingleOutOpAdaptor(operands);
  ONNXMaxPoolSingleOutOp poolOp = llvm::cast<ONNXMaxPoolSingleOutOp>(op);
  return customComputeShape(operandAdaptor.X(), /*W*/ nullptr,
      poolOp.kernel_shape(), poolOp.auto_pad(), poolOp.pads(), poolOp.strides(),
      poolOp.dilations(), /*hasFilter*/ false, poolOp.ceil_mode());
}

} // namespace onnx_mlir

LogicalResult ONNXMaxPoolSingleOutOp::verify() {
  ONNXMaxPoolSingleOutOpAdaptor operandAdaptor =
      ONNXMaxPoolSingleOutOpAdaptor(*this);

  // Mandatory and unsupported parameters.
  if (!kernel_shape())
    return emitOpError("kernel_shape is a mandatory attribute");
  // Get spatial rank from mandatory kernel_shape parameter.
  int64_t spatialRank = kernel_shape().size();
  if (spatialRank < 1)
    return emitOpError("Spatial rank must be strictly positive");
  // Not supported for storage order in column major mode.
  if (storage_order() != 0)
    return emitOpError("Column major storage order not implemented yet");

  // Get operands.
  auto X = operandAdaptor.X();
  if (hasShapeAndRank(X)) {
    auto xShape = X.getType().cast<ShapedType>().getShape();
    if (static_cast<int64_t>(xShape.size()) - 2 != spatialRank)
      return emitOpError("Input and kernel shape rank mismatch");
  }

  // Verify parameters.
  if (failed(verifyKernelShape<ONNXMaxPoolSingleOutOp>(
          this, nullptr, kernel_shape(), spatialRank)))
    return failure();
  if (failed(verifyStrides<ONNXMaxPoolSingleOutOp>(this, spatialRank)))
    return failure();
  if (failed(verifyDilations<ONNXMaxPoolSingleOutOp>(this, spatialRank)))
    return failure();
  if (failed(verifyPadding<ONNXMaxPoolSingleOutOp>(this, spatialRank)))
    return failure();
  return success();
}

LogicalResult ONNXMaxPoolSingleOutOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(X()))
    return success();

  // Verify parameters: mandatory for kernel shape.
  auto kernelShape = kernel_shape();
  assert(kernelShape && "verified that we had kernel shape");

  Type elementType = X().getType().cast<ShapedType>().getElementType();
  IndexExprBuilderForAnalysis createIE(getLoc());
  ONNXMaxPoolSingleOutOpShapeHelper shapeHelper(getOperation(), {}, &createIE);
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// MaxRoiPoolOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMaxRoiPoolOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(X()) || !hasShapeAndRank(rois()))
    return success();

  Type elementType = X().getType().cast<RankedTensorType>().getElementType();
  ONNXMaxRoiPoolOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct ONNXGenericGlobalPoolOpShapeHelper<ONNXGlobalAveragePoolOp>;
template struct ONNXGenericGlobalPoolOpShapeHelper<ONNXGlobalLpPoolOp>;
template struct ONNXGenericGlobalPoolOpShapeHelper<ONNXGlobalMaxPoolOp>;
template struct ONNXGenericPoolOpShapeHelper<ONNXAveragePoolOp>;
template struct ONNXGenericPoolOpShapeHelper<ONNXMaxPoolSingleOutOp>;
template struct ONNXNonSpecificOpShapeHelper<ONNXMaxRoiPoolOp>;

} // namespace onnx_mlir
