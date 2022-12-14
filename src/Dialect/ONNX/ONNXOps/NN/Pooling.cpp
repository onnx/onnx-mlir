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

namespace {

// Helper function to infer shapes of global pool operations.
template <typename PoolingOp>
static LogicalResult inferShapesGlobalPool(PoolingOp *op) {
  // Cannot infer shape if no shape exists.
  if (!op->X().getType().template isa<RankedTensorType>())
    return success();

  auto xTy = op->X().getType().template cast<RankedTensorType>();
  auto xShape = xTy.getShape();
  xTy.getRank();

  if (xShape.size() < 3) {
    return op->emitError("Data input shape must be at least (NxCxD1)");
  }

  SmallVector<int64_t, 4> outputDims;
  outputDims.emplace_back(xShape[0]);
  outputDims.emplace_back(xShape[1]);
  // Spatial dimensions are reduced to 1.
  outputDims.insert(outputDims.end(), xTy.getRank() - 2, 1);

  op->getResult().setType(
      RankedTensorType::get(outputDims, xTy.getElementType()));
  return success();
}

} // namespace

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
  if (!X().getType().isa<RankedTensorType>())
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
  return inferShapesGlobalPool(this);
}

//===----------------------------------------------------------------------===//
// GlobalLpPool
//===----------------------------------------------------------------------===//

LogicalResult ONNXGlobalLpPoolOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapesGlobalPool(this);
}

//===----------------------------------------------------------------------===//
// GlobalMaxPool
//===----------------------------------------------------------------------===//

LogicalResult ONNXGlobalMaxPoolOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapesGlobalPool(this);
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
  if (!X().getType().isa<RankedTensorType>())
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
  if (!X().getType().isa<RankedTensorType>())
    return success();

  if (!rois().getType().isa<RankedTensorType>())
    return success();

  auto x_type = X().getType().cast<RankedTensorType>();
  auto x_shape = x_type.getShape();
  auto rois_rank = rois().getType().cast<RankedTensorType>().getRank();
  if (rois_rank != 2)
    return success();

  // 2d tensor: (num_rois, 5)
  auto roi_shape = rois().getType().cast<RankedTensorType>().getShape();
  int64_t num_rois = roi_shape[0];
  SmallVector<int64_t, 2> pooled_dims;

  auto pooled_shape_array_attr = pooled_shape();
  for (auto pooled_shape_attr : pooled_shape_array_attr) {
    auto pooled_shape_int_attr = pooled_shape_attr.dyn_cast<IntegerAttr>();
    if (!pooled_shape_int_attr)
      return success();
    pooled_dims.push_back(pooled_shape_int_attr.getInt());
  }

  // 4-D tensor : (num_rois, channels, pooled_shape[0], pooled_shape[1]).
  SmallVector<int64_t, 2> outputDims;
  outputDims.push_back(num_rois);
  outputDims.push_back(x_shape[1]); // channel
  outputDims.push_back(pooled_dims[0]);
  outputDims.push_back(pooled_dims[1]);

  updateType(getResult(), outputDims, x_type.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct ONNXGenericPoolOpShapeHelper<ONNXAveragePoolOp>;
template struct ONNXGenericPoolOpShapeHelper<ONNXMaxPoolSingleOutOp>;

} // namespace onnx_mlir
