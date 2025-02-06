/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZHighHelper.hpp - ZHigh Helper Functions --------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OP_HELPER_H
#define ONNX_MLIR_OP_HELPER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"

namespace onnx_mlir {
namespace zhigh {

/// Check if a value type is ranked or unranked.
bool hasRankedType(mlir::Value val);

/// Get a ztensor data layout by StringAttr.
ZTensorEncodingAttr::DataLayout convertStringAttrToZTensorDataLayout(
    mlir::StringAttr layoutAttr);

/// Get a ztensor data layout by rank.
ZTensorEncodingAttr::DataLayout getZTensorDataLayoutByRank(int64_t rank);

/// Convert a data layout to StringAttr.
mlir::StringAttr convertZTensorDataLayoutToStringAttr(
    mlir::OpBuilder &builder, ZTensorEncodingAttr::DataLayout layout);

/// Get a ztensor quantized type by StringAttr.
ZTensorEncodingAttr::QuantizedType convertStringAttrToZTensorQuantizedType(
    mlir::StringAttr qtypeAttr);

/// Convert a quantized type to StringAttr.
mlir::StringAttr convertZTensorQuantizedTypeToStringAttr(
    mlir::OpBuilder &builder, ZTensorEncodingAttr::QuantizedType qtype);

//===----------------------------------------------------------------------===//
// Convenience method to query information of a ztensor

/// Return true if the tensor is a ztensor (having ZTensorEncodingAttr).
bool isZTensor(mlir::Type type);

/// Get a ztensor encoding attribute from a type.Returns null-attribute for any
/// type without an encoding.
ZTensorEncodingAttr getZTensorEncoding(mlir::Type type);

/// Get the layout of a ztensor.
ZTensorEncodingAttr::DataLayout getZTensorLayout(mlir::Type type);

/// Get the layout attribute of a ztensor.
mlir::StringAttr getZTensorLayoutAttr(
    mlir::OpBuilder &builder, mlir::Type type);

/// Get the quantized type of a ztensor.
ZTensorEncodingAttr::QuantizedType getZTensorQuantizedType(mlir::Type type);

/// Get a minus value.
mlir::Value getMinusBcastConst(mlir::OpBuilder &builder, mlir::Location loc,
    mlir::FloatAttr floatAttr, mlir::Value input);

// Get a constant tensor of given value and type.
mlir::Value getConstantOfType(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Type type, float val);

/// True if at least one of the types is `layout`.
bool oneIsOfLayout(
    mlir::Type t1, mlir::Type t2, ZTensorEncodingAttr::DataLayout layout);

/// Check if ONNXReshapeOp is reshaping 2D/3D to 4D by tiling an input
/// dimension.
bool isTiling2DTo4D(mlir::Value val);
mlir::AffineMapAttr getTiling2DTo4DMap(mlir::OpBuilder &b, mlir::Value val);
bool isLeftmostTiling3DTo4D(mlir::Value val);
bool isRightmostTiling3DTo4D(mlir::Value val, int64_t tilingSize);
mlir::AffineMapAttr getLeftmostTiling3DTo4DMap(
    mlir::OpBuilder &b, mlir::Value val);
/// Check if ONNXReshapeOp is collapsing 4D into 3D by merging the first two
/// (leftmost) dimensions.
bool isLeftmostCollapsing4DTo3D(mlir::Value val);
/// Check if ONNXReshapeOp is collapsing 4D into 3D by merging the last two
/// (rightmost) dimensions.
bool isRightmostCollapsing4DTo3D(mlir::Value val);
mlir::AffineMapAttr getLeftmostCollapsing4DTo3DMap(
    mlir::OpBuilder &b, mlir::Value val);
bool isCollapsing4DTo2D(mlir::Value val);
mlir::AffineMapAttr getCollapsing4DTo2DMap(mlir::OpBuilder &b, mlir::Value val);
/// Get an affine map for the permutation array.
mlir::AffineMapAttr getTransposeMap(
    mlir::OpBuilder &b, mlir::ArrayAttr permAttr);
/// Check the values of a transpose map to be equal to the permVals.
bool isTransposePermutationEqualTo(
    mlir::ArrayAttr permAttr, mlir::ArrayRef<int64_t> permVals);
/// Return true when shape(Value)[index] % multipleVal == 0.
/// Negative indices, count from the back (-1 is last element).
bool isShapeDimMultipleOf(mlir::Value val, int64_t index, int64_t multipleVal);
/// Get an axis for NHWC layout given an axis for NCHW layout.
mlir::IntegerAttr getAxisNHWC(mlir::IntegerAttr axisNCHWAttr);

/// Check if the value has NNPA users (or is consumed by an NNPA op).
bool hasNNPAUse(mlir::Value v);

/// Get saturation settings.
mlir::IntegerAttr getDefaultSaturation(mlir::PatternRewriter &rewriter);

} // namespace zhigh
} // namespace onnx_mlir
#endif
