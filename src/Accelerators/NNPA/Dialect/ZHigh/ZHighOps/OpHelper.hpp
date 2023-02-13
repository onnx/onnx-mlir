/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZHighHelper.hpp - ZHigh Helper Functions --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#pragma once

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

/// Get a minus value.
mlir::Value getMinusBcastConst(mlir::OpBuilder &builder, mlir::Location loc,
    mlir::FloatAttr floatAttr, mlir::Value input);

/// True if at least one of the types is NHWC layout.
bool oneIsOfNHWCLayout(mlir::Type t1, mlir::Type t2);

/// Check if ONNXReshapeOp is reshaping 2D/3D to 4D by tiling each input
/// dimension.
bool isTiling2DTo4D(mlir::Value val);
mlir::AffineMapAttr getTiling2DTo4DMap(mlir::OpBuilder &b, mlir::Value val);
bool isTiling3DTo4D(mlir::Value val);
mlir::AffineMapAttr getTiling3DTo4DMap(mlir::OpBuilder &b, mlir::Value val);
/// Check if ONNXReshapeOp is collapsing 4D into 3D/2D by merging the first two
/// dimensions.
bool isCollapsing4DTo3D(mlir::Value val);
mlir::AffineMapAttr getCollapsing4DTo3DMap(mlir::OpBuilder &b, mlir::Value val);
bool isCollapsing4DTo2D(mlir::Value val);
mlir::AffineMapAttr getCollapsing4DTo2DMap(mlir::OpBuilder &b, mlir::Value val);
/// Get an affine map for the permutation array.
mlir::AffineMapAttr getTransposeMap(
    mlir::OpBuilder &b, mlir::ArrayAttr permAttr);

} // namespace zhigh
} // namespace onnx_mlir
