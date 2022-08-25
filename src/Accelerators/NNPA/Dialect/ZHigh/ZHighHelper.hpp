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
ZTensorEncodingAttr::DataLayout convertStringAttrToDataLayout(
    mlir::StringAttr layoutAttr);

/// Get a ztensor data layout by rank.
ZTensorEncodingAttr::DataLayout getDataLayoutByRank(int64_t rank);

/// Convert a data layout to StringAttr.
mlir::StringAttr convertDataLayoutToStringAttr(
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

/// Get the layout of a ztensor.
mlir::Value getMinusBcastConst(mlir::OpBuilder &builder, mlir::Location loc,
    mlir::FloatAttr floatAttr, mlir::Value input);

} // namespace zhigh
} // namespace onnx_mlir
