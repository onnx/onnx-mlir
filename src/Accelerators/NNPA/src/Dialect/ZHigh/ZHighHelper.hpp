//===-------- ZHighHelper.hpp - DLC++ ZHigh Helper Functions --------------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "ZHighOps.hpp"

/// Check if a value type is ranked or unranked.
bool hasRankedType(mlir::Value val);

/// Get a ztensor data layout by StringAttr.
mlir::ZTensorEncodingAttr::DataLayout convertStringAttrToDataLayout(
    mlir::StringAttr layoutAttr);

/// Get a ztensor data layout by rank.
mlir::ZTensorEncodingAttr::DataLayout getDataLayoutByRank(int64_t rank);

/// Convert a data layout to StringAttr.
mlir::StringAttr convertDataLayoutToStringAttr(
    mlir::OpBuilder &builder, mlir::ZTensorEncodingAttr::DataLayout layout);

//===----------------------------------------------------------------------===//
// Convenience method to query information of a ztensor

/// Return true if the tensor is a ztensor (having ZTensorEncodingAttr).
bool isZTensor(mlir::Type type);

/// Get a ztensor encoding attribute from a type.Returns null-attribute for any
/// type without an encoding.
mlir::ZTensorEncodingAttr getZTensorEncoding(mlir::Type type);

/// Get the layout of a ztensor.
mlir::ZTensorEncodingAttr::DataLayout getZTensorLayout(mlir::Type type);
