/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ZHighToZLow.hpp - ZHigh dialect to ZLow lowering -------------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZHigh operations to ZLow operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Dialect/Mlir/IndexExpr.hpp"

/// Default 4K alignment for sticked tensors.
static constexpr int64_t gAlignment = 4096;

namespace onnx_mlir {
namespace zhigh {

/// A struct to store a MemRefType and a layout attribute, which is to
/// encapsulate ZTensor type.
struct ZMemRefType {
  mlir::MemRefType value;
  mlir::StringAttr layout;
};

/// Get the corresponding MemRefType and layout of a given ZTensorType.
ZMemRefType convertZTensorToMemRefType(mlir::Type type);

/// Emit instructions to allocate a buffer to store original dimensions.
mlir::Value insertShapeMemRefI64(mlir::PatternRewriter &rewriter,
    mlir::Location loc, mlir::ArrayRef<IndexExpr> originalDims);

/// Insert an allocation and deallocation for the given dimensions and layout.
/// By default, set aligment to 4K.
mlir::Value insertAllocAndDeallocZMemRefByDim(mlir::ArrayRef<IndexExpr> dims,
    mlir::Type layoutType, mlir::Operation *op, mlir::PatternRewriter &rewriter,
    int64_t alignment);

/// Insert an allocation and deallocation for the given ZMemRefType.
/// By default, set aligment to 4K.
mlir::Value insertAllocAndDeallocZMemRef(ZMemRefType zType,
    mlir::ArrayRef<IndexExpr> dims, mlir::Operation *op,
    mlir::PatternRewriter &rewriter, int64_t alignment);

/// Populate all conversion patterns for ZHigh Ops.
void populateZHighToZLowConversionPattern(mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx);

} // namespace zhigh
} // namespace onnx_mlir
