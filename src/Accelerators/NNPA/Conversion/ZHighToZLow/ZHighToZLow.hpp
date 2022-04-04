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

/// A list of layouts associated with newly allocated MemRefs.
/// When lowering an operation, its output Tensor (e.g.
/// `tensor<1x3x5x7x!zhigh.nhwc<f16>>`) will be converted to a Memref (e.g.
/// `memref<1x3x5x7xf16, #map>`), and we lost the layout `nhwc`.
/// Thus, make sure to put the new MemRef and its associated layout into this
/// map, so that we can obtain the layout for the MemRef later when lowering
/// other ops.
llvm::SmallMapVector<mlir::Value, mlir::StringAttr, 4> stickedLayouts;

mlir::StringAttr readLayout(mlir::Value val) { return stickedLayouts[val]; }

void storeLayout(mlir::Value val, mlir::StringAttr layout) {
  stickedLayouts[val] = layout;
}

/// Get the corresponding MemRefType and layout of a given ZTensorType.
ZMemRefType convertZTensorToMemRefType(mlir::OpBuilder b, mlir::Type type);

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

} // namespace zhigh
} // namespace onnx_mlir
