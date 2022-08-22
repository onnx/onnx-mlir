/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- TypeUtilities.hpp - functions related to MLIR Type -------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains support code related to MLIR Type, e.g. RankedTensorType,
// MemRefType, etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"

namespace onnx_mlir {
/// Get element type.
mlir::Type getElementType(mlir::Type ty);
/// Check if a type is ShapedType and has rank.
bool isRankedShapedType(mlir::Type ty);
/// Check if a type has static shape.
bool hasStaticShape(mlir::Type ty);
/// Get shape.
llvm::ArrayRef<int64_t> getShape(mlir::Type ty);
/// Get rank.
int64_t getRank(mlir::Type ty);
/// Get the number of elements.
int64_t getNumberOfElements(mlir::Type ty);
/// Get the element size in bytes.
int64_t getEltSizeInBytes(mlir::Type ty);
/// Get the size of a tensor from its ranked type in bytes.
int64_t getSizeInBytes(mlir::Type ty);

} // namespace onnx_mlir
