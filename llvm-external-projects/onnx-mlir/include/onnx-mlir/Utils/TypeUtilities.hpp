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

namespace mlir::onnx_mlir {
/// Get element type.
Type getElementType(Type ty);
/// Check if a type is ShapedType and has rank.
bool isRankedShapedType(Type ty);
/// Check if a type has static shape.
bool hasStaticShape(Type ty);
/// Get shape.
llvm::ArrayRef<int64_t> getShape(Type ty);
/// Get rank.
int64_t getRank(Type ty);
/// Get the number of elements.
int64_t getNumberOfElements(Type ty);
/// Get the element size in bytes.
int64_t getEltSizeInBytes(Type ty);
/// Get the size of a tensor from its ranked type in bytes.
int64_t getSizeInBytes(Type ty);

} // namespace onnx_mlir
