/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- TypeUtilities.cpp - functions related to MLIR Type -------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains support code related to MLIR Type, e.g. RankedTensorType,
// MemRefType, etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

/// Get element type.
Type getElementType(Type ty) { return getElementTypeOrSelf(ty); }

/// Check if a type is ShapedType and has rank.
bool isRankedShapedType(Type ty) {
  return (ty.isa<ShapedType>() && ty.cast<ShapedType>().hasRank());
}

/// Check if a type has static shape.
bool hasStaticShape(mlir::Type ty) {
  if (!isRankedShapedType(ty))
    return false;
  return ty.cast<ShapedType>().hasStaticShape();
}

/// Get shape.
ArrayRef<int64_t> getShape(Type ty) {
  assert(isRankedShapedType(ty) && "Type must be ranked");
  return ty.cast<ShapedType>().getShape();
}

/// Get rank.
int64_t getRank(Type ty) {
  assert(isRankedShapedType(ty) && "Type must be ranked");
  return ty.cast<ShapedType>().getRank();
}

/// Get the number of elements.
int64_t getNumberOfElements(Type ty) {
  ArrayRef<int64_t> shape = getShape(ty);
  return ShapedType::getNumElements(shape);
}

/// Get the element size in bytes.
int64_t getEltSizeInBytes(Type ty) {
  Type elementType = getElementTypeOrSelf(ty);
  int64_t sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

/// Get the size of a tensor from its ranked type in bytes.
int64_t getSizeInBytes(Type ty) {
  ArrayRef<int64_t> shape = getShape(ty);
  assert(ty.cast<ShapedType>().hasStaticShape() && "Has unknown dimensions");
  return ShapedType::getNumElements(shape) * getEltSizeInBytes(ty);
}

} // namespace onnx_mlir
