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
  return (mlir::isa<ShapedType>(ty) && mlir::cast<ShapedType>(ty).hasRank());
}

/// Check if a type has static shape.
bool hasStaticShape(Type ty) {
  if (!isRankedShapedType(ty))
    return false;
  return mlir::cast<ShapedType>(ty).hasStaticShape();
}

/// Get shape.
ArrayRef<int64_t> getShape(Type ty) {
  assert(isRankedShapedType(ty) && "Type must be ranked");
  return mlir::cast<ShapedType>(ty).getShape();
}

/// Get rank.
int64_t getRank(Type ty) {
  assert(isRankedShapedType(ty) && "Type must be ranked");
  return mlir::cast<ShapedType>(ty).getRank();
}

/// Get the number of elements.
int64_t getNumberOfElements(ShapedType ty) {
  assert(ty.hasStaticShape() && "Has unknown dimensions");
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
    auto vectorType = mlir::cast<VectorType>(elementType);
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

/// Get the size of a tensor from its ranked type in bytes.
int64_t getSizeInBytes(ShapedType ty) {
  ArrayRef<int64_t> shape = getShape(ty);
  assert(ty.hasStaticShape() && "Has unknown dimensions");
  return ShapedType::getNumElements(shape) * getEltSizeInBytes(ty);
}

/// Check if two RankedTensorTypes have the same encoding attribute or not.
bool sameEncodingAttr(Type t1, Type t2) {
  if (auto rtp1 = llvm::dyn_cast<RankedTensorType>(t1))
    if (auto rtp2 = llvm::dyn_cast<RankedTensorType>(t2)) {
      return rtp1.getEncoding() == rtp2.getEncoding();
    }
  return false;
}

/// Get the byte width of an int or float type.
unsigned getIntOrFloatByteWidth(Type ty) {
  return llvm::divideCeil(ty.getIntOrFloatBitWidth(), 8);
}

} // namespace onnx_mlir
