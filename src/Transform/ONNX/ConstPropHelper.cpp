/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXConstProp.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to constprop an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the constpropd operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/ConstPropHelper.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;
using namespace onnx_mlir;

/// Get the size of a tensor from its ranked type in bytes, using the largest
/// precision.
int64_t getMaxSizeInBytes(Type ty) { return getNumberOfElements(ty) * 8; }

/// Compute strides for a given shape.
std::vector<int64_t> getStrides(ArrayRef<int64_t> shape) {
  int rank = shape.size();
  std::vector<int64_t> strides;
  int64_t count = 1;
  for (int i = rank - 1; i >= 0; i--) {
    strides.insert(strides.begin(), count);
    count *= shape[i];
  }
  return strides;
}

/// Compute the linear access index.
int64_t getLinearAccessIndex(
    ArrayRef<int64_t> indices, ArrayRef<int64_t> strides) {
  int64_t index = 0;
  for (unsigned int i = 0; i < strides.size(); ++i)
    index += indices[i] * strides[i];
  return index;
}

// Compute the tensor access index from a linear index.
std::vector<int64_t> getAccessIndex(
    int64_t linearIndex, ArrayRef<int64_t> strides) {
  std::vector<int64_t> res;
  for (unsigned int i = 0; i < strides.size(); ++i) {
    int64_t s = strides[i];
    if (linearIndex < s) {
      res.emplace_back(0);
    } else {
      res.emplace_back(floor(linearIndex / s));
      linearIndex = linearIndex % s;
    }
  }
  return res;
}

/// Allocate a buffer whose size is getting from a given Value's type.
char *allocateBufferFor(Type type, bool useMaxSize) {
  assert(type.isa<ShapedType>() && "Not a shaped type");
  int64_t sizeInBytes;
  if (useMaxSize)
    sizeInBytes = getMaxSizeInBytes(type.cast<ShapedType>());
  else
    sizeInBytes = getSizeInBytes(type.cast<ShapedType>());
  char *res = (char *)malloc(sizeInBytes);
  memset(res, 0, sizeInBytes);
  return res;
}

/// Get a data array from a given ONNXConstantOp.
char *createArrayFromDenseElementsAttr(DenseElementsAttr dataAttr) {
  Type elementType = getElementType(dataAttr.getType());
  int64_t numElements = getNumberOfElements(dataAttr.getType());
  char *res = allocateBufferFor(dataAttr.getType(), /*useMaxSize=*/true);
  if (elementType.isa<FloatType>()) {
    // Use double to avoid the precision loss during computation.
    double *resArr = (double *)res;
    auto valueIt = dataAttr.getValues<APFloat>().begin();
    for (int64_t i = 0; i < numElements; ++i) {
      double val = (*valueIt++).convertToDouble();
      *(resArr + i) = val;
    }
  } else if (elementType.isa<IntegerType>()) {
    // Use int64_t to avoid the precision loss during computation.
    int64_t *resArr = (int64_t *)res;
    auto valueIt = dataAttr.getValues<APInt>().begin();
    for (int64_t i = 0; i < numElements; ++i) {
      int64_t val = (*valueIt++).getSExtValue();
      *(resArr + i) = val;
    }
  } else
    llvm_unreachable("Unknown data type");
  return res;
}

template <typename SRC_TYPE, typename DEST_TYPE>
void copyAndCastArr(char *srcRawArr, char *destRawArr, int64_t size) {
  SRC_TYPE *srcArr = (SRC_TYPE *)srcRawArr;
  DEST_TYPE *destArr = (DEST_TYPE *)destRawArr;
  std::transform(
      srcArr, srcArr + size, destArr, [](SRC_TYPE v) { return (DEST_TYPE)v; });
}

/// Convert an array whose element type is double or int_64 to an array whose
/// element type is the one of 'destType' (smaller precision). It does not
/// support converting from floating point to integer and vise versa.
void convertDoubleInt64ToExactType(
    Type destType, char *srcRawArr, char *destRawArr) {
  int64_t numElements = getNumberOfElements(destType);
  Type destElemTy = getElementType(destType);

  if (destElemTy.isa<FloatType>()) {
    FloatType destFloatTy = destElemTy.cast<FloatType>();
    if (destFloatTy.getWidth() == 32) // to f32
      copyAndCastArr<double, float>(srcRawArr, destRawArr, numElements);
    else if (destFloatTy.getWidth() == 64) // to f64
      copyAndCastArr<double, double>(srcRawArr, destRawArr, numElements);
    else
      llvm_unreachable("Unknown data type");
  } else if (destElemTy.isa<IntegerType>()) {
    IntegerType destIntTy = destElemTy.cast<IntegerType>();
    if (destIntTy.getWidth() == 1) // to bool
      copyAndCastArr<int64_t, bool>(srcRawArr, destRawArr, numElements);
    else if (destIntTy.getWidth() == 8) // to i8
      copyAndCastArr<int64_t, int8_t>(srcRawArr, destRawArr, numElements);
    else if (destIntTy.getWidth() == 16) // to i16
      copyAndCastArr<int64_t, int16_t>(srcRawArr, destRawArr, numElements);
    else if (destIntTy.getWidth() == 32) // to i32
      copyAndCastArr<int64_t, int32_t>(srcRawArr, destRawArr, numElements);
    else if (destIntTy.getWidth() == 64) // to i64
      copyAndCastArr<int64_t, int64_t>(srcRawArr, destRawArr, numElements);
    else
      llvm_unreachable("Unknown data type");
  } else
    llvm_unreachable("Unknown data type");
}

/// A helper function to contruct a RankedTensorType from a ShapedType.
RankedTensorType constructRankedTensorType(ShapedType type) {
  assert(type.hasRank() && "Not a ranked type");
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for split.
//===----------------------------------------------------------------------===//

template <typename T>
void IterateConstPropSplit(char *constArray, ArrayRef<int64_t> constShape,
    uint64_t splitAxis, ArrayRef<int64_t> splitOffsets,
    ArrayRef<Type> replacingTypes, std::vector<char *> &resBuffers) {
  // Basic info.
  unsigned int rank = constShape.size();
  unsigned int numOfResults = replacingTypes.size();

  // Data pointers.
  T *constArrayT = reinterpret_cast<T *>(constArray);
  // Strides info.
  std::vector<int64_t> constStrides = getStrides(constShape);

  // Allocate temporary buffers.
  for (unsigned int i = 0; i < numOfResults; ++i) {
    // Use maximum size (double or int64_t) to avoid the precision loss.
    char *resArray = allocateBufferFor(replacingTypes[i], /*useMaxSize=*/true);
    resBuffers.emplace_back(resArray);
  }

  // Do splitting
  for (int64_t i = 0; i < ShapedType::getNumElements(constShape); ++i) {
    // Input indices.
    std::vector<int64_t> constIndices = getAccessIndex(i, constStrides);

    // Find the corresponding output and compute access indices.
    int toResult = numOfResults - 1;
    SmallVector<int64_t, 4> resIndices(rank, 0);
    for (unsigned int r = 0; r < rank; ++r) {
      if (r == splitAxis) {
        for (int k = 0; k < (int)numOfResults - 1; ++k)
          if (constIndices[r] >= splitOffsets[k] &&
              constIndices[r] < splitOffsets[k + 1]) {
            toResult = k;
            break;
          }
        resIndices[r] = constIndices[r] - splitOffsets[toResult];
      } else {
        resIndices[r] = constIndices[r];
      }
    }

    // Get linear access indices.
    std::vector<int64_t> resStrides =
        getStrides(replacingTypes[toResult].cast<ShapedType>().getShape());
    int64_t resOffset = getLinearAccessIndex(resIndices, resStrides);

    // Copy data.
    T *resArrayT = reinterpret_cast<T *>(resBuffers[toResult]);
    *(resArrayT + resOffset) = *(constArrayT + i);
  }
}

void ConstPropSplitImpl(Type elementType, char *constArray,
    llvm::ArrayRef<int64_t> constShape, uint64_t splitAxis,
    llvm::ArrayRef<int64_t> splitOffsets,
    llvm::ArrayRef<mlir::Type> replacingTypes,
    std::vector<char *> &resBuffers) {
  if (elementType.isa<FloatType>()) {
    IterateConstPropSplit<double>(constArray, constShape, splitAxis,
        splitOffsets, replacingTypes, resBuffers);
  } else if (elementType.isa<IntegerType>()) {
    IterateConstPropSplit<int64_t>(constArray, constShape, splitAxis,
        splitOffsets, replacingTypes, resBuffers);
  } else
    llvm_unreachable("Unknown data type");
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for transpose.
//===----------------------------------------------------------------------===//

template <typename T>
void IterateConstPropTranspose(char *constArray, ArrayRef<int64_t> constShape,
    ArrayRef<uint64_t> perm, ArrayRef<int64_t> resShape, char *resArray) {
  // Data pointers.
  T *constArrayT = reinterpret_cast<T *>(constArray);
  T *resArrayT = reinterpret_cast<T *>(resArray);

  // Get a reversed perm.
  SmallVector<uint64_t, 4> reversedPerm(perm.size(), 0);
  for (unsigned int i = 0; i < perm.size(); ++i)
    reversedPerm[perm[i]] = i;

  // Strides info.
  std::vector<int64_t> constStrides = getStrides(constShape);
  std::vector<int64_t> resStrides = getStrides(resShape);

  // Calculate transpose result.
  for (int64_t i = 0; i < ShapedType::getNumElements(resShape); ++i) {
    // Indices.
    std::vector<int64_t> resIndices = getAccessIndex(i, resStrides);
    SmallVector<int64_t, 4> constIndices(perm.size(), 0);
    for (unsigned int j = 0; j < constIndices.size(); ++j)
      constIndices[j] = resIndices[reversedPerm[j]];
    // Transpose.
    int64_t constOffset = getLinearAccessIndex(constIndices, constStrides);
    int64_t resOffset = getLinearAccessIndex(resIndices, resStrides);
    *(resArrayT + resOffset) = *(constArrayT + constOffset);
  }
}

void ConstPropTransposeImpl(Type elementType, char *constArray,
    llvm::ArrayRef<int64_t> constShape, llvm::ArrayRef<uint64_t> perm,
    llvm::ArrayRef<int64_t> resShape, char *resArray) {
  if (elementType.isa<FloatType>()) {
    // Use double to avoid the precision loss during computation.
    IterateConstPropTranspose<double>(
        constArray, constShape, perm, resShape, resArray);
  } else if (elementType.isa<IntegerType>()) {
    // Use int64_t to avoid the precision loss during computation.
    IterateConstPropTranspose<int64_t>(
        constArray, constShape, perm, resShape, resArray);
  } else
    llvm_unreachable("Unknown data type");
}

/// Explicit instantiation of all templated API functions.
template void copyAndCastArr<double, bool>(
    char *srcRawArr, char *destRawArr, int64_t size);
template void copyAndCastArr<double, int8_t>(
    char *srcRawArr, char *destRawArr, int64_t size);
template void copyAndCastArr<double, int32_t>(
    char *srcRawArr, char *destRawArr, int64_t size);
template void copyAndCastArr<double, int64_t>(
    char *srcRawArr, char *destRawArr, int64_t size);
template void copyAndCastArr<double, float>(
    char *srcRawArr, char *destRawArr, int64_t size);
template void copyAndCastArr<double, double>(
    char *srcRawArr, char *destRawArr, int64_t size);

template void copyAndCastArr<int64_t, bool>(
    char *srcRawArr, char *destRawArr, int64_t size);
template void copyAndCastArr<int64_t, int8_t>(
    char *srcRawArr, char *destRawArr, int64_t size);
template void copyAndCastArr<int64_t, int32_t>(
    char *srcRawArr, char *destRawArr, int64_t size);
template void copyAndCastArr<int64_t, int64_t>(
    char *srcRawArr, char *destRawArr, int64_t size);
template void copyAndCastArr<int64_t, float>(
    char *srcRawArr, char *destRawArr, int64_t size);
template void copyAndCastArr<int64_t, double>(
    char *srcRawArr, char *destRawArr, int64_t size);
