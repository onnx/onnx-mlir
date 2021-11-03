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

using namespace mlir;

/// Get the element size in bytes. Use the biggest size to avoid loss in
/// casting.
int64_t getEltSizeInBytes(Type ty) {
  auto elementType = ty.cast<ShapedType>().getElementType();

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

/// Get the number of elements.
int64_t getNumberOfElements(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (unsigned int i = 0; i < shape.size(); ++i) {
    count *= shape[i];
  }
  return count;
}

/// Get the size of a tensor from its ranked type in bytes.
int64_t getSizeInBytes(Type ty) {
  ShapedType shapedType = ty.dyn_cast<ShapedType>();
  auto shape = shapedType.getShape();
  return getNumberOfElements(shape) * getEltSizeInBytes(shapedType);
}

/// Get the size of a tensor from its ranked type in bytes, using the largest
/// precision.
int64_t getMaxSizeInBytes(Type ty) {
  auto shape = ty.dyn_cast<ShapedType>().getShape();
  return getNumberOfElements(shape) * 8;
}

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
  return res;
}

/// Get a data array from a given ONNXConstantOp.
char *createArrayFromDenseElementsAttr(DenseElementsAttr dataAttr) {
  Type elementType = dataAttr.getType().getElementType();
  int64_t numElements = getNumberOfElements(dataAttr.getType().getShape());
  char *res = allocateBufferFor(dataAttr.getType(), /*useMaxSize=*/true);
  if (elementType.isa<FloatType>()) {
    // Use double to avoid the precision loss during computation.
    double *resArr = (double *)res;
    auto valueIt = dataAttr.getValues<APFloat>().begin();
    for (int64_t i = 0; i < numElements; ++i) {
      double val = (double)(*valueIt++).convertToFloat();
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

/// A helper function to construct a DenseElementsAttr from an array.
DenseElementsAttr createDenseElementsAttrFromArray(char *arr, Type outputType) {
  int64_t sizeInBytes = getSizeInBytes(outputType);
  RankedTensorType resType =
      constructRankedTensorType(outputType.cast<ShapedType>());
  bool isSplat;
  if (resType.getShape().size() == 0)
    isSplat = true;
  else if (llvm::all_of(
               resType.getShape(), [](int64_t dim) { return dim == 1; }))
    isSplat = true;
  else
    isSplat = false;
  return DenseElementsAttr::getFromRawBuffer(
      resType, ArrayRef<char>(arr, sizeInBytes), /*isSplat=*/isSplat);
}

/// Create a dense ONNXConstantOp from a byte array.
ONNXConstantOp createDenseONNXConstantOp(PatternRewriter &rewriter,
    Location loc, ShapedType resultType, char *array) {
  char *resArray = allocateBufferFor(resultType);
  convertDoubleInt64ToExactType(resultType, array, resArray);
  DenseElementsAttr denseAttr =
      createDenseElementsAttrFromArray(resArray, resultType);
  free(resArray);
  return rewriter.create<ONNXConstantOp>(loc, resultType, Attribute(),
      denseAttr, FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(),
      StringAttr(), ArrayAttr());
}

/// Convert an array whose element type is double or int_64 to an array whose
/// element type is the one of 'outType' (smaller precision). It does not
/// support converting from floating point to integer and vise versa.
void convertDoubleInt64ToExactType(Type outType, char *inArr, char *outArr) {
  ShapedType shapedType = outType.cast<ShapedType>();
  int64_t maxSizeInBytes = getMaxSizeInBytes(shapedType);
  int64_t numElements = getNumberOfElements(shapedType.getShape());
  Type elementType = shapedType.getElementType();

  if (elementType.isa<FloatType>()) {
    FloatType floatTy = elementType.cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      double *inArrDouble = (double *)inArr;
      float *inArrFloat = (float *)outArr;
      for (int64_t i = 0; i < numElements; ++i)
        *(inArrFloat + i) = (float)*(inArrDouble + i);
    } else if (floatTy.getWidth() == 64) {
      std::copy(inArr, inArr + maxSizeInBytes, outArr);
    } else
      llvm_unreachable("Unknown data type");
  } else if (elementType.isa<IntegerType>()) {
    IntegerType intTy = elementType.cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      int64_t *inArrInt64 = (int64_t *)inArr;
      int32_t *inArrInt32 = (int32_t *)outArr;
      for (int64_t i = 0; i < numElements; ++i)
        *(inArrInt32 + i) = (int32_t)(*(inArrInt64 + i));
    } else if (intTy.getWidth() == 64) {
      std::copy(inArr, inArr + maxSizeInBytes, outArr);
    } else
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
  for (int64_t i = 0; i < getNumberOfElements(constShape); ++i) {
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
  for (int64_t i = 0; i < getNumberOfElements(resShape); ++i) {
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
