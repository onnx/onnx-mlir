/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXConstProp.hpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to constprop an ONNX operation into
// composition of other ONNX operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

using namespace mlir;

///// Get the element size in bytes. Use the biggest size to avoid loss in
///// casting.
//int64_t getEltSizeInBytes(mlir::Type ty);
//
///// Get the number of elements.
//int64_t getNumberOfElements(llvm::ArrayRef<int64_t> shape);
//
///// Get the size of a tensor from its ranked type in bytes.
//int64_t getSizeInBytes(mlir::Type ty);

///// Get the size of a tensor from its ranked type in bytes, using the largest
///// precision.
//int64_t getMaxSizeInBytes(mlir::Type ty);
//
///// Compute strides for a given shape.
//std::vector<int64_t> getStrides(llvm::ArrayRef<int64_t> shape);
//
///// Compute the linear access index.
//int64_t getLinearAccessIndex(
//    llvm::ArrayRef<int64_t> indices, llvm::ArrayRef<int64_t> strides);
//
//// Compute the tensor access index from a linear index.
//std::vector<int64_t> getAccessIndex(
//    int64_t linearIndex, llvm::ArrayRef<int64_t> strides);
//
///// A helper function to contruct a RankedTensorType from a ShapedType.
//mlir::RankedTensorType constructRankedTensorType(mlir::ShapedType type);
//
///// Allocate a buffer whose size is getting from a given Value's type.
//char *allocateBufferFor(mlir::Type type, bool useMaxSize = false);
//
///// Get a data array from a given ONNXConstantOp.
//char *createArrayFromDenseAttribute(mlir::DenseElementsAttr dataAttr);
//
///// A helper function to construct a DenseElementsAttr from an array.
//mlir::DenseElementsAttr createDenseElementsAttr(
//    char *arr, mlir::Type outputType);
//
///// Convert an array whose element type is double or int_64 to an array whose
///// element type is the one of 'outType' (smaller precision). It does not
///// support converting from floating point to integer and vise versa.
//void convertDoubleInt64ToExactType(
//    mlir::Type outType, char *inArr, char *outArr);
//
///// Constant propagation for split.
//template <typename T>
//void IterateConstPropSplit(char *constArray, llvm::ArrayRef<int64_t> constShape,
//    uint64_t splitAxis, llvm::ArrayRef<int64_t> splitOffsets,
//    llvm::ArrayRef<mlir::Type> replacingTypes, std::vector<char *> &resBuffers);
//
///// Constant propagation for transpose.
//template <typename T>
//void IterateConstPropTranspose(char *constArray,
//    llvm::ArrayRef<int64_t> constShape, llvm::ArrayRef<uint64_t> perm,
//    llvm::ArrayRef<int64_t> resShape, char *resArray);
