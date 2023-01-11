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

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include <math.h>

namespace onnx_mlir {

/// Get the size of a tensor from its ranked type in bytes, using the largest
/// precision.
int64_t getMaxSizeInBytes(mlir::Type ty);

/// Compute strides for a given shape.
std::vector<int64_t> getStrides(llvm::ArrayRef<int64_t> shape);

/// Compute the linear access index.
int64_t getLinearAccessIndex(
    llvm::ArrayRef<int64_t> indices, llvm::ArrayRef<int64_t> strides);

// Compute the tensor access index from a linear index.
std::vector<int64_t> getAccessIndex(
    int64_t linearIndex, llvm::ArrayRef<int64_t> strides);

/// A helper function to contruct a RankedTensorType from a ShapedType.
mlir::RankedTensorType constructRankedTensorType(mlir::ShapedType type);

/// Allocate a buffer whose size is getting from a given Value's type.
char *allocateBufferFor(mlir::Type type, bool useMaxSize = false);

/// Get a data array from a given ONNXConstantOp.
char *createArrayFromDenseElementsAttr(mlir::DenseElementsAttr dataAttr);

/// Copy and cast an array of a type to another array of another type.
/// It simply uses C++ type casting. Users must take care about precision loss.
template <typename SRC_TYPE, typename DEST_TYPE>
void copyAndCastArr(char *srcRawArr, char *destRawArr, int64_t size);

/// Convert an array whose element type is double or int_64 to an array whose
/// element type is the one of 'outType' (smaller precision). It does not
/// support converting from floating point to integer and vise versa.
void convertDoubleInt64ToExactType(
    mlir::Type destType, char *srcRawArr, char *destRawArr);

/// Constant propagation for split.
void ConstPropSplitImpl(mlir::Type elementType, char *constArray,
    llvm::ArrayRef<int64_t> constShape, uint64_t splitAxis,
    llvm::ArrayRef<int64_t> splitOffsets,
    llvm::ArrayRef<mlir::Type> replacingTypes, std::vector<char *> &resBuffers);

/// Constant propagation for transpose.
void ConstPropTransposeImpl(mlir::Type elementType, char *constArray,
    llvm::ArrayRef<int64_t> constShape, llvm::ArrayRef<uint64_t> perm,
    llvm::ArrayRef<int64_t> resShape, char *resArray);

} // namespace onnx_mlir