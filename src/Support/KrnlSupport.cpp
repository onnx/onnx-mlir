/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- KrnlSupport.cpp - Krnl-level support functions -----------===//
//
// Copyright 2020-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains support code used at the level of the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Support/KrnlSupport.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Return various operations.
//===----------------------------------------------------------------------===//

/// Return the top block.
Block *getTopBlock(Operation *op) {
  // Get current block as the first top block candidate.
  Block *topBlock = op->getBlock();
  Operation *parentBlockOp = topBlock->getParentOp();

  while (!llvm::dyn_cast_or_null<func::FuncOp>(parentBlockOp)) {
    topBlock = parentBlockOp->getBlock();
    parentBlockOp = topBlock->getParentOp();
  }

  return topBlock;
}

/// Retrieve function which contains the current operation.
func::FuncOp getContainingFunction(Operation *op) {
  Operation *parentFuncOp = op->getParentOp();

  // While parent is not a FuncOp and its cast to a FuncOp is null.
  while (!llvm::dyn_cast_or_null<func::FuncOp>(parentFuncOp))
    parentFuncOp = parentFuncOp->getParentOp();

  return cast<func::FuncOp>(parentFuncOp);
}

//===----------------------------------------------------------------------===//
// Perform checks or get statistics about Krnl-level operations.
//===----------------------------------------------------------------------===//

/// Operation is a LoadOp or AffineLoadOp.
bool isLoad(Operation *op) { return llvm::dyn_cast_or_null<KrnlLoadOp>(op); }

/// Operation is a StoreOp or AffineStoreOp.
bool isStore(Operation *op) { return llvm::dyn_cast_or_null<KrnlStoreOp>(op); }

/// Operation is a KrnlMemcpyOp.
bool isKrnlMemcpy(Operation *op) {
  return llvm::dyn_cast_or_null<KrnlMemcpyOp>(op);
}

/// Check if this value is an argument of one of the blocks nested around it.
bool isBlockArgument(Operation *op, Value operand) {
  // Parent operation of the current block.
  Operation *parentBlockOp;
  Block *currentBlock = op->getBlock();

  do {
    // Check the arguments of the current block.
    for (auto arg : currentBlock->getArguments())
      if (operand == arg)
        return true;

    parentBlockOp = currentBlock->getParentOp();
    currentBlock = parentBlockOp->getBlock();

  } while (!llvm::dyn_cast_or_null<func::FuncOp>(parentBlockOp));

  return false;
}

/// Check if an operation is in the top-level block of the function.
bool opInTopLevelBlock(Operation *op) {
  Block *currentBlock = op->getBlock();

  // If the parent operation of the current block is a FuncOp then
  // this operation is in the top-level block.
  return llvm::dyn_cast_or_null<func::FuncOp>(currentBlock->getParentOp());
}

/// This function returns true if `beforeOp` is visited before `op` in a
/// traversal of the provided block.
bool opBeforeOp(Block *block, Operation *beforeOp, Operation *afterOp) {
  bool beforeOpIsBefore = true;
  bool beforeOpFound = false;
  block->walk(
      [&beforeOpIsBefore, &beforeOpFound, beforeOp, afterOp](Operation *op) {
        if (op == beforeOp)
          beforeOpFound = true;
        else if (op == afterOp && !beforeOpFound)
          beforeOpIsBefore = false;
      });
  return beforeOpIsBefore;
}

/// Check if all dimensions are known at compile time.
bool hasAllConstantDimensions(MemRefType memRefType) {
  auto memRefShape = memRefType.getShape();
  for (unsigned int i = 0; i < memRefShape.size(); ++i)
    if (memRefShape[i] == ShapedType::kDynamic)
      return false;
  return true;
}

/// Get the MemRef element size in bytes.
unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
  Type elementType = memRefType.getElementType();

  unsigned sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else if (mlir::isa<krnl::StringType>(elementType)) {
    auto stringType = mlir::cast<krnl::StringType>(elementType);
    sizeInBits = stringType.getElementSize();
  } else {
    assert(mlir::isa<VectorType>(elementType) &&
           "elementType is not a VectorType");
    auto vectorType = mlir::cast<VectorType>(elementType);
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

/// Get the size of a static MemRef in bytes.
int64_t getMemRefSizeInBytes(Value value) {
  MemRefType memRefType = mlir::dyn_cast<MemRefType>(value.getType());
  auto memRefShape = memRefType.getShape();
  int64_t size = 1;
  for (unsigned int i = 0; i < memRefShape.size(); i++)
    size *= memRefShape[i];
  size *= getMemRefEltSizeInBytes(memRefType);
  return size;
}

/// Get the size of a MemRef.
/// If all the dimensions are static, emit a constant.
/// Otherwise, emit runtime computations.
Value getDynamicMemRefSize(PatternRewriter &rewriter, Location loc, Value val) {
  assert(mlir::isa<MemRefType>(val.getType()) &&
         "Value type should be a MemRefType");
  MemRefType memRefType = mlir::cast<MemRefType>(val.getType());
  auto shape = memRefType.getShape();
  // Accumulate static dimensions first.
  int64_t staticSizeInBytes = 1;
  bool allStaticDimensions = true;
  for (unsigned i = 0; i < shape.size(); i++) {
    if (shape[i] != ShapedType::kDynamic)
      staticSizeInBytes *= shape[i];
    else
      allStaticDimensions = false;
  }
  // Accumulate the remaining dimensions that are unknown.
  MultiDialectBuilder<MemRefBuilder, MathBuilder> create(rewriter, loc);
  Value sizeInBytes =
      create.math.constant(rewriter.getI64Type(), staticSizeInBytes);
  if (!allStaticDimensions) {
    for (unsigned i = 0; i < shape.size(); i++) {
      if (ShapedType::isDynamic(shape[i])) {
        Value index = create.mem.dim(val, i);
        Value dim = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI64Type(), index);
        sizeInBytes = create.math.mul(sizeInBytes, dim);
      }
    }
  }
  return sizeInBytes;
}

/// Get the size of a MemRef in bytes.
/// If all the dimensions are static, emit a constant.
/// Otherwise, emit runtime computations.
Value getDynamicMemRefSizeInBytes(
    PatternRewriter &rewriter, Location loc, Value val) {
  assert(mlir::isa<MemRefType>(val.getType()) &&
         "Value type should be a MemRefType");
  MemRefType memRefType = mlir::cast<MemRefType>(val.getType());
  auto shape = memRefType.getShape();
  // Accumulate static dimensions first.
  int64_t staticSizeInBytes = getMemRefEltSizeInBytes(memRefType);
  bool allStaticDimensions = true;
  for (unsigned i = 0; i < shape.size(); i++) {
    if (shape[i] != ShapedType::kDynamic)
      staticSizeInBytes *= shape[i];
    else
      allStaticDimensions = false;
  }
  // Accumulate the remaining dimensions that are unknown.
  MultiDialectBuilder<MemRefBuilder, MathBuilder> create(rewriter, loc);
  Value sizeInBytes =
      create.math.constant(rewriter.getI64Type(), staticSizeInBytes);
  if (!allStaticDimensions) {
    for (unsigned i = 0; i < shape.size(); i++) {
      if (ShapedType::isDynamic(shape[i])) {
        Value index = create.mem.dim(val, i);
        Value dim = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI64Type(), index);
        sizeInBytes = create.math.mul(sizeInBytes, dim);
      }
    }
  }
  return sizeInBytes;
}

/// Get the size of a dynamic MemRef in bytes.
Value getDynamicMemRefSizeInBytes(MemRefType type, Location loc,
    PatternRewriter &rewriter, memref::AllocOp allocOp) {
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);

  // Initialize the size variable with the size in bytes of the type.
  int64_t typeSize = getMemRefEltSizeInBytes(type);
  Value result = create.math.constant(rewriter.getIndexType(), typeSize);

  // Multiply all dimensions (constant and dynamic).
  auto memRefShape = type.getShape();
  auto rank = memRefShape.size();
  int dynDimIdx = 0;

  for (unsigned int idx = 0; idx < rank; ++idx) {
    if (memRefShape[idx] == ShapedType::kDynamic) {
      // Dynamic size.
      auto dynamicDim = allocOp.getOperands()[dynDimIdx];
      dynDimIdx++;
      result = create.math.mul(result, dynamicDim);
    } else {
      // Static size.
      auto staticDim =
          create.math.constant(rewriter.getIndexType(), memRefShape[idx]);
      result = create.math.mul(result, staticDim);
    }
  }

  return result;
}

/// Get the order number of the dynamic index passed as input.
/// Example for the following shape:
///   <1x2x?x3x?x4xf32>
///
/// getAllocArgIndex(<1x2x?x3x?x4xf32>, 2) will return 0.
/// getAllocArgIndex(<1x2x?x3x?x4xf32>, 4) will return 1.
///
int64_t getAllocArgIndex(memref::AllocOp allocOp, int64_t index) {
  auto memRefShape =
      mlir::dyn_cast<MemRefType>(allocOp.getResult().getType()).getShape();
  auto rank = memRefShape.size();

  int dynDimIdx = 0;
  for (unsigned int idx = 0; idx < rank; ++idx) {
    if (memRefShape[idx] == ShapedType::kDynamic) {
      if (idx == index)
        return dynDimIdx;
      dynDimIdx++;
    }
  }

  return -1;
}

/// Get alignment of an AllocOp if it exists else return zero.
int64_t getAllocAlignment(memref::AllocOp allocOp) {
  if (IntegerAttr alignmentAttr = allocOp.getAlignmentAttr())
    return alignmentAttr.getInt();

  return 0;
}

} // namespace onnx_mlir
