/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- KrnlSupport.hpp - Krnl-level support functions -----------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains support code used at the level of the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_KRNL_SUPPORT_H
#define ONNX_MLIR_KRNL_SUPPORT_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"

#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Return various operations.
//===----------------------------------------------------------------------===//

/// Return the top block.
mlir::Block *getTopBlock(mlir::Operation *op);

/// Retrieve function which contains the current operation.
mlir::func::FuncOp getContainingFunction(mlir::Operation *op);

//===----------------------------------------------------------------------===//
// Perform checks or get statistics about Krnl-level operations.
//===----------------------------------------------------------------------===//

/// Operation is a LoadOp or AffineLoadOp.
bool isLoad(mlir::Operation *op);

/// Operation is a StoreOp or AffineStoreOp.
bool isStore(mlir::Operation *op);

/// Operation is a KrnlMemcpyOp.
bool isKrnlMemcpy(mlir::Operation *op);

/// Check if this value is an argument of one of the blocks nested around it.
bool isBlockArgument(mlir::Operation *op, mlir::Value operand);

/// Check if an operation is in the top-level block of the function.
bool opInTopLevelBlock(mlir::Operation *op);

/// This function returns true if `beforeOp` is visited before `op` in a
/// traversal of the provided block.
bool opBeforeOp(
    mlir::Block *block, mlir::Operation *beforeOp, mlir::Operation *afterOp);

/// Check is all dimensions are known at compile time.
bool hasAllConstantDimensions(mlir::MemRefType memRefType);

/// Get the MemRef element size in bytes.
unsigned getMemRefEltSizeInBytes(mlir::MemRefType memRefType);

/// Get the size of a MemRef in bytes.
int64_t getMemRefSizeInBytes(mlir::Value value);

/// Get the size of a MemRef.
/// If all the dimensions are static, emit a constant.
/// Otherwise, emit runtime computations.
mlir::Value getDynamicMemRefSize(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value val);

/// Get the size of a MemRef in bytes.
/// If all the dimensions are static, emit a constant.
/// Otherwise, emit runtime computations.
mlir::Value getDynamicMemRefSizeInBytes(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value val);

/// Get the size of a dynamic MemRef in bytes.
mlir::Value getDynamicMemRefSizeInBytes(mlir::MemRefType type,
    mlir::Location loc, mlir::PatternRewriter &rewriter,
    mlir::memref::AllocOp allocOp);

/// Get order number of dynamic index.
int64_t getAllocArgIndex(mlir::memref::AllocOp allocOp, int64_t index);

/// Get AllocOp alignment if it exists otherwise return zero.
int64_t getAllocAlignment(mlir::memref::AllocOp allocOp);

} // namespace onnx_mlir
#endif
