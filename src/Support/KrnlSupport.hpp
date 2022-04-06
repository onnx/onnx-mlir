/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- KrnlSupport.hpp - Krnl-level support functions -----------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains support code used at the level of the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

/// Get the AllocOp of the current GetRef.
mlir::memref::AllocOp getAllocOfGetRef(mlir::KrnlGetRefOp *getRef);

/// Return the top block.
mlir::Block *getTopBlock(mlir::Operation *op);

/// Retrieve function which contains the current operation.
mlir::FuncOp getContainingFunction(mlir::Operation *op);

// Emit a constant of a specific type.
// Use this function for small values only to avoid unexpected loss in type
// casting.
mlir::Value emitConstantOp(mlir::OpBuilder &rewriter, mlir::Location loc,
    mlir::Type type, double value);

//===----------------------------------------------------------------------===//
// Perform checks or get statistics about Krnl-level operations.
//===----------------------------------------------------------------------===//

/// Operation is a LoadOp or AffineLoadOp.
bool isLoad(mlir::Operation *op);

/// Operation is a StoreOp or AffineStoreOp.
bool isStore(mlir::Operation *op);

/// Operation is a KrnlMemcpyOp.
bool isKrnlMemcpy(mlir::Operation *op);

/// Checks if this operation loads/stores from the result of a specific getRef.
/// A krnl.memcpy acts as both load and store.
bool isLoadStoreForGetRef(mlir::KrnlGetRefOp getRef, mlir::Operation *op);

/// Check if this value is an argument of one of the blocks nested around it.
bool isBlockArgument(mlir::Operation *op, mlir::Value operand);

/// Check if two GetRefs participate in the same krnl.memcpy.
bool usedBySameKrnlMemcpy(
    mlir::KrnlGetRefOp *firstGetRef, mlir::KrnlGetRefOp *secondGetRef);

/// Check if two GetRefs participate in the same operation.
bool usedBySameOp(
    mlir::KrnlGetRefOp *firstGetRef, mlir::KrnlGetRefOp *secondGetRef);

/// Get the number of GetRef ops associated with this AllocOp.
int64_t getAllocGetRefNum(mlir::memref::AllocOp *allocOp);

/// Check if an operation is in the top-level block of the function.
bool opInTopLevelBlock(mlir::Operation *op);

/// This function returns true if `beforeOp` is visited before `op` in a
/// traversal of the provided block.
bool opBeforeOp(
    mlir::Block *block, mlir::Operation *beforeOp, mlir::Operation *afterOp);

/// Check Alloc operation result is used by a krnl.getref.
bool checkOpResultIsUsedByGetRef(mlir::memref::AllocOp *allocOp);

/// Check is all dimensions are known at compile time.
bool hasAllConstantDimensions(mlir::MemRefType memRefType);

/// Get the MemRef element size in bytes.
unsigned getMemRefEltSizeInBytes(mlir::MemRefType memRefType);

/// Get the size of a MemRef in bytes.
int64_t getMemRefSizeInBytes(mlir::Value value);

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

//===----------------------------------------------------------------------===//
// Live range analysis support.
//===----------------------------------------------------------------------===//

/// Returns the first operation in the live range of a getRef.
mlir::Operation *getLiveRangeFirstOp(mlir::KrnlGetRefOp getRef);

/// Returns the last operation in the live range of a getRef.
mlir::Operation *getLiveRangeLastOp(mlir::KrnlGetRefOp getRef);

/// Check if an operation is in an existing live range.
bool operationInLiveRange(
    mlir::Operation *operation, std::vector<mlir::Operation *> liveRangeOpList);

/// Function that returns the live range of a GetRef operation. The live
/// range consists of all the operations in the in-order traversal of the
/// source code between the first load/store instruction from that GetRef
/// and the last load/store instruction from that GetRef.
std::vector<mlir::Operation *> getLiveRange(mlir::KrnlGetRefOp getRef);

/// The live range is contained between firstOp and lastOp.
bool liveRangeIsContained(mlir::Operation *firstOp, mlir::Operation *lastOp,
    std::vector<mlir::Operation *> liveRangeOpList);

} // namespace onnx_mlir
