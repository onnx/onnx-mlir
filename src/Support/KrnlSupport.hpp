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

//===----------------------------------------------------------------------===//
// Return various operations.
//===----------------------------------------------------------------------===//

/// Get the AllocOp of the current GetRef.
memref::AllocOp getAllocOfGetRef(KrnlGetRefOp *getRef);

/// Return the top block.
Block *getTopBlock(Operation *op);

/// Retrieve function which contains the current operation.
FuncOp getContainingFunction(Operation *op);

// Emit a constant of a specific type.
// Use this function for small values only to avoid unexpected loss in type
// casting.
Value emitConstantOp(
    OpBuilder &rewriter, Location loc, Type type, double value);

//===----------------------------------------------------------------------===//
// Perform checks or get statistics about Krnl-level operations.
//===----------------------------------------------------------------------===//

/// Operation is a LoadOp or AffineLoadOp.
bool isLoad(Operation *op);

/// Operation is a StoreOp or AffineStoreOp.
bool isStore(Operation *op);

/// Operation is a KrnlMemcpyOp.
bool isKrnlMemcpy(Operation *op);

/// Checks if this operation loads/stores from the result of a specific getRef.
/// A krnl.memcpy acts as both load and store.
bool isLoadStoreForGetRef(KrnlGetRefOp getRef, Operation *op);

/// Check if this value is an argument of one of the blocks nested around it.
bool isBlockArgument(Operation *op, Value operand);

/// Check if two GetRefs participate in the same krnl.memcpy.
bool usedBySameKrnlMemcpy(
    KrnlGetRefOp *firstGetRef, KrnlGetRefOp *secondGetRef);

/// Check if two GetRefs participate in the same operation.
bool usedBySameOp(KrnlGetRefOp *firstGetRef, KrnlGetRefOp *secondGetRef);

/// Get the number of GetRef ops associated with this AllocOp.
int64_t getAllocGetRefNum(memref::AllocOp *allocOp);

/// Check if an operation is in the top-level block of the function.
bool opInTopLevelBlock(Operation *op);

/// This function returns true if `beforeOp` is visited before `op` in a
/// traversal of the provided block.
bool opBeforeOp(Block *block, Operation *beforeOp, Operation *afterOp);

/// Check Alloc operation result is used by a krnl.getref.
bool checkOpResultIsUsedByGetRef(memref::AllocOp *allocOp);

/// Check is all dimensions are known at compile time.
bool hasAllConstantDimensions(MemRefType memRefType);

/// Get the MemRef element size in bytes.
unsigned getMemRefEltSizeInBytes(MemRefType memRefType);

/// Get the size of a MemRef in bytes.
int64_t getMemRefSizeInBytes(Value value);

/// Get the size of a MemRef in bytes.
/// If all the dimensions are static, emit a constant.
/// Otherwise, emit runtime computations.
Value getDynamicMemRefSizeInBytes(
    PatternRewriter &rewriter, Location loc, Value val);

/// Get the size of a dynamic MemRef in bytes.
Value getDynamicMemRefSizeInBytes(MemRefType type, Location loc,
    PatternRewriter &rewriter, memref::AllocOp allocOp);

/// Get order number of dynamic index.
int64_t getAllocArgIndex(memref::AllocOp allocOp, int64_t index);

/// Get AllocOp alignment if it exists otherwise return zero.
int64_t getAllocAlignment(memref::AllocOp allocOp);

//===----------------------------------------------------------------------===//
// Live range analysis support.
//===----------------------------------------------------------------------===//

/// Returns the first operation in the live range of a getRef.
Operation *getLiveRangeFirstOp(KrnlGetRefOp getRef);

/// Returns the last operation in the live range of a getRef.
Operation *getLiveRangeLastOp(KrnlGetRefOp getRef);

/// Check if an operation is in an existing live range.
bool operationInLiveRange(
    Operation *operation, std::vector<Operation *> liveRangeOpList);

/// Function that returns the live range of a GetRef operation. The live
/// range consists of all the operations in the in-order traversal of the
/// source code between the first load/store instruction from that GetRef
/// and the last load/store instruction from that GetRef.
std::vector<Operation *> getLiveRange(KrnlGetRefOp getRef);

/// The live range is contained between firstOp and lastOp.
bool liveRangeIsContained(Operation *firstOp, Operation *lastOp,
    std::vector<Operation *> liveRangeOpList);
