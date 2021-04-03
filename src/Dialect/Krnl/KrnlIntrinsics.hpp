/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------------- KrnlHelper.hpp - Krnl Dialect Helper----------------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// This file implements intrinsics methods to build Krnl Dialect ops.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <queue>

#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

//====---------------- EDSC Support with Value ---------------------------===//

Value krnl_load(Value memref, ArrayRef<Value> indices);
void krnl_store(Value val, Value memref, ArrayRef<Value> indices);

ValueRange krnl_define_loop(int64_t originalLoopNum);
ValueRange krnl_block(Value loop, int64_t blockSize);
void krnl_permute(ArrayRef<Value> loops, ArrayRef<int64_t> map);
ValueRange krnl_get_induction_var_value(ArrayRef<Value> loops);

void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<Value> optimizedLoop,
    ArrayRef<Value> lb, ArrayRef<Value> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value> args)> bodyBuilderFn);
void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<Value> lb,
    ArrayRef<Value> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value> args)> bodyBuilderFn);

void krnl_copy_to_buffer(Value bufferMemref, Value memref,
    ArrayRef<Value> starts, Value padValue, ArrayRef<int64_t> tileSize,
    ArrayRef<int64_t> padToNext);
void krnl_copy_to_buffer(
    Value bufferMemref, Value memref, ArrayRef<Value> starts, Value padValue);

void krnl_copy_from_buffer(Value bufferMemref, Value memref,
    ArrayRef<Value> starts, ArrayRef<int64_t> tileSize);
void krnl_copy_from_buffer(
    Value bufferMemref, Value memref, ArrayRef<Value> starts);

void krnl_matmul(Value A, ArrayRef<Value> aStart, Value B,
    ArrayRef<Value> bStart, Value C, ArrayRef<Value> cStart,
    ArrayRef<Value> loops, ArrayRef<Value> computeStarts,
    ArrayRef<Value> globalUBs, ArrayRef<int64_t> computeTileSize,
    ArrayRef<int64_t> aTileSize, ArrayRef<int64_t> bTileSize,
    ArrayRef<int64_t> cTileSize, bool simdize, bool unroll, bool overcompute);

void krnl_matmul(Value A, ArrayRef<Value> aStart, Value B,
    ArrayRef<Value> bStart, Value C, ArrayRef<Value> cStart,
    ArrayRef<Value> loops, ArrayRef<Value> computeStarts,
    ArrayRef<Value> globalUBs, bool simdize, bool unroll, bool overcompute);

//====---------------- EDSC Support with IndexExpr -----------------------===//

Value krnl_load(Value memref, ArrayRef<IndexExpr> indices);
void krnl_store(Value val, Value memref, ArrayRef<IndexExpr> indices);

void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<Value> optimizedLoop,
    ArrayRef<IndexExpr> lb, ArrayRef<IndexExpr> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value> args)> bodyBuilderFn);
void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<IndexExpr> lb,
    ArrayRef<IndexExpr> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value> args)> bodyBuilderFn);

void krnl_copy_to_buffer(Value bufferMemref, Value memref,
    ArrayRef<IndexExpr> starts, Value padValue, ArrayRef<int64_t> tileSize,
    ArrayRef<int64_t> padToNext);
void krnl_copy_to_buffer(Value bufferMemref, Value memref,
    ArrayRef<IndexExpr> starts, Value padValue);

void krnl_copy_from_buffer(Value bufferMemref, Value memref,
    ArrayRef<IndexExpr> starts, ArrayRef<int64_t> tileSize);
void krnl_copy_from_buffer(
    Value bufferMemref, Value memref, ArrayRef<Value> starts);

} // namespace intrinsics
} // namespace edsc
} // namespace mlir
