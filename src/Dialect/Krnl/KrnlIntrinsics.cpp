/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------------- KrnlHelper.cpp - Krnl Dialect Helper----------------===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// This file declares intrinsic methods to build Krnl Dialect Ops.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/Krnl/KrnlIntrinsics.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;
// using namespace mlir::edsc;
// using namespace mlir::edsc::intrinsics;
namespace mlir {
namespace edsc {
namespace intrinsics {

//====---------------- EDSC Support with Value ---------------------------===//

Value krnl_load(Value memref, ArrayRef<Value> indices) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return ScopedContext::getBuilderRef().create<KrnlLoadOp>(
      ScopedContext::getLocation(), memref, indices);
}

void krnl_store(Value val, Value memref, ArrayRef<Value> indices) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  ScopedContext::getBuilderRef().create<KrnlStoreOp>(
      ScopedContext::getLocation(), val, memref, indices);
}

ValueRange krnl_define_loop(int64_t originalLoopNum) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  KrnlDefineLoopsOp newOp =
      ScopedContext::getBuilderRef().create<KrnlDefineLoopsOp>(
          ScopedContext::getLocation(), originalLoopNum);
  return newOp.getResults();
}

ValueRange krnl_block(Value loop, int64_t blockSize) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return ScopedContext::getBuilderRef()
      .create<KrnlBlockOp>(ScopedContext::getLocation(), loop, blockSize)
      .getResults();
}

void krnl_permute(ArrayRef<Value> loops, ArrayRef<int64_t> map) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  ScopedContext::getBuilderRef().create<KrnlPermuteOp>(
      ScopedContext::getLocation(), loops, map);
}

ValueRange krnl_get_induction_var_value(ArrayRef<Value> loops) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return ScopedContext::getBuilderRef()
      .create<KrnlGetInductionVariableValueOp>(
          ScopedContext::getLocation(), loops)
      .getResults();
}

void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<Value> optimizedLoop,
    ArrayRef<Value> lb, ArrayRef<Value> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value>)> bodyBuilderFn) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(lb.size() == ub.size() && "expected matching number of lb & ub");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();
  KrnlIterateOperandPack pack(builder, originalLoop, optimizedLoop);
  for (int i = 0; i < lb.size(); ++i) {
    pack.pushOperandBound(lb[i]);
    pack.pushOperandBound(ub[i]);
  }
  KrnlIterateOp iterateOp =
      builder.create<KrnlIterateOp>(ScopedContext::getLocation(), pack);
  // auto savedInsertionPoint = builder.saveInsertionPoint();
  Block *iterBlock = &iterateOp.bodyRegion().front();

  if (bodyBuilderFn) { // Scope for the scoped context of the loop.
    ScopedContext nestedContext(builder, loc);
    builder.setInsertionPointToStart(iterBlock);
    bodyBuilderFn(iterArgs);
  }
}

void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<Value> lb,
    ArrayRef<Value> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value>)> bodyBuilderFn) {
  // When no optimized loops are given, use original for the optimized.
  krnl_iterate(originalLoop, originalLoop, lb, ub, iterArgs, bodyBuilderFn);
}

void krnl_copy_to_buffer(Value bufferMemref, Value memref,
    ArrayRef<Value> starts, Value padValue, ArrayRef<int64_t> tileSize,
    ArrayRef<int64_t> padToNext) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  ScopedContext::getBuilderRef().create<KrnlCopyToBufferOp>(
      ScopedContext::getLocation(), bufferMemref, memref, starts, padValue,
      tileSize, padToNext);
}

void krnl_copy_to_buffer(
    Value bufferMemref, Value memref, ArrayRef<Value> starts, Value padValue) {
  ArrayRef<int64_t> empty;
  krnl_copy_to_buffer(bufferMemref, memref, starts, padValue, empty, empty);
}

void krnl_copy_from_buffer(Value bufferMemref, Value memref,
    ArrayRef<Value> starts, ArrayRef<int64_t> tileSize) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  ScopedContext::getBuilderRef().create<KrnlCopyFromBufferOp>(
      ScopedContext::getLocation(), bufferMemref, memref, starts, tileSize);
}
void krnl_copy_from_buffer(
    Value bufferMemref, Value memref, ArrayRef<Value> starts) {
  ArrayRef<int64_t> empty;
  krnl_copy_from_buffer(bufferMemref, memref, starts, empty);
}

void krnl_matmul(Value A, ArrayRef<Value> aStart, Value B,
    ArrayRef<Value> bStart, Value C, ArrayRef<Value> cStart,
    ArrayRef<Value> loops, ArrayRef<Value> computeStarts,
    ArrayRef<Value> globalUBs, ArrayRef<int64_t> computeTileSize,
    ArrayRef<int64_t> aTileSize, ArrayRef<int64_t> bTileSize,
    ArrayRef<int64_t> cTileSize, bool simdize, bool unroll, bool overcompute) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(aStart.size() == 2 && "A start needs 2 dim");
  assert(bStart.size() == 2 && "B start needs 2 dim");
  assert(cStart.size() == 2 && "C start needs 2 dim");
  assert(loops.size() == 3 && "loops needs 3 dim");
  assert(computeStarts.size() == 3 && "compute starts needs 3 dim");
  assert(globalUBs.size() == 3 && "global UBs needs 3 dim");
  ScopedContext::getBuilderRef().create<KrnlMatMulOp>(
      ScopedContext::getLocation(), A, aStart[0], aStart[1], B, bStart[0],
      bStart[1], C, cStart[0], cStart[1], loops, computeStarts[0],
      computeStarts[1], computeStarts[2], globalUBs[0], globalUBs[1],
      globalUBs[2], computeTileSize, aTileSize, bTileSize, cTileSize, simdize,
      unroll, overcompute);
}

void krnl_matmul(Value A, ArrayRef<Value> aStart, Value B,
    ArrayRef<Value> bStart, Value C, ArrayRef<Value> cStart,
    ArrayRef<Value> loops, ArrayRef<Value> computeStarts,
    ArrayRef<Value> globalUBs, bool simdize, bool unroll, bool overcompute) {
  ArrayRef<int64_t> empty;
  krnl_matmul(A, aStart, B, bStart, C, cStart, loops, computeStarts, globalUBs,
      empty, empty, empty, empty, simdize, unroll, overcompute);
}

//====---------------- EDSC Support with IndexExpr -----------------------===//

Value krnl_load(Value memref, ArrayRef<IndexExpr> indices) {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  return krnl_load(memref, indexValues);
}

void krnl_store(Value val, Value memref, ArrayRef<IndexExpr> indices) {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  krnl_store(val, memref, indexValues);
}

void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<Value> optimizedLoop,
    ArrayRef<IndexExpr> lb, ArrayRef<IndexExpr> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value>)> bodyBuilderFn) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(lb.size() == ub.size() && "expected matching number of lb & ub");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();
  KrnlIterateOperandPack pack(builder, originalLoop, optimizedLoop);
  for (int i = 0; i < lb.size(); ++i) {
    pack.pushIndexExprBound(lb[i]);
    pack.pushIndexExprBound(ub[i]);
  }
  KrnlIterateOp iterateOp =
      builder.create<KrnlIterateOp>(ScopedContext::getLocation(), pack);
  // auto savedInsertionPoint = builder.saveInsertionPoint();
  Block *iterBlock = &iterateOp.bodyRegion().front();

  if (bodyBuilderFn) { // Scope for the scoped context of the loop.
    ScopedContext nestedContext(builder, loc);
    builder.setInsertionPointToStart(iterBlock);
    bodyBuilderFn(iterArgs);
  }
}

void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<IndexExpr> lb,
    ArrayRef<IndexExpr> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value>)> bodyBuilderFn) {
  // When no optimized loops are given, use original for the optimized.
  krnl_iterate(originalLoop, originalLoop, lb, ub, iterArgs, bodyBuilderFn);
}

void krnl_copy_to_buffer(Value bufferMemref, Value memref,
    ArrayRef<IndexExpr> starts, Value padValue, ArrayRef<int64_t> tileSize,
    ArrayRef<int64_t> padToNext) {
  SmallVector<Value, 4> startValues;
  IndexExpr::getValues(starts, startValues);
  krnl_copy_to_buffer(
      bufferMemref, memref, startValues, padValue, tileSize, padToNext);
}

void krnl_copy_to_buffer(Value bufferMemref, Value memref,
    ArrayRef<IndexExpr> starts, Value padValue) {
  ArrayRef<int64_t> empty;
  krnl_copy_to_buffer(bufferMemref, memref, starts, padValue, empty, empty);
}

void krnl_copy_from_buffer(Value bufferMemref, Value memref,
    ArrayRef<IndexExpr> starts, ArrayRef<int64_t> tileSize) {
  SmallVector<Value, 4> startValues;
  IndexExpr::getValues(starts, startValues);
  krnl_copy_from_buffer(bufferMemref, memref, starts, tileSize);
}

void krnl_copy_from_buffer(
    Value bufferMemref, Value memref, ArrayRef<IndexExpr> starts) {
  ArrayRef<int64_t> empty;
  krnl_copy_from_buffer(bufferMemref, memref, starts, empty);
}
} // namespace intrinsics
} // namespace edsc
} // namespace mlir
