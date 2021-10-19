//===- MLIRDialectBuilder.cpp - Support migration to MLIR without EDSC --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MLIRDialectBuilder.hpp"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// from MLIR Utils.cpp, modified to handle index values too.
//===----------------------------------------------------------------------===//

Value MathBuilder::_and(Value lhs, Value rhs) {
  return b.create<AndOp>(loc, lhs, rhs);
}
Value MathBuilder::add(Value lhs, Value rhs) {
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<AddIOp>(loc, lhs, rhs);
  return b.create<AddFOp>(loc, lhs, rhs);
}
Value MathBuilder::sub(Value lhs, Value rhs) {
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<SubIOp>(loc, lhs, rhs);
  return b.create<SubFOp>(loc, lhs, rhs);
}
Value MathBuilder::mul(Value lhs, Value rhs) {
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<MulIOp>(loc, lhs, rhs);
  return b.create<MulFOp>(loc, lhs, rhs);
}

Value MathBuilder::div(Value lhs, Value rhs) {
  if (lhs.getType().isa<FloatType>() && rhs.getType().isa<FloatType>())
    return b.create<DivFOp>(loc, lhs, rhs);
  else
    llvm_unreachable("Only support float type at this moment.");
}

Value MathBuilder::exp(Value val) {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return b.create<math::ExpOp>(loc, val);
}

Value MathBuilder::sgt(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>() ||
      lhs.getType().isa<IndexType>())
    return b.create<CmpIOp>(loc, CmpIPredicate::sgt, lhs, rhs);
  return b.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
}
Value MathBuilder::slt(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>() ||
      lhs.getType().isa<IndexType>())
    return b.create<CmpIOp>(loc, CmpIPredicate::slt, lhs, rhs);
  return b.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
}
Value MathBuilder::eq(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>() ||
      lhs.getType().isa<IndexType>())
    return b.create<CmpIOp>(loc, CmpIPredicate::eq, lhs, rhs);
  return b.create<CmpFOp>(loc, CmpFPredicate::OEQ, lhs, rhs);
}
Value MathBuilder::select(Value cmp, Value lhs, Value rhs) {
  return b.create<SelectOp>(loc, cmp, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Memref support, including inserting default alignment.
//===----------------------------------------------------------------------===//

memref::AllocOp MemRefBuilder::alloc(MemRefType type, ValueRange dynSymbols) {
  return b.create<memref::AllocOp>(loc, type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(MemRefType type) {
  return b.create<memref::AllocOp>(loc, type);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, int64_t alignment) {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  IntegerAttr alignmentAttr = b.getI64IntegerAttr(alignment);
  if (type.getShape().size() == 0) // Drop align for scalars.
    return b.create<memref::AllocOp>(loc, type);
  return b.create<memref::AllocOp>(loc, type, alignmentAttr);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, ValueRange dynSymbols, int64_t alignment) {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  IntegerAttr alignmentAttr = b.getI64IntegerAttr(alignment);
  if (type.getShape().size() == 0) // Drop align for scalars.
    return b.create<memref::AllocOp>(loc, type, dynSymbols);
  return b.create<memref::AllocOp>(loc, type, dynSymbols, alignmentAttr);
}

memref::AllocaOp MemRefBuilder::alloca(MemRefType type) {
  return b.create<memref::AllocaOp>(loc, type);
}

memref::AllocaOp MemRefBuilder::alignedAlloca(
    MemRefType type, int64_t alignment) {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  IntegerAttr alignmentAttr = b.getI64IntegerAttr(alignment);
  if (type.getShape().size() == 0) // Drop align for scalars.
    return b.create<memref::AllocaOp>(loc, type);
  return b.create<memref::AllocaOp>(loc, type, alignmentAttr);
}

memref::DeallocOp MemRefBuilder::dealloc(Value val) {
  return b.create<memref::DeallocOp>(loc, val);
}

Value MemRefBuilder::dim(Value val, int64_t index) {
  Value i = b.create<ConstantIndexOp>(loc, index);
  return b.createOrFold<memref::DimOp>(loc, val, i);
}

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF).
//===----------------------------------------------------------------------===//

void SCFBuilder::ifThenElse(Value cond,
    function_ref<void(SCFBuilder &createSCF)> thenFn,
    function_ref<void(SCFBuilder &createSCF)> elseFn) {
  if (!elseFn)
    b.create<scf::IfOp>(
        loc, /*resultTypes=*/llvm::None, cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          thenFn(scfBuilder);
        },
        /*else*/
        llvm::None);
  else
    b.create<scf::IfOp>(
        loc, /*resultTypes=*/llvm::None, cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          thenFn(scfBuilder);
        },
        /*else*/
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          elseFn(scfBuilder);
        });
}
