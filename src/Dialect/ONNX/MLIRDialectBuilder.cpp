//===---- MLIRDialectBuilder.cpp - Helper functions for MLIR dialects -----===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for building MLIR operations.
//
//===----------------------------------------------------------------------===//

#include "MLIRDialectBuilder.hpp"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Original code for MathBuilder is copied from LLVM MLIR Utils.cpp
// Modified here to add operations, add super class.
// Liscense added here for this class for completness.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

Value MathBuilder::_and(Value lhs, Value rhs) {
  return b.create<AndOp>(loc, lhs, rhs);
}

Value MathBuilder::_or(Value lhs, Value rhs) {
  return b.create<OrOp>(loc, lhs, rhs);
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

Value MathBuilder::exp2(Value val) {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return b.create<math::Exp2Op>(loc, val);
}

Value MathBuilder::log2(Value val) {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return b.create<math::Log2Op>(loc, val);
}

Value MathBuilder::min(Value lhs, Value rhs) {
  // TODO: use MinSIOp, MinUIOp, MinFOp when MLIR is updated.
  assert(
      lhs.getType() == rhs.getType() && "Two operands must have the same type");
  Value gt;
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    gt = b.create<CmpIOp>(loc, CmpIPredicate::sgt, lhs, rhs);
  else
    gt = b.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
  return b.create<SelectOp>(loc, gt, rhs, lhs);
}

Value MathBuilder::max(Value lhs, Value rhs) {
  // TODO: use MaxSIOp, MaxUIOp, MaxFOp when MLIR is updated.
  assert(
      lhs.getType() == rhs.getType() && "Two operands must have the same type");
  Value gt;
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    gt = b.create<CmpIOp>(loc, CmpIPredicate::sgt, lhs, rhs);
  else
    gt = b.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
  return b.create<SelectOp>(loc, gt, lhs, rhs);
}

Value MathBuilder::sgt(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>() ||
      lhs.getType().isa<IndexType>())
    return b.create<CmpIOp>(loc, CmpIPredicate::sgt, lhs, rhs);
  return b.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
}

Value MathBuilder::sge(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>() ||
      lhs.getType().isa<IndexType>())
    return b.create<CmpIOp>(loc, CmpIPredicate::sge, lhs, rhs);
  return b.create<CmpFOp>(loc, CmpFPredicate::OGE, lhs, rhs);
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

Value MathBuilder::constant(Type type, double val) {
  Attribute constantAttr;
  TypeSwitch<Type>(type)
      .Case<Float16Type>(
          [&](Type) { constantAttr = b.getF16FloatAttr((float)val); })
      .Case<Float32Type>(
          [&](Type) { constantAttr = b.getF32FloatAttr((float)val); })
      .Case<Float64Type>(
          [&](Type) { constantAttr = b.getF64FloatAttr((float)val); })
      .Case<IntegerType>([&](Type) {
        assert(val == (int64_t)val && "value is ambiguous");
        auto width = type.cast<IntegerType>().getWidth();
        if (width == 1) {
          constantAttr = b.getBoolAttr(val != 0);
        } else {
          constantAttr = b.getIntegerAttr(type, APInt(width, (int64_t)val));
        }
      })
      .Case<IndexType>(
          [&](Type) { constantAttr = b.getIntegerAttr(type, (int64_t)val); })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });
  return b.create<ConstantOp>(loc, constantAttr);
}

Value MathBuilder::constantIndex(int64_t val) {
  Attribute constantAttr = b.getIntegerAttr(b.getIndexType(), val);
  return b.create<ConstantOp>(loc, constantAttr);
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
