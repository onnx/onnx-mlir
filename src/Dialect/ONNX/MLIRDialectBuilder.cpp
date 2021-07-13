//===- MLIRDialectBuilder.cpp - Support migration to MLIR without EDSC --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MLIRDialectBuilder.hpp"
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
Value MathBuilder::select(Value cmp, Value lhs, Value rhs) {
  return b.create<SelectOp>(loc, cmp, lhs, rhs);
}
