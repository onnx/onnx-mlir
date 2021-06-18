//===- TmpMLIRUtils.cpp - Support migration to MLIR without EDSC --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code here is lifted from MLIR to facilitate the migration to EDSC-free MLIR.
// TODO: remove file once transition is completed.
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "TmpMlirUtils.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// from Utils.cpp
//===----------------------------------------------------------------------===//

Value ArithBuilder::_and(Value lhs, Value rhs) {
  return b.create<AndOp>(loc, lhs, rhs);
}
Value ArithBuilder::add(Value lhs, Value rhs) {
  if (lhs.getType().isa<IntegerType>())
    return b.create<AddIOp>(loc, lhs, rhs);
  return b.create<AddFOp>(loc, lhs, rhs);
}
Value ArithBuilder::mul(Value lhs, Value rhs) {
  if (lhs.getType().isa<IntegerType>())
    return b.create<MulIOp>(loc, lhs, rhs);
  return b.create<MulFOp>(loc, lhs, rhs);
}
Value ArithBuilder::sgt(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>())
    return b.create<CmpIOp>(loc, CmpIPredicate::sgt, lhs, rhs);
  return b.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
}
Value ArithBuilder::slt(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>())
    return b.create<CmpIOp>(loc, CmpIPredicate::slt, lhs, rhs);
  return b.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
}
Value ArithBuilder::select(Value cmp, Value lhs, Value rhs) {
  return b.create<SelectOp>(loc, cmp, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// from ImplicitLocObBuilder.cpp
//===----------------------------------------------------------------------===//

// none, all in the header file