/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ DialectBuilder.cpp - Helper functions for MLIR dialects -------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for building MLIR operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

// Please do not add dependences on ONNX or KRNL dialects.
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"

#include <algorithm>

#define DEBUG_TYPE "dialect-builder"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Original code for MathBuilder is copied from LLVM MLIR Utils.cpp
// Modified here to add operations, add super class.
// License added here for this class for completeness.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

// Test for unsigned as signless are treated as signed. For reference, check in
// MLIR AffineToStandard where comparison of indices are done with slt and sgt,
// for example. Indices are signless. Also, in ONNX, we currently treat all
// ONNX Integers as MLIR signless, and only flag the ONNX Unsigned Integer as
// MLIR unsigned integer.

/* static */ bool MathBuilder::isVector(Value val) {
  return isVector(val.getType());
}

/* static */ bool MathBuilder::isVector(Type type) {
  return mlir::dyn_cast<VectorType>(type) != nullptr;
}

/* static */ Type MathBuilder::elementTypeOfScalarOrVector(Value val) {
  return elementTypeOfScalarOrVector(val.getType());
}

/* static */ Type MathBuilder::elementTypeOfScalarOrVector(
    Type elementOrVectorType) {
  VectorType vectorType = mlir::dyn_cast<VectorType>(elementOrVectorType);
  if (vectorType)
    return vectorType.getElementType();
  return elementOrVectorType;
}

// return a vector of "elementType" with the same vector shape as "vectorType"
/* static */ Type MathBuilder::getTypeWithVector(
    Type vectorType, Type elementType) {
  assert(!isVector(elementType) && "element type expected to be a scalar");
  // When vectorType is not a vector, then we need to return a scalar of the
  // type elementType.
  if (!isVector(vectorType))
    return elementType;
  // When vectorType is actually a vector, then replicate the shape of
  // vectorType with the element type of elementType.
  return VectorType::get(
      mlir::cast<VectorType>(vectorType).getShape(), elementType);
}

/* static */ bool MathBuilder::isScalarOrVectorInteger(Value val) {
  return isScalarOrVectorInteger(val.getType());
}

/* static */ bool MathBuilder::isScalarOrVectorInteger(
    Type elementOrVectorType) {
  Type elementType = elementTypeOfScalarOrVector(elementOrVectorType);
  return mlir::isa<IntegerType>(elementType) ||
         mlir::isa<IndexType>(elementType);
}

/* static */ bool MathBuilder::isScalarOrVectorUnsignedInteger(Value val) {
  return isScalarOrVectorUnsignedInteger(val.getType());
}

/* static */ bool MathBuilder::isScalarOrVectorUnsignedInteger(
    Type elementOrVectorType) {
  Type elementType = elementTypeOfScalarOrVector(elementOrVectorType);
  return elementType.isUnsignedInteger();
}

/* static */ bool MathBuilder::isScalarOrVectorFloat(Value val) {
  return isScalarOrVectorFloat(val.getType());
}

/* static */ bool MathBuilder::isScalarOrVectorFloat(Type elementOrVectorType) {
  Type elementType = elementTypeOfScalarOrVector(elementOrVectorType);
  return mlir::isa<FloatType>(elementType);
}

bool MathBuilder::splatToMatch(Value &first, Value &second) const {
  Type firstType = first.getType();
  Type secondType = second.getType();
  VectorType firstVectorType = mlir::dyn_cast<VectorType>(firstType);
  VectorType secondVectorType = mlir::dyn_cast<VectorType>(secondType);
  MultiDialectBuilder<VectorBuilder> create(*this);
  LLVM_DEBUG(llvm::dbgs() << "Splat to match first: " << firstType << "\n";
             llvm::dbgs() << "  second: " << secondType << "\n";);

  // Splat first if needed.
  if (!firstVectorType && secondVectorType) {
    firstVectorType = VectorType::get(secondVectorType.getShape(), firstType);
    first = create.vec.broadcast(firstVectorType, first);
    LLVM_DEBUG(llvm::dbgs() << "  splat first\n");
    return true;
  }
  // Splat second if needed.
  if (firstVectorType && !secondVectorType) {
    secondVectorType = VectorType::get(firstVectorType.getShape(), secondType);
    second = create.vec.broadcast(secondVectorType, second);
    LLVM_DEBUG(llvm::dbgs() << "  splat second\n");
    return true;
  }
  // Otherwise check compatibility.
  assert(create.vec.compatibleShapes(firstType, secondType) &&
         "expected compatible shapes");
  return false;
}

bool MathBuilder::splatToMatch(
    Value &first, Value &second, Value &third) const {
  bool changeIn12 = splatToMatch(first, second);
  bool changeIn13 = splatToMatch(first, third);
  if (!changeIn12 && changeIn13)
    // Have missed changes in 1-2 pair, redo.
    splatToMatch(first, second);
  return changeIn12 || changeIn13;
}

void MathBuilder::splatToMatch(llvm::SmallVectorImpl<Value> &vals) const {
  // Do not check the types when matching splats as this interface is called
  // blindly on a list of vals.
  int64_t size = vals.size();
  if (size <= 1)
    return; // Nothing to do with 0 or 1 values.
  if (size == 2) {
    splatToMatch(vals[0], vals[1]);
  } else if (size == 3) {
    splatToMatch(vals[0], vals[1], vals[2]);
  } else {
    llvm_unreachable("can only splat to match up to 3 values");
  }
}

Value MathBuilder::abs(Value val) const {
  if (isScalarOrVectorInteger(val))
    return math::AbsIOp::create(b(), loc(), val);
  if (isScalarOrVectorFloat(val))
    return math::AbsFOp::create(b(), loc(), val);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::andi(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs))
    return arith::AndIOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::ori(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs))
    return arith::OrIOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::xori(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs))
    return arith::XOrIOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::add(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs)) {
    Type elemType = elementTypeOfScalarOrVector(lhs);
    if (elemType.isUnsignedInteger()) {
      unsigned elemWidth = mlir::cast<IntegerType>(elemType).getWidth();
      Value castLhs = castToSignless(lhs, elemWidth);
      Value castRhs = castToSignless(rhs, elemWidth);
      Value castAdd =
          arith::AddUIExtendedOp::create(b(), loc(), castLhs, castRhs).getSum();
      return castToUnsigned(castAdd, elemWidth);
    } else
      return arith::AddIOp::create(b(), loc(), lhs, rhs);
  }
  if (isScalarOrVectorFloat(lhs))
    return arith::AddFOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sub(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs))
    return arith::SubIOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorFloat(lhs))
    return arith::SubFOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::mul(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs)) {
    Type elemType = elementTypeOfScalarOrVector(lhs);
    if (elemType.isUnsignedInteger()) {
      unsigned elemWidth = mlir::cast<IntegerType>(elemType).getWidth();
      Value castLhs = castToSignless(lhs, elemWidth);
      Value castRhs = castToSignless(rhs, elemWidth);
      Value castMul =
          arith::MulUIExtendedOp::create(b(), loc(), castLhs, castRhs).getLow();
      return castToUnsigned(castMul, elemWidth);
    } else
      return arith::MulIOp::create(b(), loc(), lhs, rhs);
  }
  if (isScalarOrVectorFloat(lhs))
    return arith::MulFOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::div(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorFloat(lhs))
    return arith::DivFOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorUnsignedInteger(lhs))
    return arith::DivUIOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorInteger(lhs))
    return arith::DivSIOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::rem(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorFloat(lhs))
    return arith::RemFOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorUnsignedInteger(lhs))
    return arith::RemUIOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorInteger(lhs))
    return arith::RemSIOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::round(Value x) const {
  Type type = x.getType();
  assert(isScalarOrVectorFloat(type) && "expected float");
  return math::RoundOp::create(b(), loc(), x);
}

Value MathBuilder::roundEven(Value x) const {
  Type type = x.getType();
  assert(isScalarOrVectorFloat(type) && "expected float");
  return math::RoundEvenOp::create(b(), loc(), x);
}

Value MathBuilder::roundEvenEmulation(Value x) const {
  Type type = x.getType();
  assert(isScalarOrVectorFloat(type) && "expected float");

  // Use algorithm originally posted in ONNXtoKRNL/Math/Elementwise.cpp
  // lowering.

  // Use numpy algorithm for rint as follows.
  // ```
  // double y, r;
  // y = npy_floor(x);
  // r = x - y;
  //
  // if (r > 0.5) {
  //     y += 1.0;
  // }
  //
  // /* Round to nearest even */
  // if (r == 0.5) {
  //     r = y - 2.0*npy_floor(0.5*y);
  //     if (r == 1.0) {
  //         y += 1.0;
  //     }
  // }
  // return y;
  // ```
  Value one = constant(type, 1.0);
  Value two = constant(type, 2.0);
  Value half = constant(type, 0.5);
  Value y = floor(x);
  Value r = sub(x, y);
  // r > 0.5
  Value rGreaterThanHalf = sgt(r, half);
  Value y1 = select(rGreaterThanHalf, add(y, one), y);
  // r == 0.5: round to nearest even.
  Value y2 = mul(half, y);
  y2 = floor(y2);
  y2 = mul(y2, two);
  Value rr = sub(y, y2);
  Value rrEqualOne = eq(rr, one);
  y2 = select(rrEqualOne, add(y, one), y);

  Value rEqualHalf = eq(r, half);
  return select(rEqualHalf, y2, y1);
}

Value MathBuilder::copySign(Value rem, Value dividend) const {
  splatToMatch(rem, dividend);
  assert(rem.getType() == dividend.getType() && "expected same type");
  if (isScalarOrVectorFloat(rem))
    return math::CopySignOp::create(b(), loc(), rem, dividend);
  llvm_unreachable("expected float");
}

Value MathBuilder::ceilDiv(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorUnsignedInteger(lhs))
    return arith::CeilDivUIOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorInteger(lhs))
    return arith::CeilDivSIOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::clip(Value val, Value lb, Value ub) const {
  // Don't perform type assert and/or splats as it will be done in the min/max
  // operations.
  val = max(val, lb);  // Clip lower range.
  return min(val, ub); // Clip upper range.
}

Value MathBuilder::cos(Value val) const {
  if (isScalarOrVectorFloat(val))
    return math::CosOp::create(b(), loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::floorDiv(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorUnsignedInteger(lhs))
    // Using regular unsigned div is ok as it rounds toward zero.
    return arith::DivUIOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorInteger(lhs))
    return arith::FloorDivSIOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int");
}

// return (lhs * rhs) + acc
Value MathBuilder::fma(Value lhs, Value rhs, Value acc) const {
  splatToMatch(lhs, rhs, acc);
  assert((lhs.getType() == rhs.getType()) && (rhs.getType() == acc.getType()) &&
         "expected same type");
  if (isScalarOrVectorFloat(lhs) && isVector(lhs)) {
    return vector::FMAOp::create(b(), loc(), lhs, rhs, acc);
  }
  return add(mul(lhs, rhs), acc); // Handle broadcast there.
}

Value MathBuilder::erf(Value val) const {
  return math::ErfOp::create(b(), loc(), val);
}

Value MathBuilder::exp(Value val) const {
  if (isScalarOrVectorFloat(val))
    return math::ExpOp::create(b(), loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::exp2(Value val) const {
  if (isScalarOrVectorFloat(val))
    return math::Exp2Op::create(b(), loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::log(Value val) const {
  if (isScalarOrVectorFloat(val))
    return math::LogOp::create(b(), loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::log2(Value val) const {
  if (isScalarOrVectorFloat(val))
    return math::Log2Op::create(b(), loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::sqrt(Value val) const {
  if (isScalarOrVectorFloat(val))
    return math::SqrtOp::create(b(), loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::pow(Value base, Value exp) const {
  splatToMatch(base, exp);
  if (isScalarOrVectorFloat(base) && isScalarOrVectorFloat(exp))
    return math::PowFOp::create(b(), loc(), base, exp);
  if (isScalarOrVectorFloat(base) && isScalarOrVectorInteger(exp))
    return math::FPowIOp::create(b(), loc(), base, exp);
  if (isScalarOrVectorInteger(base) && isScalarOrVectorInteger(exp))
    return math::IPowIOp::create(b(), loc(), base, exp);
  llvm_unreachable("expected pow: float ^ float, int ^ int, float ^ int");
}

Value MathBuilder::neg(Value val) const {
  if (isScalarOrVectorInteger(val))
    // Returns 0 - val.
    return sub(constant(val.getType(), 0), val);
  if (isScalarOrVectorFloat(val))
    return arith::NegFOp::create(b(), loc(), val);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::ceil(Value val) const {
  if (isScalarOrVectorFloat(val))
    return math::CeilOp::create(b(), loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::floor(Value val) const {
  if (isScalarOrVectorFloat(val))
    return math::FloorOp::create(b(), loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::tanh(Value val) const {
  if (isScalarOrVectorFloat(val))
    return math::TanhOp::create(b(), loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::min(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorFloat(lhs))
    return arith::MinNumFOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorUnsignedInteger(lhs))
    return arith::MinUIOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorInteger(lhs))
    return arith::MinSIOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::max(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorFloat(lhs))
    return arith::MaxNumFOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorUnsignedInteger(lhs))
    return arith::MaxUIOp::create(b(), loc(), lhs, rhs);
  if (isScalarOrVectorInteger(lhs))
    return arith::MaxSIOp::create(b(), loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sgt(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sgt);
  if (isScalarOrVectorFloat(lhs))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OGT);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sge(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sge);
  if (isScalarOrVectorFloat(lhs))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OGE);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::slt(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::slt);
  if (isScalarOrVectorFloat(lhs))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OLT);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sle(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sle);
  if (isScalarOrVectorFloat(lhs))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OLE);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::ugt(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorUnsignedInteger(lhs))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ugt);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::uge(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorUnsignedInteger(lhs))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::uge);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::ult(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorUnsignedInteger(lhs))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ult);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::ule(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorUnsignedInteger(lhs))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ule);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::shli(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs)) {
    Type elemType = elementTypeOfScalarOrVector(lhs);
    if (elemType.isUnsignedInteger()) {
      unsigned elemWidth = mlir::cast<IntegerType>(elemType).getWidth();
      Value castLhs = castToSignless(lhs, elemWidth);
      Value castRhs = castToSignless(rhs, elemWidth);
      Value castShift = arith::ShLIOp::create(b(), loc(), castLhs, castRhs);
      return castToUnsigned(castShift, elemWidth);
    } else
      return arith::ShLIOp::create(b(), loc(), lhs, rhs);
  }
  llvm_unreachable("expected int");
}

Value MathBuilder::shri(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs)) {
    Type elemType = elementTypeOfScalarOrVector(lhs);
    if (elemType.isUnsignedInteger()) {
      unsigned elemWidth = mlir::cast<IntegerType>(elemType).getWidth();
      Value castLhs = castToSignless(lhs, elemWidth);
      Value castRhs = castToSignless(rhs, elemWidth);
      Value castShift = arith::ShRUIOp::create(b(), loc(), castLhs, castRhs);
      return castToUnsigned(castShift, elemWidth);
    } else
      return arith::ShRSIOp::create(b(), loc(), lhs, rhs);
  }
  llvm_unreachable("expected int");
}

Value MathBuilder::gt(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  if (isScalarOrVectorUnsignedInteger(lhs))
    return ugt(lhs, rhs);
  return sgt(lhs, rhs);
}

Value MathBuilder::ge(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  if (isScalarOrVectorUnsignedInteger(lhs))
    return uge(lhs, rhs);
  return sge(lhs, rhs);
}

Value MathBuilder::lt(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  if (isScalarOrVectorUnsignedInteger(lhs))
    return ult(lhs, rhs);
  return slt(lhs, rhs);
}

Value MathBuilder::le(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  if (isScalarOrVectorUnsignedInteger(lhs))
    return ule(lhs, rhs);
  return sle(lhs, rhs);
}

Value MathBuilder::eq(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::eq);
  if (isScalarOrVectorFloat(lhs))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OEQ);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::neq(Value lhs, Value rhs) const {
  splatToMatch(lhs, rhs);
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isScalarOrVectorInteger(lhs))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ne);
  if (isScalarOrVectorFloat(lhs))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::ONE);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::select(Value cmp, Value trueVal, Value falseVal) const {
  splatToMatch(cmp, trueVal, falseVal);
  assert(trueVal.getType() == falseVal.getType() && "expected same type");
  return arith::SelectOp::create(b(), loc(), cmp, trueVal, falseVal);
}

Value MathBuilder::constant(Type type, double val) const {
  Value constant = nullptr;
  // Could be a vector type; look at the element type.
  Type elementType = elementTypeOfScalarOrVector(type);
  TypeSwitch<Type>(elementType)
      .Case<Float16Type>([&](Type) {
        constant =
            arith::ConstantOp::create(b(), loc(), b().getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant =
            arith::ConstantOp::create(b(), loc(), b().getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant =
            arith::ConstantOp::create(b(), loc(), b().getF64FloatAttr(val));
      })
      .Case<IntegerType>([&](IntegerType elementType) {
        assert(val == static_cast<int64_t>(val) && "value is ambiguous");
        unsigned width = elementType.getWidth();

        if (width == 1)
          constant =
              arith::ConstantOp::create(b(), loc(), b().getBoolAttr(val != 0));
        else {
          // If unsigned, create a signless constant, then cast it to unsigned.
          if (elementType.isUnsignedInteger()) {
            Type signlessTy = b().getIntegerType(width);
            constant = arith::ConstantOp::create(b(), loc(),
                b().getIntegerAttr(signlessTy,
                    APInt(width, static_cast<int64_t>(val), false, true)));
            constant = castToUnsigned(constant, width);
          } else {
            constant = arith::ConstantOp::create(b(), loc(),
                b().getIntegerAttr(elementType,
                    APInt(width, static_cast<int64_t>(val), false, true)));
          }
        }
      })
      .Case<IndexType>([&](Type elementType) {
        constant = arith::ConstantOp::create(
            b(), loc(), b().getIntegerAttr(elementType, val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  if (mlir::isa<VectorType>(type)) {
    // For vectors, need to splat the constant.
    MultiDialectBuilder<VectorBuilder> create(*this);
    VectorType vecType = mlir::dyn_cast<VectorType>(type);
    constant = create.vec.broadcast(vecType, constant);
  }
  return constant;
}

Value MathBuilder::constantIndex(int64_t val) const {
  IntegerAttr constantAttr = b().getIntegerAttr(b().getIndexType(), val);
  return arith::ConstantOp::create(b(), loc(), constantAttr);
}

TypedAttr MathBuilder::negativeInfAttr(Type type) const {
  TypedAttr attr;
  TypeSwitch<Type>(type)
      .Case<Float32Type>([&](Type) {
        attr = b().getF32FloatAttr(-std::numeric_limits<float>::infinity());
      })
      .Case<Float64Type>([&](Type) {
        attr = b().getF64FloatAttr(-std::numeric_limits<double>::infinity());
      })
      .Case<IntegerType>([&](IntegerType type) {
        unsigned width = type.getWidth();
        bool isSignless = type.isSignless();
        bool isSigned = type.isSigned();
        int64_t value;
        switch (width) {
        case 8:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int8_t>::min()
                      : std::numeric_limits<uint8_t>::min();
          break;
        case 16:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int16_t>::min()
                      : std::numeric_limits<uint16_t>::min();
          break;
        case 32:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int32_t>::min()
                      : std::numeric_limits<uint32_t>::min();
          break;
        case 64:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int64_t>::min()
                      : std::numeric_limits<uint64_t>::min();
          break;
        default:
          llvm_unreachable("unsupported element type");
        }
        attr = b().getIntegerAttr(type, APInt(width, value, false, true));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });
  assert(attr != nullptr && "Expecting valid attribute");
  return attr;
}

TypedAttr MathBuilder::positiveInfAttr(Type type) const {
  TypedAttr attr;
  TypeSwitch<Type>(type)
      .Case<Float32Type>([&](Type) {
        attr = b().getF32FloatAttr(std::numeric_limits<float>::infinity());
      })
      .Case<Float64Type>([&](Type) {
        attr = b().getF64FloatAttr(std::numeric_limits<double>::infinity());
      })
      .Case<IntegerType>([&](IntegerType type) {
        unsigned width = type.getWidth();
        bool isSignless = type.isSignless();
        bool isSigned = type.isSigned();
        int64_t value;
        switch (width) {
        case 8:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int8_t>::max()
                      : std::numeric_limits<uint8_t>::max();
          break;
        case 16:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int16_t>::max()
                      : std::numeric_limits<uint16_t>::max();
          break;
        case 32:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int32_t>::max()
                      : std::numeric_limits<uint32_t>::max();
          break;
        case 64:
          value = (isSignless || isSigned)
                      ? std::numeric_limits<int64_t>::max()
                      : std::numeric_limits<uint64_t>::max();
          break;
        default:
          llvm_unreachable("unsupported element type");
        }
        attr = b().getIntegerAttr(type, APInt(width, value, false, true));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });
  assert(attr != nullptr && "Expecting valid attribute");
  return attr;
}

Value MathBuilder::negativeInf(Type type) const {
  // Strip vector type if any.
  Type elementType = elementTypeOfScalarOrVector(type);
  TypedAttr attr = negativeInfAttr(elementType);
  Value constant = arith::ConstantOp::create(b(), loc(), attr);
  assert(constant != nullptr && "Expecting valid constant value");
  if (mlir::isa<VectorType>(type)) {
    // For vectors, need to splat the constant.
    MultiDialectBuilder<VectorBuilder> create(*this);
    VectorType vecType = mlir::dyn_cast<VectorType>(type);
    constant = create.vec.broadcast(vecType, constant);
  }
  return constant;
}

Value MathBuilder::positiveInf(Type type) const {
  // Strip vector type if any.
  Type elementType = elementTypeOfScalarOrVector(type);
  TypedAttr attr = positiveInfAttr(elementType);
  Value constant = arith::ConstantOp::create(b(), loc(), attr);
  assert(constant != nullptr && "Expecting valid constant value");
  if (mlir::isa<VectorType>(type)) {
    // For vectors, need to splat the constant.
    MultiDialectBuilder<VectorBuilder> create(*this);
    VectorType vecType = mlir::dyn_cast<VectorType>(type);
    constant = create.vec.broadcast(vecType, constant);
  }
  return constant;
}

Value MathBuilder::createArithCmp(
    Value lhs, Value rhs, arith::CmpIPredicate pred) const {
  Type type = lhs.getType();
  assert(type == rhs.getType() && "Operands should have the same type");
  assert(isScalarOrVectorInteger(type) && "expected int");
  return arith::CmpIOp::create(b(), loc(), pred, lhs, rhs);
}

Value MathBuilder::createArithCmp(
    Value lhs, Value rhs, arith::CmpFPredicate pred) const {
  Type type = lhs.getType();
  assert(type == rhs.getType() && "Operands should have the same type");
  assert(isScalarOrVectorFloat(type) && "expected float");
  return arith::CmpFOp::create(b(), loc(), pred, lhs, rhs);
}

// Several operations in the arith dialect require signless integers. This
// cast remove the sign of integer types for successful processing, to the
// best of my understanding.
Value MathBuilder::castToSignless(Value val, int64_t width) const {
  Type valType = val.getType();
  Type valElemType = elementTypeOfScalarOrVector(valType);
  assert(mlir::isa<IntegerType>(valElemType) &&
         !valElemType.isSignlessInteger() && "Expecting signed integer type");
  Type destType = getTypeWithVector(valType, b().getIntegerType(width));
  return UnrealizedConversionCastOp::create(b(), loc(), destType, val)
      .getResult(0);
}

Value MathBuilder::castToUnsigned(Value val, int64_t width) const {
  Type valType = val.getType();
  Type valElemType = elementTypeOfScalarOrVector(valType);
  assert(mlir::isa<IntegerType>(valElemType) && "Expecting integer type");
  Type destType =
      getTypeWithVector(valType, b().getIntegerType(width, false /*signed*/));
  return UnrealizedConversionCastOp::create(b(), loc(), destType, val)
      .getResult(0);
}

// Methods inspired from MLIR TosaToLinalg CastOp.
Value MathBuilder::cast(Type destType, Value src) const {
  Type srcType = src.getType();
  // Check if we even need a cast.
  if (srcType == destType)
    return src;
  // Get element type and vector types (if any, i.e. possibly nullptr).

  ///////////////////////////////////////////////////////////////////////
  // WARNING: do not confuse (src|dest) ElemType and (src|dest) Type!
  //
  // ElemTypes and Types are the same for scalar BUT NOT for vector inputs.
  // For vectors inputs, Types are vector<shape x element type> and ElemTypes
  // are the element type associated with the vector.
  //
  // When testing for properties (is int, float,...): use ElemTypes.
  // When creating ops, use Types for types to translate to, as if we have a
  // scalar input, we need a scalar output; and if we have a vector input, then
  // we need a vector output.
  ///////////////////////////////////////////////////////////////////////

  Type srcElemType = elementTypeOfScalarOrVector(srcType);
  Type destElemType = elementTypeOfScalarOrVector(destType);
  VectorType srcVecType = mlir::dyn_cast<VectorType>(srcType);
  VectorType destVecType = mlir::dyn_cast<VectorType>(destType);
  assert(VectorBuilder::compatibleShapes(srcType, destType) &&
         "expected compatible vector shape (if any)");

  // Handling of special cases for vectors.
  if (destVecType && !srcVecType) {
    // When the destination type is requested to be a vector type, but the input
    // is not, then perform a scalar cast first, and then splat the output.
    Value scalarCastVal = cast(destElemType, src);
    MultiDialectBuilder<VectorBuilder> create(*this);
    return create.vec.broadcast(destVecType, scalarCastVal);
  }
  if (srcVecType && !destVecType) {
    // When the source (to be cast) is a vector, but the destination type is
    // not, then just transform the destination type to a vector of the same
    // shape as srcType and the elementType of destType.
    destType = getTypeWithVector(srcType, destElemType);
    assert(destElemType == elementTypeOfScalarOrVector(destType) &&
           "correctness check");
  }

  // Process index types first.
  if (mlir::isa<IndexType>(srcElemType)) {
    // If the source is an index type, first convert it into a signless int of
    // size 64.
    srcElemType = b().getIntegerType(64);
    srcType = getTypeWithVector(srcType, srcElemType);
    src = arith::IndexCastOp::create(b(), loc(), srcType, src);
  }
  bool destIsIndex = false;
  Type savedDestType = destType; // Used when destIsIndex is true.
  if (mlir::isa<IndexType>(destElemType)) {
    // If the dest is an index type, pretend for now that we want it to be
    // converted to signless int of size 64.
    destElemType = b().getIntegerType(64);
    destType = getTypeWithVector(destType, destElemType);
    destIsIndex = true;
  }

  // Only support Integer or Float type at this stage. Index were transformed
  // to signless int.
  // TODO: add support for shaped tensor (MemRef, Vector, Tensor?) if needed.
  assert((mlir::isa<IntegerType>(srcElemType) ||
             mlir::isa<FloatType>(srcElemType)) &&
         "support only float or int");
  assert((mlir::isa<IntegerType>(destElemType) ||
             mlir::isa<FloatType>(destElemType)) &&
         "support only float or int");
  // Get source and dest type width.
  int64_t srcElemWidth = srcElemType.getIntOrFloatBitWidth();
  int64_t destElemWidth = destElemType.getIntOrFloatBitWidth();
  bool bitExtend = srcElemWidth < destElemWidth;
  bool bitTrunc = srcElemWidth > destElemWidth;

  LLVM_DEBUG(llvm::dbgs() << "srcType: " << srcType << "\n";
             llvm::dbgs() << "destType: " << destType << "\n";);

  // Before we process with the actual cast, there is a special case that we
  // want to handle here. Cast from float to int that have different width, llvm
  // generate better patterns if we first cast from float to int of the same
  // width, and then from int to a different size int.
  // Skip that optimization if the result is a 1 bit (boolean).
  if (mlir::isa<FloatType>(srcElemType) &&
      mlir::isa<IntegerType>(destElemType) && bitTrunc && destElemWidth > 1) {
    // Quantization: float to smaller int. First determine the intermediary
    // type, same integer type as destination type, with the same type width as
    // the source float type.
    Type step1ElementType;
    IntegerType destIntType = mlir::cast<IntegerType>(destElemType);
    bool destIssSigned = destIntType.isSignless() || destIntType.isSigned();
    if (destIssSigned)
      step1ElementType = b().getIntegerType(srcElemWidth);
    else
      step1ElementType = b().getIntegerType(srcElemWidth, false);
    // Perform (recursively) the 2 step conversion. Exceptionally ok here to use
    // element type here as cast will promote it to a vector if src is a vector.
    Value step1Val = cast(step1ElementType, src);
    return cast(destType, step1Val);
  }
  if (mlir::isa<IntegerType>(srcElemType) &&
      mlir::isa<FloatType>(destElemType) && bitExtend) {
    // Dequantization: small int to a float. First determine the intermediary
    // type,  same integer type as source type, with the same type width as
    // the destination float type.
    Type step1ElementType;
    IntegerType srcIntType = mlir::cast<IntegerType>(srcElemType);
    bool srcIssSigned = srcIntType.isSignless() || srcIntType.isSigned();
    if (srcIssSigned)
      step1ElementType = b().getIntegerType(destElemWidth);
    else
      step1ElementType = b().getIntegerType(destElemWidth, false);
    // Perform (recursively) the 2 step conversion. Exceptionally ok here to use
    // element type here as cast will promote it to a vector if src is a vector.
    Value step1Val = cast(step1ElementType, src);
    return cast(destType, step1Val);
  }

  // Handle boolean first because they need special handling.
  // Boolean to int/float conversions. Boolean are unsigned.
  if (srcElemType.isInteger(1)) {
    if (mlir::isa<FloatType>(destElemType)) {
      return arith::UIToFPOp::create(b(), loc(), destType, src);
    } else {
      Value dest = arith::ExtUIOp::create(b(), loc(), destType, src);
      if (destIsIndex)
        dest = arith::IndexCastOp::create(b(), loc(), savedDestType, dest);
      return dest;
    }
  }

  // Int/Float to booleans, just compare value to be unequal zero.
  if (destElemType.isInteger(1)) {
    Type constantType = srcType;
    if (mlir::isa<IntegerType>(srcElemType) &&
        !srcElemType.isSignlessInteger()) {
      // An integer constant must be signless.
      unsigned srcElemWidth = mlir::cast<IntegerType>(srcElemType).getWidth();
      constantType = getTypeWithVector(
          srcType, IntegerType::get(srcElemType.getContext(), srcElemWidth));
      src = castToSignless(src, srcElemWidth);
    }
    Value zero = constant(constantType, 0);
    return neq(src, zero);
  }

  // Float to float conversions.
  if (mlir::isa<FloatType>(srcElemType) && mlir::isa<FloatType>(destElemType)) {
    assert((bitExtend || bitTrunc) && "expected extend or trunc");
    if (bitExtend)
      return arith::ExtFOp::create(b(), loc(), destType, src);
    else
      return arith::TruncFOp::create(b(), loc(), destType, src);
  }

  // Float to int conversions.
  if (mlir::isa<FloatType>(srcElemType) &&
      mlir::isa<IntegerType>(destElemType)) {
    // TosaToLinalg in MLIR uses a fancier algorithm that clamps values to
    // min/max signed/unsigned integer values.
    if (destElemType.isUnsignedInteger()) {
      Type castElementType = b().getIntegerType(destElemWidth);
      Type castType = getTypeWithVector(destType, castElementType);
      Value cast = arith::FPToUIOp::create(b(), loc(), castType, src);
      return castToUnsigned(cast, destElemWidth);
    } else {
      // Handle signed int.
      Value dest = arith::FPToSIOp::create(b(), loc(), destType, src);
      if (destIsIndex)
        dest = arith::IndexCastOp::create(b(), loc(), savedDestType, dest);
      return dest;
    }
  }

  // Int to float conversion.
  if (mlir::isa<IntegerType>(srcElemType) &&
      mlir::isa<FloatType>(destElemType)) {
    if (srcElemType.isUnsignedInteger()) {
      Value cast = castToSignless(src, srcElemWidth);
      return arith::UIToFPOp::create(b(), loc(), destType, cast);
    } else {
      // Handle signed int.
      return arith::SIToFPOp::create(b(), loc(), destType, src);
    }
  }

  // Int to int conversion.
  if (mlir::isa<IntegerType>(srcElemType) &&
      mlir::isa<IntegerType>(destElemType)) {
    if (srcElemType.isUnsignedInteger()) {
      // Unsigned to unsigned/signed conversion.
      // Same bit width for unsigned to signed conversion.
      if ((srcElemWidth == destElemWidth) && destElemType.isSignlessInteger())
        return castToSignless(src, srcElemWidth);
      // Different bit width.
      assert((bitExtend || bitTrunc) && "expected extend or trunc");
      // Has to convert to signless first, and reconvert output to unsigned.
      Value cast = castToSignless(src, srcElemWidth);
      Type castElemType = b().getIntegerType(destElemWidth);
      Type castType = getTypeWithVector(destType, castElemType);
      if (bitExtend) {
        cast = arith::ExtUIOp::create(b(), loc(), castType, cast);
      } else {
        // TosaToLinalg use a clipping algo, not sure if needed.
        cast = arith::TruncIOp::create(b(), loc(), castType, cast);
      }
      if (destElemType.isUnsignedInteger()) {
        // Unsigned to unsigned conversion.
        return castToUnsigned(cast, destElemWidth);
      } else {
        // Unsigned to signed conversion.
        return cast;
      }
    } else {
      // Signed to unsigned/signed conversion.
      // Handle signed integer
      // Same bit width for signed to unsigned conversion.
      if ((srcElemWidth == destElemWidth) && destElemType.isUnsignedInteger())
        return castToUnsigned(src, srcElemWidth);
      // Different bit width.
      Value dest = src;
      if (bitExtend)
        dest = arith::ExtSIOp::create(b(), loc(), destType, src);
      if (bitTrunc)
        // TosaToLinalg use a clipping algo
        dest = arith::TruncIOp::create(b(), loc(), destType, src);
      if (destIsIndex)
        return arith::IndexCastOp::create(b(), loc(), b().getIndexType(), dest);
      if (destElemType.isUnsignedInteger()) {
        return castToUnsigned(dest, destElemWidth);
      } else {
        return dest;
      }
    }
  }

  // Handled all the cases supported so far.
  llvm_unreachable("unsupported element type");
  return nullptr;
}

Value MathBuilder::castToIndex(Value src) const {
  return cast(b().getIndexType(), src);
}

// Add offsets to least significant values in indices. So if indices has 4
// values, (i, j, k, l) and offsets has 2 values (K, L), the results will be (i,
// j, k+K, l+L).
void MathBuilder::addOffsetToLeastSignificant(ValueRange indices,
    ValueRange offsets, llvm::SmallVectorImpl<Value> &computedIndices) const {
  int64_t indexRank = indices.size();
  int64_t offsetRank = offsets.size();
  int64_t firstOffset = indexRank - offsetRank;
  assert(firstOffset >= 0 && "indexOffset should not have a higher rank than "
                             "the indices in the memref");
  computedIndices.clear();
  for (int64_t i = 0; i < indexRank; i++) {
    if (i < firstOffset) {
      computedIndices.emplace_back(indices[i]);
    } else {
      computedIndices.emplace_back(add(offsets[i - firstOffset], indices[i]));
    }
  }
}

void MathBuilder::addOffsetToLeastSignificant(ArrayRef<IndexExpr> indices,
    ValueRange offsets, llvm::SmallVectorImpl<Value> &computedIndices) const {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  addOffsetToLeastSignificant(indexValues, offsets, computedIndices);
}

//===----------------------------------------------------------------------===//
// Shape support.
//===----------------------------------------------------------------------===//

Value ShapeBuilder::dim(Value val, int64_t index) const {
  Value inputShape = shapeOf(val);
  return getExtent(inputShape, index);
}

Value ShapeBuilder::shapeOf(Value val) const {
  return shape::ShapeOfOp::create(b(), loc(), val);
}

Value ShapeBuilder::fromExtents(ValueRange extents) const {
  return shape::FromExtentsOp::create(b(), loc(), extents);
}

Value ShapeBuilder::toExtentTensor(Type type, Value shape) const {
  return shape::ToExtentTensorOp::create(b(), loc(), type, shape);
}

Value ShapeBuilder::getExtent(Value val, int64_t index) const {
  return shape::GetExtentOp::create(b(), loc(), val, index);
}

//===----------------------------------------------------------------------===//
// Memref support, including inserting default alignment.
//===----------------------------------------------------------------------===//

const int64_t MemRefBuilder::defaultAlign = -1;

//===----------------------------------------------------------------------===//
// Helper private functions.

// Compute alignment, which is at least gDefaultAllocAlign.
IntegerAttr MemRefBuilder::computeAlignment(int64_t alignment) const {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  return b().getI64IntegerAttr(alignment);
}

// Alloc calls need a list of values, only for the dynamic shapes. Extract these
// values from the list of index expressions that represent the shape of the
// memref.

void MemRefBuilder::computeDynSymbols(MemRefType type, DimsExprRef dims,
    llvm::SmallVectorImpl<Value> &dynSymbols) const {
  dynSymbols.clear();
  int64_t rank = type.getRank();
  ArrayRef<int64_t> shape = type.getShape();
  for (int64_t i = 0; i < rank; ++i)
    if (shape[i] == ShapedType::kDynamic)
      dynSymbols.emplace_back(dims[i].getValue());
}

// Alloc calls need a list of values, only for the dynamic shapes. Extract these
// values from an existing operands that has the same shape. Use dim ops for
// each dynamic dimension.
void MemRefBuilder::computeDynSymbols(Value operandOfSameType, MemRefType type,
    llvm::SmallVectorImpl<Value> &dynSymbols) const {
  dynSymbols.clear();
  if (operandOfSameType == nullptr)
    return;
  int64_t rank = type.getRank();
  ArrayRef<int64_t> shape = type.getShape();
  for (int64_t i = 0; i < rank; ++i)
    if (shape[i] == ShapedType::kDynamic)
      dynSymbols.emplace_back(dim(operandOfSameType, i));
}

//===----------------------------------------------------------------------===//
// Load Store ops.

Value MemRefBuilder::load(
    Value memref, ValueRange indices, ValueRange offsets) const {
  return onnx_mlir::impl::load<MemRefBuilder, memref::LoadOp>(
      *this, memref, indices, offsets);
}
Value MemRefBuilder::loadIE(
    Value memref, ArrayRef<IndexExpr> indices, ValueRange offsets) const {
  return onnx_mlir::impl::loadIE<MemRefBuilder, memref::LoadOp>(
      *this, memref, indices, offsets);
}

// Add offsets (if any) to the least significant memref dims.
void MemRefBuilder::store(
    Value val, Value memref, ValueRange indices, ValueRange offsets) const {
  onnx_mlir::impl::store<MemRefBuilder, memref::StoreOp>(
      *this, val, memref, indices, offsets);
}

void MemRefBuilder::storeIE(Value val, Value memref,
    ArrayRef<IndexExpr> indices, ValueRange offsets) const {
  onnx_mlir::impl::storeIE<MemRefBuilder, memref::StoreOp>(
      *this, val, memref, indices, offsets);
}

//===----------------------------------------------------------------------===//
// Alloc functions without alignment.

memref::AllocOp MemRefBuilder::alloc(MemRefType type) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  return alloc(type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(
    MemRefType type, ValueRange dynSymbols) const {
  // Constant, ignore the dynamic symbols.
  if (dynSymbols.size() == 0)
    return memref::AllocOp::create(b(), loc(), type);
  return memref::AllocOp::create(b(), loc(), type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(
    Value operandOfSameType, MemRefType type) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(operandOfSameType, type, dynSymbols);
  return alloc(type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(MemRefType type, DimsExprRef dims) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return alloc(type, dynSymbols);
}

//===----------------------------------------------------------------------===//
// Alloc functions with alignment.

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, int64_t alignment) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  return alignedAlloc(type, dynSymbols, alignment);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, ValueRange dynSymbols, int64_t alignment) const {
  // Drop align for scalars.
  if (type.getShape().size() == 0)
    return alloc(type, dynSymbols);
  // Has array, use alignment.
  IntegerAttr alignmentAttr = computeAlignment(alignment);
  // Constant, ignore the dynamic symbols.
  if (dynSymbols.size() == 0)
    return memref::AllocOp::create(b(), loc(), type, alignmentAttr);
  return memref::AllocOp::create(b(), loc(), type, dynSymbols, alignmentAttr);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    Value operandOfSameType, MemRefType type, int64_t alignment) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(operandOfSameType, type, dynSymbols);
  return alignedAlloc(type, dynSymbols, alignment);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, DimsExprRef dims, int64_t alignment) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return alignedAlloc(type, dynSymbols, alignment);
}

//===----------------------------------------------------------------------===//
// Info about memory size.

// Compute static size of memref in elements. Return true if has
// static size.
/*static*/ bool MemRefBuilder::getStaticMemSize(
    MemRefType type, int64_t &staticSize, int64_t range) {
  Type elementType = type.getElementType();
  assert(!(mlir::isa<VectorType>(elementType)) && "unsupported vector type");
  ArrayRef<int64_t> shape = type.getShape();
  staticSize = 1;          // Multiplication of static sizes.
  bool staticShape = true; // Static until proven otherwise.
  int64_t rank = type.getRank();
  // Process with range [lb inclusive, ub exclusive)
  int64_t lb = 0, ub = rank;
  if (range == 0)
    // Empty range, nothing to do.
    return staticShape;
  if (range > 0) {
    // Positive range r: interval is [ 0, min(r, rank) ).
    ub = (range < rank) ? range : rank;
  } else {
    // Negative range r: interval is [ max(0, r+rank) to rank ).
    range += rank;
    lb = range > 0 ? range : 0;
  }
  assert(lb >= 0 && ub <= rank && "out of bound range");
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] == ShapedType::kDynamic) {
      if (i >= lb && i < ub) {
        // Keep track of static shape and dynamic sizes only when inbounds.
        staticShape = false;
      }
    } else {
      // Has constant shape.
      if (i >= lb && i < ub) {
        // Keep track of static size only when inbounds.
        staticSize *= shape[i];
      }
    }
  }
  return staticShape;
}

// Compute static and dynamic size of memref in elements. Return true if has
// static size.
bool MemRefBuilder::getStaticAndDynamicMemSize(MemRefType type,
    ValueRange dynSymbols, int64_t &staticSize, IndexExpr &dynSize,
    int64_t range) const {
  Type elementType = type.getElementType();
  assert(!(mlir::isa<VectorType>(elementType)) && "unsupported vector type");
  ArrayRef<int64_t> shape = type.getShape();
  staticSize = 1;          // Multiplication of static sizes.
  dynSize = LitIE(1);      // Multiplication of dyn sizes.
  bool staticShape = true; // Static until proven otherwise.
  int64_t rank = type.getRank();
  // Process with range [lb inclusive, ub exclusive)
  int64_t lb = 0, ub = rank;
  if (range == 0)
    // Empty range, nothing to do.
    return staticShape;
  if (range > 0) {
    // Positive range r: interval is [ 0, min(r, rank) ).
    ub = (range < rank) ? range : rank;
  } else {
    // Negative range r: interval is [ max(0, r+rank) to rank ).
    range += rank;
    lb = range > 0 ? range : 0;
  }
  assert(lb >= 0 && ub <= rank && "out of bound range");
  int64_t iDim = 0;
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] == ShapedType::kDynamic) {
      assert(iDim < (int64_t)dynSymbols.size() && "not enough dynamic symbols");
      if (i >= lb && i < ub) {
        // Keep track of static shape and dynamic sizes only when inbounds.
        staticShape = false;
        dynSize = dynSize * SymIE(dynSymbols[iDim]);
      }
      iDim++;
    } else {
      // Has constant shape.
      if (i >= lb && i < ub) {
        // Keep track of static size only when inbounds.
        staticSize *= shape[i];
      }
    }
  }
  return staticShape;
}

bool MemRefBuilder::getStaticAndDynamicMemSize(MemRefType type,
    DimsExprRef dims, int64_t &staticSize, IndexExpr &dynSize,
    int64_t range) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return getStaticAndDynamicMemSize(
      type, dynSymbols, staticSize, dynSize, range);
}

//===----------------------------------------------------------------------===//
// Alloc functions with alignment and padding for SIMD

Value MemRefBuilder::alignedAllocWithSimdPadding(
    MemRefType type, int64_t VL, int64_t alignment) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  return alignedAllocWithSimdPadding(type, dynSymbols, VL, alignment);
}

Value MemRefBuilder::alignedAllocWithSimdPadding(MemRefType type,
    ValueRange dynSymbols, int64_t VL, int64_t alignment) const {
  Type elementType = type.getElementType();
  assert(!hasNonIdentityLayout(type) && "unsupported layout");
  assert(!(mlir::isa<VectorType>(elementType)) && "unsupported vector type");
  assert(VL >= 1 && "expected positive simd unroll factor");
  // Compute total size of memref (in unit of element type).
  int64_t staticSize;
  IndexExpr dynSize;
  bool staticShape =
      getStaticAndDynamicMemSize(type, dynSymbols, staticSize, dynSize);
  // Get vector length for this element type, multiplied by the unroll factor.
  // If the static size component is already a multiple of VL, no matter the
  // values of the dynamic shapes, the last value is part of a full SIMD. No
  // need for extra padding then.
  if (staticSize % VL == 0)
    return alignedAlloc(type, dynSymbols, alignment);

  // We now need some padding. VL as this is an upper bound on padding. Padding
  // in element size.
  int64_t paddingSize = VL;
  if (staticShape)
    // Static shape: we can pad by the exact right amount.
    paddingSize = VL - staticSize % VL;

  // Allocate data as byte.
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  IndexExpr totPaddedByteSize;
  if (bitWidth % 8 == 0) {
    // We have elements that have sizes of 1 or more bytes.
    int64_t byteWidth = bitWidth / 8;
    IndexExpr totByteSize = LitIE(staticSize * byteWidth) * dynSize;
    totPaddedByteSize = totByteSize + LitIE(paddingSize * byteWidth);
  } else {
    // We have sub-byte element sizes. Need to do precise computations. Namely
    // first compute tot total number of bits (including static/dynamic
    // and padding bit sizes), and then doing a ceil division by
    // 8 (number of bits in a byte).
    IndexExpr totBitSize = LitIE(staticSize * bitWidth) * dynSize;
    IndexExpr totPaddedBitSize = totBitSize + LitIE(paddingSize * bitWidth);
    totPaddedByteSize = totPaddedBitSize.ceilDiv(LitIE(8));
  }
  if (staticShape)
    assert(totPaddedByteSize.isLiteral() && "expected literal padded tot size");
  // Construct memref for padded array of bytes.
  memref::AllocOp paddedAlloc;
  if (totPaddedByteSize.isLiteral()) {
    MemRefType paddedType =
        MemRefType::get({totPaddedByteSize.getLiteral()}, b().getI8Type());
    paddedAlloc = alignedAlloc(paddedType, alignment);
  } else {
    MemRefType paddedType =
        MemRefType::get({ShapedType::kDynamic}, b().getI8Type());
    paddedAlloc =
        alignedAlloc(paddedType, {totPaddedByteSize.getValue()}, alignment);
  }
  // Used to create a subview, it does not appear that the view cares about
  // whether the entire input data participates in the viewed data or not.
  return view(paddedAlloc, /*offset*/ 0, type, dynSymbols);
}

Value MemRefBuilder::alignedAllocWithSimdPadding(Value operandOfSameType,
    MemRefType type, int64_t VL, int64_t alignment) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(operandOfSameType, type, dynSymbols);
  return alignedAllocWithSimdPadding(type, dynSymbols, VL, alignment);
}

Value MemRefBuilder::alignedAllocWithSimdPadding(
    MemRefType type, DimsExprRef dims, int64_t VL, int64_t alignment) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return alignedAllocWithSimdPadding(type, dynSymbols, VL, alignment);
}

//===----------------------------------------------------------------------===//
// Alloca

memref::AllocaOp MemRefBuilder::alloca(MemRefType type) const {
  return memref::AllocaOp::create(b(), loc(), type);
}

memref::AllocaOp MemRefBuilder::alignedAlloca(
    MemRefType type, int64_t alignment) const {
  // Drop align for scalars.
  if (type.getShape().size() == 0)
    return memref::AllocaOp::create(b(), loc(), type);
  // Has array, use alignment.
  IntegerAttr alignmentAttr = computeAlignment(alignment);
  return memref::AllocaOp::create(b(), loc(), type, alignmentAttr);
}

//===----------------------------------------------------------------------===//
// Dealloc.

memref::DeallocOp MemRefBuilder::dealloc(Value val) const {
  return memref::DeallocOp::create(b(), loc(), val);
}

//===----------------------------------------------------------------------===//
// Reshape.

memref::ReshapeOp MemRefBuilder::reshape(MemRefType destType,
    Value valToReshape, Value outputShapeStoredInMem) const {
  return memref::ReshapeOp::create(
      b(), loc(), destType, valToReshape, outputShapeStoredInMem);
}

memref::ReshapeOp MemRefBuilder::reshape(
    DimsExpr &destDims, Value valToReshape) const {
  // Compute Shape.
  llvm::SmallVector<int64_t, 4> outputShape;
  IndexExpr::getShape(destDims, outputShape);
  // Allocate data structure for dimensions.
  // Question: is there a more optimized sequence if destDims is entirely
  // literal.
  Type indexType = b().getIndexType();
  int64_t outputRank = destDims.size();
  Value outputShapeInMem =
      alignedAlloc(MemRefType::get({outputRank}, indexType));
  // Store shape into data structure.
  MultiDialectBuilder<AffineBuilder, MathBuilder> create(*this);
  for (int64_t d = 0; d < outputRank; ++d) {
    Value dd = create.math.constantIndex(d);
    create.affine.store(destDims[d].getValue(), outputShapeInMem, {dd});
  }
  // Create output type.
  Type elementType =
      mlir::cast<MemRefType>(valToReshape.getType()).getElementType();
  MemRefType destType = MemRefType::get(outputShape, elementType);
  // Perform actual reshape operation
  return reshape(destType, valToReshape, outputShapeInMem);
}

// Flatten the innermost dimsToFlatten of the value valToReshape. Return in
// flattenSize the cumulative size of the flattened dimensions. Expect to
// flatten at least 1 dim (which is a noop). Output rank is Rank(input) -
// dimsToFlatten + 1.
Value MemRefBuilder::reshapeToFlatInnermost(Value valToReshape,
    DimsExprRef dims, DimsExpr &flattenedDims, int64_t dimsToFlatten) const {
  // Parse input.
  MemRefType inputType = mlir::cast<MemRefType>(valToReshape.getType());
  assert(!hasNonIdentityLayout(inputType) && "MemRef is not normalized");
  int64_t inputRank = inputType.getRank();
  // Verify dims has the right number of elements.
  assert(inputRank == (int64_t)dims.size() && "rank mismatch");
  assert(dimsToFlatten > 0 && dimsToFlatten <= inputRank &&
         "dimsToFlatten is out of range");
  if (dimsToFlatten == 1) {
    // Flattening of the last dim is really no flattening at all. Return
    // original value before doing the actual reshaping, which is unnecessary.
    for (IndexExpr d : dims)
      flattenedDims.emplace_back(d);
    return valToReshape;
  }
  // Compute the dimensions of the flattened array.
  int64_t axis = inputRank - dimsToFlatten;
  flattenedDims.clear();
  // Up to axis, flatten dims == input dims.
  for (int64_t d = 0; d < axis; ++d)
    flattenedDims.emplace_back(dims[d]);
  // Last flatten dim is the product of remaining input dims.
  IndexExpr numOfFlattenedElements = LitIE(1);
  for (int64_t d = axis; d < inputRank; ++d)
    numOfFlattenedElements = numOfFlattenedElements * dims[d];
  flattenedDims.emplace_back(numOfFlattenedElements);
  // Reshape.
  return reshape(flattenedDims, valToReshape);
}

Value MemRefBuilder::reshapeToFlat2D(Value valToReshape, DimsExprRef dims,
    DimsExpr &flattenedDims, int64_t axis) const {
  // Parse input.
  MemRefType inputType = mlir::cast<MemRefType>(valToReshape.getType());
  assert(!hasNonIdentityLayout(inputType) && "MemRef is not normalized");
  int64_t inputRank = inputType.getRank();
  // Verify dims has the right number of elements.
  assert(inputRank == (int64_t)dims.size() && "rank mismatch");
  if (axis < 0)
    axis += inputRank;
  assert(axis > 0 && axis < inputRank && "axis is out of range");
  if (inputRank == 2) {
    // Input is already 2D, nothing to do.
    for (IndexExpr d : dims)
      flattenedDims.emplace_back(d);
    return valToReshape;
  }
  // Compute the dimensions of the flattened array.
  flattenedDims.clear();
  // First output dim: product of input dims until axis (exclusively).
  IndexExpr numElement1stDim = LitIE(1);
  for (int64_t d = 0; d < axis; ++d)
    numElement1stDim = numElement1stDim * dims[d];
  flattenedDims.emplace_back(numElement1stDim);
  // Second output dim: product of input dims after axis (inclusively).
  IndexExpr numElement2ndDim = LitIE(1);
  for (int64_t d = axis; d < inputRank; ++d)
    numElement2ndDim = numElement2ndDim * dims[d];
  flattenedDims.emplace_back(numElement2ndDim);
  // Reshape.
  return reshape(flattenedDims, valToReshape);
}

memref::ReshapeOp MemRefBuilder::reshapeFromFlat(
    Value valToReshape, DimsExpr &outputDims, MemRefType outputType) const {
  assert(!hasNonIdentityLayout(outputType) && "MemRef is not normalized");
  return reshape(outputDims, valToReshape);
}

//===----------------------------------------------------------------------===//
// Casts and views.

memref::CastOp MemRefBuilder::cast(Value input, MemRefType outputType) const {
  return memref::CastOp::create(b(), loc(), outputType, input);
}

Value MemRefBuilder::reinterpretCast(Value input, DimsExpr &outputDims) const {
  return reinterpretCast(input, nullptr, outputDims);
}

Value MemRefBuilder::reinterpretCast(
    Value input, Value offset, DimsExpr &outputDims) const {
  // Compute new sizes and strides.
  int64_t rank = outputDims.size();
  SmallVector<IndexExpr, 4> sizesIE, stridesIE;
  sizesIE.resize(rank);
  stridesIE.resize(rank);
  IndexExpr strideIE = LitIE(1);
  for (int i = rank - 1; i >= 0; --i) {
    sizesIE[i] = outputDims[i];
    stridesIE[i] = strideIE;
    if (i > 0)
      strideIE = strideIE * sizesIE[i];
  }
  // Compute output type
  SmallVector<int64_t, 4> outputShape;
  SmallVector<OpFoldResult, 4> sizes, strides;
  IndexExpr::getShape(outputDims, outputShape);
  IndexExpr::getOpOrFoldResults(sizesIE, sizes);
  IndexExpr::getOpOrFoldResults(stridesIE, strides);
  Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
  MemRefType outputMemRefType = MemRefType::get(outputShape, elementType);
  if (offset)
    return memref::ReinterpretCastOp::create(
        b(), loc(), outputMemRefType, input, offset, sizes, strides);
  // Null offset: use zero attribute (remain compatible with old lit tests).
  return memref::ReinterpretCastOp::create(b(), loc(), outputMemRefType, input,
      /*offset*/ b().getIndexAttr(0), sizes, strides);
}

Value MemRefBuilder::collapseShape(
    Value input, ArrayRef<ReassociationIndices> reassociation) {
  // Extract input info.
  MemRefType inputType = mlir::cast<MemRefType>(input.getType());
  assert(inputType && "expected input with memref type");
  assert(!hasNonIdentityLayout(inputType) &&
         "collapse only for identity layout at this time");
  int64_t inputRank = inputType.getRank();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  // Compute shape of output.
  int64_t outputRank = reassociation.size();
  SmallVector<int64_t, 4> outputShape;
  for (int64_t r = 0; r < outputRank; ++r) {
    int64_t indexNum = reassociation[r].size();
    assert(indexNum > 0 && "expect one or more index in reassociation indices");
    // Compute the cumulative size of the output dim as the product of all dim
    // of the sizes in the input being re-associated with this output.
    int64_t currShape = 1;
    for (int64_t i = 0; i < indexNum; i++) {
      int64_t ii = reassociation[r][i];
      assert(ii >= 0 && ii < inputRank && "out of bound reassociation index");
      int64_t ss = inputShape[ii];
      if (ss == ShapedType::kDynamic) {
        // If a re-associated shapes is dynamic, output is dynamic.
        currShape = ShapedType::kDynamic;
        break;
      }
      currShape *= ss;
    }
    outputShape.emplace_back(currShape);
  }
  // Compute type of output.
  MemRefType outputType =
      MemRefType::get(outputShape, inputType.getElementType());
  // Create collapse shape op.
  return memref::CollapseShapeOp::create(
      b(), loc(), outputType, input, reassociation);
}

memref::ViewOp MemRefBuilder::view(Value input, int64_t byteOffset,
    MemRefType outputType, ValueRange outputDynSymbols) const {
  MultiDialectBuilder<MathBuilder> create(*this);
  Value offset = create.math.constantIndex(byteOffset);
  // auto offset = b().createOrFold<arith::ConstantIndexOp>(byteOffset);
  return memref::ViewOp::create(
      b(), loc(), outputType, input, offset, outputDynSymbols);
}

memref::SubViewOp MemRefBuilder::subview(Value val, ArrayRef<int64_t> offsets,
    ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides) const {
  return memref::SubViewOp::create(b(), loc(), val, offsets, sizes, strides);
}

memref::SubViewOp MemRefBuilder::subview(MemRefType outputType, Value val,
    ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> strides) const {
  return memref::SubViewOp::create(
      b(), loc(), outputType, val, offsets, sizes, strides);
}

memref::SubViewOp MemRefBuilder::subview(Value input,
    ArrayRef<IndexExpr> offsetsIE, ArrayRef<IndexExpr> sizesIE,
    ArrayRef<IndexExpr> stridesIE) const {
  SmallVector<OpFoldResult, 4> offsets, sizes, strides;
  IndexExpr::getOpOrFoldResults(offsetsIE, offsets);
  IndexExpr::getOpOrFoldResults(sizesIE, sizes);
  IndexExpr::getOpOrFoldResults(stridesIE, strides);
  SmallVector<int64_t, 4> outputShape;
  IndexExpr::getShape(sizesIE, outputShape);
  MemRefType inputType = mlir::dyn_cast<MemRefType>(input.getType());
  MemRefLayoutAttrInterface layout;
  MemRefType outputType = MemRefType::get(outputShape,
      inputType.getElementType(), layout, inputType.getMemorySpace());
  return memref::SubViewOp::create(
      b(), loc(), outputType, input, offsets, sizes, strides);
}

//===----------------------------------------------------------------------===//
// Dims.

Value MemRefBuilder::dim(Value val, int64_t index) const {
  assert(index >= 0 && "Expecting a valid index");
  return dim(val, arith::ConstantIndexOp::create(b(), loc(), index));
}

Value MemRefBuilder::dim(Value val, Value index) const {
  // assert((mlir::isa<MemRefType>(val.getType()) ||
  //           mlir::isa<UnrankedMemRefType>(val.getType())) &&
  //       "memref::DimOp expects input operand to have MemRefType or "
  //       "UnrankedMemRefType");
  return Value(b().createOrFold<memref::DimOp>(loc(), val, index));
}

//===----------------------------------------------------------------------===//
// Prefetch.

void MemRefBuilder::prefetch(Value memref, ValueRange indices, bool isWrite,
    unsigned locality, bool isData) {
  if (disableMemRefPrefetch)
    return;
  memref::PrefetchOp::create(
      b(), loc(), memref, indices, isWrite, locality, isData);
}

void MemRefBuilder::prefetchIE(Value memref, ArrayRef<IndexExpr> indices,
    bool isWrite, unsigned locality, bool isData) {
  if (disableMemRefPrefetch)
    return;
  SmallVector<Value, 4> indexVals;
  IndexExpr::getValues(indices, indexVals);
  prefetch(memref, indexVals, isWrite, locality, isData);
}

//===----------------------------------------------------------------------===//
// Queries

/*static*/ bool MemRefBuilder::isNoneValue(Value value) {
  return mlir::isa<NoneType>(value.getType());
}

/*static*/ bool MemRefBuilder::hasOneElementInInnermostDims(
    Value value, int64_t innerDim) {
  // Get info.
  ShapedType type = mlir::dyn_cast<ShapedType>(value.getType());
  assert(type && "expected shaped type");
  int64_t rank = type.getRank();
  ArrayRef<int64_t> shape = type.getShape();
  for (int64_t i = std::max((int64_t)0, rank - innerDim); i < rank; ++i)
    if (shape[i] != 1)
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF).
//===----------------------------------------------------------------------===//

void SCFBuilder::ifThenElse(
    Value cond, SCFThenElseBodyFn thenFn, SCFThenElseBodyFn elseFn) const {
  if (!elseFn) {
    scf::IfOp::create(b(), loc(), cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          thenFn(scfBuilder);
          yield();
        });
  } else {
    scf::IfOp::create(
        b(), loc(), cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          thenFn(scfBuilder);
          scf::YieldOp::create(b(), loc());
        },
        /*else*/
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          elseFn(scfBuilder);
          yield();
        });
  }
}

void SCFBuilder::forLoop(
    Value lb, Value ub, int64_t step, SCFLoopBodyFn bodyFn) const {
  MathBuilder createMath(*this);
  Value stepVal = createMath.constantIndex(step);
  scf::ForOp::create(b(), loc(), lb, ub, stepVal, llvm::ArrayRef<mlir::Value>(),
      [&](OpBuilder &childBuilder, Location childLoc, Value inductionVar,
          ValueRange args) {
        SCFBuilder builder(childBuilder, childLoc);
        bodyFn(builder, {inductionVar});
        yield();
      });
}

void SCFBuilder::forLoopIE(IndexExpr lb, IndexExpr ub, int64_t step,
    bool useParallel, SCFLoopBodyFn bodyFn) const {
  if (useParallel) {
    MathBuilder createMath(*this);
    Value stepVal = createMath.constantIndex(step);
    parallelLoops({lb.getValue()}, {ub.getValue()}, {stepVal}, bodyFn);
  } else {
    forLoop(lb.getValue(), ub.getValue(), step, bodyFn);
  }
}

void SCFBuilder::forLoopsIE(ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
    ArrayRef<int64_t> steps, ArrayRef<bool> useParallel,
    SCFLoopBodyFn builderFn) const {
  impl::forLoopsIE<SCFBuilder>(*this, lbs, ubs, steps, useParallel, builderFn);
}

void SCFBuilder::parallelLoops(ValueRange lbs, ValueRange ubs, ValueRange steps,
    SCFLoopBodyFn bodyFn) const {
  scf::ParallelOp::create(b(), loc(), lbs, ubs, steps,
      [&](OpBuilder &childBuilder, Location childLoc,
          ValueRange inductionVars) {
        SCFBuilder builder(childBuilder, childLoc);
        bodyFn(builder, inductionVars);
        yield();
      });
}

void SCFBuilder::yield() const { scf::YieldOp::create(b(), loc()); }

void SCFBuilder::simdIterateIE(IndexExpr lb, IndexExpr ub, int64_t VL,
    bool fullySimd, bool useParallel, ArrayRef<Value> inputs,
    ArrayRef<DimsExpr> inputAFs, ArrayRef<Value> outputs,
    ArrayRef<DimsExpr> outputAFs,
    ArrayRef<SCFSimdIterateBodyFn> bodyFnList) const {
  onnx_mlir::impl::simdIterateIE<SCFBuilder, MemRefBuilder>(*this, lb, ub, VL,
      fullySimd, useParallel, inputs, inputAFs, outputs, outputAFs, bodyFnList);
}

void SCFBuilder::simdReduceIE(IndexExpr lb, IndexExpr ub, int64_t VL,
    bool fullySimd, ArrayRef<Value> inputs, ArrayRef<DimsExpr> inputAFs,
    ArrayRef<Value> tmps, ArrayRef<DimsExpr> tmpAFs, ArrayRef<Value> outputs,
    ArrayRef<DimsExpr> outputAFs, ArrayRef<Value> initVals,
    /* reduction function (simd or scalar) */
    mlir::ArrayRef<SCFSimdReductionBodyFn> reductionFnList,
    /* post reduction function (simd to scalar + post processing)*/
    mlir::ArrayRef<SCFSimdPostReductionBodyFn> postReductionFnList) const {
  onnx_mlir::impl::simdReduceIE<SCFBuilder, MemRefBuilder>(*this, lb, ub, VL,
      fullySimd, inputs, inputAFs, tmps, tmpAFs, outputs, outputAFs, initVals,
      reductionFnList, postReductionFnList);
}

void SCFBuilder::simdReduce2DIE(IndexExpr lb, IndexExpr ub, int64_t VL,
    bool fullySimd, Value input, DimsExpr inputAF, Value tmp, DimsExpr tmpAF,
    Value output, DimsExpr outputAF, Value initVal,
    /* reduction functions (simd or scalar) */
    SCFSimdReductionBodyFn reductionBodyFn,
    /* post reduction functions (post processing ONLY)*/
    SCFSimdPostReductionBodyFn postReductionBodyFn) const {
  onnx_mlir::impl::simdReduce2DIE<SCFBuilder, MemRefBuilder>(*this, lb, ub, VL,
      fullySimd, input, inputAF, tmp, tmpAF, output, outputAF, initVal,
      reductionBodyFn, postReductionBodyFn);
}

//===----------------------------------------------------------------------===//
// Vector Builder
//===----------------------------------------------------------------------===//

/*static*/ bool VectorBuilder::compatibleShapes(const Type t1, const Type t2) {
  // If both are vectors, check that the shapes are identical.
  VectorType vt1 = mlir::dyn_cast<VectorType>(t1);
  VectorType vt2 = mlir::dyn_cast<VectorType>(t2);
  if (vt1 && vt2) {
    auto shape1 = vt1.getShape();
    auto shape2 = vt2.getShape();
    // Different rank, return false.
    if (shape1.size() != shape2.size())
      return false;
    for (int64_t i = 0; i < (int64_t)shape1.size(); ++i)
      if (shape1[i] != shape2[i])
        return false;
    // Same dim and shapes
    return true;
  }
  // Neither is a vector (no shape tests) or only one is a vector (and the other
  // one can thus be broadcasted to it), we have compatible shapes.
  return true;
}

/*static*/ bool VectorBuilder::compatibleTypes(const Type t1, const Type t2) {
  Type e1 = MathBuilder::elementTypeOfScalarOrVector(t1);
  Type e2 = MathBuilder::elementTypeOfScalarOrVector(t2);
  return (e1 == e2) && compatibleShapes(t1, t2);
}

int64_t VectorBuilder::getArchVectorLength(const Type &elementType) const {
  // Even if unsupported, we can always compute one result per vector.
  return std::max(
      (int64_t)1, VectorMachineSupport::getArchVectorLength(elementType));
}

int64_t VectorBuilder::getArchVectorLength(const VectorType &vecType) const {
  return getArchVectorLength(vecType.getElementType());
}

int64_t VectorBuilder::getArchVectorLength(Value vecValue) const {
  VectorType vecType = mlir::dyn_cast_or_null<VectorType>(vecValue.getType());
  assert(vecType && "expected vector type");
  return getArchVectorLength(vecType.getElementType());
}

Value VectorBuilder::load(VectorType vecType, Value memref, ValueRange indices,
    ValueRange offsets) const {
  // Cannot use the onnx_mlir::impl::load because we also need to pass the type.
  llvm::SmallVector<Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(*this);
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  return vector::LoadOp::create(b(), loc(), vecType, memref, computedIndices);
}

Value VectorBuilder::loadIE(VectorType vecType, Value memref,
    llvm::ArrayRef<IndexExpr> indices, ValueRange offsets) const {
  // Cannot use the onnx_mlir::impl::load because we also need to pass the type.
  llvm::SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  return load(vecType, memref, indexValues, offsets);
}

void VectorBuilder::store(
    Value val, Value memref, ValueRange indices, ValueRange offsets) const {
  onnx_mlir::impl::store<VectorBuilder, vector::StoreOp>(
      *this, val, memref, indices, offsets);
}

void VectorBuilder::storeIE(Value val, Value memref,
    llvm::ArrayRef<IndexExpr> indices, ValueRange offsets) const {
  onnx_mlir::impl::storeIE<VectorBuilder, vector::StoreOp>(
      *this, val, memref, indices, offsets);
}

Value VectorBuilder::fma(Value lhs, Value rhs, Value acc) const {
  return vector::FMAOp::create(b(), loc(), lhs, rhs, acc);
}

Value VectorBuilder::broadcast(VectorType vecType, Value val) const {
  return vector::BroadcastOp::create(b(), loc(), vecType, val);
}

Value VectorBuilder::shuffle(
    Value lhs, Value rhs, SmallVectorImpl<int64_t> &mask) const {
  return vector::ShuffleOp::create(b(), loc(), lhs, rhs, mask);
}

Value VectorBuilder::typeCast(Type resTy, Value val) const {
  return vector::TypeCastOp::create(b(), loc(), resTy, val);
}

// Private vector utilities.
bool VectorBuilder::isPowerOf2(uint64_t num) const {
  return (num & (num - 1)) == 0;
}

uint64_t VectorBuilder::getLengthOf1DVector(Value vec) const {
  VectorType vecType = mlir::dyn_cast_or_null<VectorType>(vec.getType());
  assert(vecType && "expected a vector type");
  auto vecShape = vecType.getShape();
  assert(vecShape.size() == 1 && "expected a 1D vector");
  return vecShape[0];
}

Value VectorBuilder::mergeHigh(Value lhs, Value rhs, int64_t step) const {
  // Inputs: lrs <l0, l1, l2, l3, l4, l5, l6, l7>;
  //         rhs <r0, r1, r2, r3, r4, r5, r6, r7>.
  // Merge alternatively the low (least significant) values of lrs and rhs
  // Step 1:     <(l0), (r0), (l1), (r1), (l2), (r2), (l3), (r3)> (1x sizes)
  // Step 2:     <(l0, l1),   (r0, r1),   (l2, l3),   (r2, r3)>   (2x sizes)
  // Step 4:     <(l0, l1, l2, l3),       (r0, r1, r2, r3)>       (4x sizes)
  uint64_t VL = getLengthOf1DVector(lhs);
  assert(getLengthOf1DVector(rhs) == VL && "expected same sized vectors");
  assert(isPowerOf2(VL) && "expected power of 2 vector length");
  SmallVector<int64_t, 8> mask(VL, 0);
  int i = 0;
  int64_t pairsOfLhsRhs = VL / (2 * step);
  int64_t firstHalf = 0;
  for (int64_t p = 0; p < pairsOfLhsRhs; ++p) {
    // One step-sized item from the LHS
    for (int64_t e = 0; e < step; ++e)
      mask[i++] = firstHalf + p * step + e;
    // One step-sized item from the RHS (RHS offset is VL for the shuffle op).
    for (int64_t e = 0; e < step; ++e)
      mask[i++] = firstHalf + VL + p * step + e;
  }
  return shuffle(lhs, rhs, mask);
}

Value VectorBuilder::mergeLow(Value lhs, Value rhs, int64_t step) const {
  // Inputs: lrs <l0, l1, l2, l3, l4, l5, l6, l7>;
  //         rhs <r0, r1, r2, r3, r4, r5, r6, r7>.
  // Merge alternatively the low (least significant) values of lrs and rhs
  // Step 1:     <(l4), (r4), (l5), (r5), (l6), (r6), (l7), (r7)> (1x sizes)
  // Step 2:     <(l4, l5),   (r4, r5),   (l6, l7),   (r6, r7)>   (2x sizes)
  // Step 4:     <(l4, l5, l6, l7),       (r4, r5, r6, r7)>       (4x sizes)
  uint64_t VL = getLengthOf1DVector(lhs);
  assert(getLengthOf1DVector(rhs) == VL && "expected same sized vectors");
  assert(isPowerOf2(VL) && "expected power of 2 vector length");
  SmallVector<int64_t, 8> mask(VL, 0);
  int i = 0;
  int64_t pairsOfLhsRhs = VL / (2 * step);
  int64_t secondHalf = VL / 2;
  for (int64_t p = 0; p < pairsOfLhsRhs; ++p) {
    // One step-sized item from the LHS
    for (int64_t e = 0; e < step; ++e)
      mask[i++] = secondHalf + p * step + e;
    // One step-sized item from the RHS (RHS offset is VL for the shuffle op).
    for (int64_t e = 0; e < step; ++e)
      mask[i++] = secondHalf + VL + p * step + e;
  }
  return shuffle(lhs, rhs, mask);
}

Value VectorBuilder::reduction(
    VectorBuilder::CombiningKind kind, Value value) const {
  Type type = value.getType();
  switch (kind) {
  case CombiningKind::ADD: {
    return vector::ReductionOp::create(
        b(), loc(), vector::CombiningKind::ADD, value);
  }
  case CombiningKind::MUL: {
    return vector::ReductionOp::create(
        b(), loc(), vector::CombiningKind::MUL, value);
  }
  case CombiningKind::MAX: {
    if (MathBuilder::isScalarOrVectorUnsignedInteger(type))
      return vector::ReductionOp::create(
          b(), loc(), vector::CombiningKind::MAXUI, value);
    if (MathBuilder::isScalarOrVectorInteger(type))
      return vector::ReductionOp::create(
          b(), loc(), vector::CombiningKind::MAXSI, value);
    if (MathBuilder::isScalarOrVectorFloat(type))
      return vector::ReductionOp::create(
          b(), loc(), vector::CombiningKind::MAXNUMF, value);
    llvm_unreachable("unknown type in max");
  }
  case CombiningKind::MIN: {
    if (MathBuilder::isScalarOrVectorUnsignedInteger(type))
      return vector::ReductionOp::create(
          b(), loc(), vector::CombiningKind::MINUI, value);
    if (MathBuilder::isScalarOrVectorInteger(type))
      return vector::ReductionOp::create(
          b(), loc(), vector::CombiningKind::MINSI, value);
    if (MathBuilder::isScalarOrVectorFloat(type))
      return vector::ReductionOp::create(
          b(), loc(), vector::CombiningKind::MINNUMF, value);
    llvm_unreachable("unknown type in min");
  }
  case CombiningKind::AND: {
    if (MathBuilder::isScalarOrVectorInteger(type))
      return vector::ReductionOp::create(
          b(), loc(), vector::CombiningKind::AND, value);
    llvm_unreachable("unknown type in and");
  }
  case CombiningKind::OR: {
    if (MathBuilder::isScalarOrVectorInteger(type))
      return vector::ReductionOp::create(
          b(), loc(), vector::CombiningKind::OR, value);
    llvm_unreachable("unknown type in or");
  }
  case CombiningKind::XOR: {
    if (MathBuilder::isScalarOrVectorInteger(type))
      return vector::ReductionOp::create(
          b(), loc(), vector::CombiningKind::XOR, value);
    llvm_unreachable("unknown type in xor");
  }
  } // Switch.
  llvm_unreachable("unknown combining kind");
}

// Do a parallel-simd reduction of N vectors of SIMD length VL.
// Restrictions:
// *  VL is the vector length of the machine SIMD vectors.
// *  N is a multiple of VL as we can perform consecutive VL x VL
//    reductions.
// For example, when we passe N=VL input vectors, the output has one vector;
// when we passe N=2VL input vectors, the output has 2 vectors...

void VectorBuilder::multiReduction(ArrayRef<Value> inputVecArray,
    F2 reductionFct, SmallVectorImpl<Value> &outputVecArray) {
  uint64_t N = inputVecArray.size();
  assert(N > 0 && "expected at least one value to reduce");
  uint64_t VL = getLengthOf1DVector(inputVecArray[0]);
  uint64_t archVL = getArchVectorLength(inputVecArray[0]);
  // TODO alex, should relax this
  assert(VL == archVL && "only natural sizes supported at this time");
  assert(N % archVL == 0 &&
         "can only reduces multiple of VL vectors at this time");
  LLVM_DEBUG(llvm::dbgs() << "reduction with N " << N << ", VL " << VL
                          << ", archVL " << archVL << "\n";);

  // Emplace all input vectors in a temporary array.
  SmallVector<Value, 8> tmpArray;
  for (uint64_t i = 0; i < N; ++i) {
    tmpArray.emplace_back(inputVecArray[i]);
    // Also verify that all have the same vector length.
    assert(getLengthOf1DVector(inputVecArray[i]) == VL &&
           "different vector length");
  }

  // Reductions of full physical vectors.
  outputVecArray.clear();
  MultiDialectBuilder<MathBuilder> create(*this);
  // Process each block of archVL input vectors at a time.
  for (uint64_t r = 0; r < N; r += archVL) {
    // Algorithm for the set of input arrays from tmp[r] to
    // tmp[r+archVL-1].
    // With archVL inputs, we have archVL/2 initial pairs.
    uint64_t numPairs = archVL / 2;
    // While we have pairs...
    for (uint64_t step = 1; step < archVL; step = step * 2) {
      // For each pair, reduce pair 2p and 2p+1 and save sum into p.
      for (uint64_t p = 0; p < numPairs; ++p) {
        Value highVal =
            mergeHigh(tmpArray[r + 2 * p], tmpArray[r + 2 * p + 1], step);
        Value lowVal =
            mergeLow(tmpArray[r + 2 * p], tmpArray[r + 2 * p + 1], step);
        Value red = reductionFct(highVal, lowVal);
        tmpArray[r + p] = red;
      }
      numPairs = numPairs / 2; // Pair number decrease by power of 2.
    }
    // Completed the archVL x archVL reduction, save it in the output.
    outputVecArray.emplace_back(tmpArray[r]);
  }
}

// Cast vectors to vectors of different shape (e.g. 1D to 2D and back).
Value VectorBuilder::shapeCast(VectorType newType, Value vector) const {
  return vector::ShapeCastOp::create(b(), loc(), newType, vector);
}

// Extract  1D vector from 2D vector.
Value VectorBuilder::extractFrom2D(Value vector2D, int64_t position) const {
  llvm::SmallVector<int64_t> pos = {position};
  return vector::ExtractOp::create(b(), loc(), vector2D, pos);
}

// Insert 1D vector into 2D vector.
Value VectorBuilder::insertInto2D(
    Value vector, Value vector2D, int64_t position) const {
  llvm::SmallVector<int64_t> pos = {position};
  return vector::InsertOp::create(b(), loc(), vector, vector2D, pos);
}

Value VectorBuilder::extractElement(Value vector, int64_t index) const {
  MultiDialectBuilder<VectorBuilder, MathBuilder> create(*this);
  VectorType type = llvm::cast<VectorType>(vector.getType());
  int64_t VL = type.getShape()[0];
  assert(type.getRank() == 1 && "expected 1D vector only");
  assert(index >= 0 && index < VL && "out of range vector index");
  Value position = create.math.constantIndex(index);
  return vector::ExtractOp::create(b(), loc(), vector, position);
}

Value VectorBuilder::insertElement(
    Value vector, Value element, int64_t index) const {
  MultiDialectBuilder<VectorBuilder, MathBuilder> create(*this);
  VectorType type = llvm::cast<VectorType>(vector.getType());
  int64_t VL = type.getShape()[0];
  assert(type.getRank() == 1 && "expected 1D vector only");
  assert(index >= 0 && index < VL && "out of range vector index");
  Value position = create.math.constantIndex(index);
  // Unlike LLVM insert element which takes <dest, source, position>, vector
  // take <source, dest, position>
  return vector::InsertOp::create(b(), loc(), element, vector, position);
}

//===----------------------------------------------------------------------===//
// LLVM Builder
//===----------------------------------------------------------------------===//

Value LLVMBuilder::add(Value lhs, Value rhs) const {
  return LLVM::AddOp::create(b(), loc(), lhs, rhs);
}

Value LLVMBuilder::addressOf(LLVM::GlobalOp op) const {
  return LLVM::AddressOfOp::create(b(), loc(), op);
}

Value LLVMBuilder::_alloca(
    Type resultType, Type elementType, Value size, int64_t alignment) const {
  return LLVM::AllocaOp::create(
      b(), loc(), resultType, elementType, size, alignment);
}

Value LLVMBuilder::andi(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return LLVM::AndOp::create(b(), loc(), lhs, rhs);
}

Value LLVMBuilder::bitcast(Type type, Value val) const {
  return LLVM::BitcastOp::create(b(), loc(), type, val);
}

void LLVMBuilder::br(ArrayRef<Value> destOperands, Block *destBlock) const {
  LLVM::BrOp::create(b(), loc(), destOperands, destBlock);
}

void LLVMBuilder::handleVarArgCall(LLVM::CallOp &callOp,
    ArrayRef<Type> resultTypes, ArrayRef<Value> inputs) const {
  // Define result type (void or 1).
  Type resultType;
  if (resultTypes.size() == 0 || isa<LLVM::LLVMVoidType>(resultTypes[0])) {
    MLIRContext *ctx = b().getContext();
    resultType = LLVM::LLVMVoidType::get(ctx);
  } else {
    resultType = resultTypes[0];
  }
  // Define input types.
  llvm::SmallVector<Type, 4> inputTypes;
  for (int64_t i = 0; i < (int64_t)inputs.size(); ++i)
    inputTypes.emplace_back(inputs[i].getType());
  auto typeSignature =
      LLVM::LLVMFunctionType::get(resultType, inputTypes, /*is var arg*/ true);
  callOp.setVarCalleeType(typeSignature);
}

Value LLVMBuilder::call(ArrayRef<Type> resultTypes, StringRef funcName,
    ArrayRef<Value> inputs, bool isVarArg) const {
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  LLVM::CallOp callOp =
      LLVM::CallOp::create(b(), loc(), resultTypes, funcName, inputs);
  if (isVarArg)
    handleVarArgCall(callOp, resultTypes, inputs);
  // CallOp may return either 0 or 1 value.
  if (resultTypes.empty())
    return nullptr;
  return callOp.getResult();
}

Value LLVMBuilder::call(ArrayRef<Type> resultTypes,
    FlatSymbolRefAttr funcSymbol, ArrayRef<Value> inputs, bool isVarArg) const {
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  LLVM::CallOp callOp =
      LLVM::CallOp::create(b(), loc(), resultTypes, funcSymbol, inputs);
  if (isVarArg)
    handleVarArgCall(callOp, resultTypes, inputs);
  // CallOp may return either 0 or 1 value.
  if (resultTypes.empty())
    return nullptr;
  return callOp.getResult();
}

void LLVMBuilder::condBr(Value cond, Block *trueBlock,
    llvm::ArrayRef<Value> trueOperands, Block *falseBlock,
    llvm::ArrayRef<Value> falseOperands) const {
  LLVM::CondBrOp::create(
      b(), loc(), cond, trueBlock, trueOperands, falseBlock, falseOperands);
}

Value LLVMBuilder::constant(Type type, int64_t val) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<IntegerType>([&](IntegerType type) {
        unsigned width = type.getWidth();
        if (width == 1)
          constant = LLVM::ConstantOp::create(
              b(), loc(), type, b().getBoolAttr(val != 0));
        else {
          assert(type.isSignless() &&
                 "LLVM::ConstantOp requires a signless type.");
          constant = LLVM::ConstantOp::create(b(), loc(), type,
              b().getIntegerAttr(
                  type, APInt(width, static_cast<int64_t>(val), false, true)));
        }
      })
      .Case<IndexType>([&](Type) {
        constant = LLVM::ConstantOp::create(
            b(), loc(), type, b().getIntegerAttr(type, val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value LLVMBuilder::constant(Type type, double val) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<Float16Type>([&](Type) {
        constant = LLVM::ConstantOp::create(
            b(), loc(), type, b().getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant = LLVM::ConstantOp::create(
            b(), loc(), type, b().getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant = LLVM::ConstantOp::create(
            b(), loc(), type, b().getF64FloatAttr(val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value LLVMBuilder::extractElement(
    Type resultType, Value container, int64_t position) const {
  Value posVal = constant(b().getI64Type(), position);
  return LLVM::ExtractElementOp::create(
      b(), loc(), resultType, container, posVal);
}

Value LLVMBuilder::extractValue(
    Type resultType, Value container, ArrayRef<int64_t> position) const {
  return LLVM::ExtractValueOp::create(
      b(), loc(), resultType, container, position);
}

LLVM::LLVMFuncOp LLVMBuilder::func(
    StringRef funcName, Type funcType, bool createUniqueFunc) const {
  // If createUniqueFunc, we create two functions: name and name_postfix.
  // They have the same signatures and `name` will call `name_postfix`.
  // `name_postfix` function is expected to be unique across all generated
  // modules, allowing to run multiple models at the same time.
  LLVM::LLVMFuncOp funcOp =
      LLVM::LLVMFuncOp::create(b(), loc(), funcName, funcType);
  if (!createUniqueFunc)
    return funcOp;

  // Create uniqueFuncOp if there exists a postfix.
  // Since `funcOp` calls `uniqueFuncOp`, put `uniqueFuncOp`'s definition
  // before `funcOp`.
  b().setInsertionPoint(funcOp);
  ModuleOp module = funcOp.getOperation()->getParentOfType<ModuleOp>();
  std::string uniqueFuncName =
      LLVMBuilder::SymbolPostfix(module, funcName.str());
  if (uniqueFuncName == funcName.str())
    return funcOp;

  auto uniqueFuncType = cast<LLVM::LLVMFunctionType>(funcType);
  LLVM::LLVMFuncOp uniqueFuncOp =
      LLVM::LLVMFuncOp::create(b(), loc(), uniqueFuncName, uniqueFuncType);

  // Call uniqueFuncOp inside funcOp.
  Block *entryBlock = funcOp.addEntryBlock(b());
  OpBuilder::InsertionGuard bodyGuard(b());
  b().setInsertionPointToStart(entryBlock);
  ValueRange args = entryBlock->getArguments();
  TypeRange resultTypes = uniqueFuncType.getReturnTypes();
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  if (resultTypes.size() == 0 || isa<LLVM::LLVMVoidType>(resultTypes[0])) {
    LLVM::CallOp::create(b(), loc(), ArrayRef<Type>({}), uniqueFuncName, args);
    LLVM::ReturnOp::create(b(), loc(), ArrayRef<Value>({}));
  } else {
    LLVM::CallOp callOp =
        LLVM::CallOp::create(b(), loc(), resultTypes, uniqueFuncName, args);
    LLVM::ReturnOp::create(b(), loc(), ArrayRef<Value>({callOp.getResult()}));
  }

  return uniqueFuncOp;
}

Value LLVMBuilder::getElemPtr(Type resultType, Type elemType, Value base,
    ArrayRef<LLVM::GEPArg> indices) const {
  return LLVM::GEPOp::create(b(), loc(), resultType, elemType, base, indices);
}

LLVM::GlobalOp LLVMBuilder::globalOp(Type resultType, bool isConstant,
    LLVM::Linkage linkage, StringRef name, Attribute valueAttr,
    uint64_t alignment, bool uniqueName) const {
  LLVM::GlobalOp gop = LLVM::GlobalOp::create(b(), loc(), resultType,
      /*isConstant=*/isConstant, linkage, name, valueAttr);
  if (!uniqueName)
    return gop;

  // Append to `name` a unique string to make it unique across multiple
  // generated LLVMIR.
  ModuleOp module = gop.getOperation()->getParentOfType<ModuleOp>();
  gop.setName(LLVMBuilder::SymbolPostfix(module, name.str()));
  return gop;
}

Value LLVMBuilder::icmp(LLVM::ICmpPredicate cond, Value lhs, Value rhs) const {
  return LLVM::ICmpOp::create(b(), loc(), cond, lhs, rhs);
}

Value LLVMBuilder::insertElement(Value vec, Value val, int64_t position) const {
  Value posVal = constant(b().getI64Type(), position);
  return LLVM::InsertElementOp::create(b(), loc(), vec, val, posVal);
}

Value LLVMBuilder::insertValue(Type resultType, Value container, Value val,
    llvm::ArrayRef<int64_t> position) const {
  return LLVM::InsertValueOp::create(
      b(), loc(), resultType, container, val, position);
}

Value LLVMBuilder::inttoptr(Type type, Value val) const {
  return LLVM::IntToPtrOp::create(b(), loc(), type, val);
}

Value LLVMBuilder::lshr(Value lhs, Value rhs) const {
  return LLVM::LShrOp::create(b(), loc(), lhs, rhs);
}

Value LLVMBuilder::load(Type elementType, Value addr) const {
  return LLVM::LoadOp::create(b(), loc(), elementType, addr);
}

Value LLVMBuilder::mul(Value lhs, Value rhs) const {
  return LLVM::MulOp::create(b(), loc(), lhs, rhs);
}

Value LLVMBuilder::null(Type type) const {
  return LLVM::ZeroOp::create(b(), loc(), type);
}

Value LLVMBuilder::ori(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return LLVM::OrOp::create(b(), loc(), lhs, rhs);
}

Value LLVMBuilder::ptrtoint(Type type, Value val) const {
  return LLVM::PtrToIntOp::create(b(), loc(), type, val);
}

void LLVMBuilder::_return() const {
  LLVM::ReturnOp::create(b(), loc(), ArrayRef<Value>({}));
}

void LLVMBuilder::_return(Value val) const {
  LLVM::ReturnOp::create(b(), loc(), ArrayRef<Value>({val}));
}

Value LLVMBuilder::select(Value cmp, Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return LLVM::SelectOp::create(b(), loc(), cmp, lhs, rhs);
}

Value LLVMBuilder::sext(Type type, Value val) const {
  return LLVM::SExtOp::create(b(), loc(), type, val);
}

Value LLVMBuilder::shl(Value lhs, Value rhs) const {
  return LLVM::ShlOp::create(b(), loc(), lhs, rhs);
}

void LLVMBuilder::store(Value val, Value addr) const {
  LLVM::StoreOp::create(b(), loc(), val, addr);
}

Value LLVMBuilder::trunc(Type type, Value val) const {
  return LLVM::TruncOp::create(b(), loc(), type, val);
}

Value LLVMBuilder::zext(Type type, Value val) const {
  return LLVM::ZExtOp::create(b(), loc(), type, val);
}

FlatSymbolRefAttr LLVMBuilder::getOrInsertSymbolRef(ModuleOp module,
    StringRef funcName, Type resultType, ArrayRef<Type> operandTypes,
    bool isVarArg) const {
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    OpBuilder::InsertionGuard guard(b());
    b().setInsertionPointToStart(module.getBody());
    LLVM::LLVMFunctionType funcType =
        LLVM::LLVMFunctionType::get(resultType, operandTypes, isVarArg);
    LLVM::LLVMFuncOp::create(b(), module.getLoc(), funcName, funcType);
  }
  return SymbolRefAttr::get(b().getContext(), funcName);
}

void LLVMBuilder::ifThenElse(
    valueFuncRef cond, voidFuncRef thenFn, voidFuncRef elseFn) const {
  LLVMBuilder createLLVM(b(), loc());

  // Split the current block into IF, THEN, ELSE and END blocks.
  Block *ifBlock, *thenBlock, *elseBlock, *endBlock;
  ifBlock = b().getInsertionBlock();
  endBlock = ifBlock->splitBlock(b().getInsertionPoint());
  thenBlock = b().createBlock(
      ifBlock->getParent(), std::next(Region::iterator(ifBlock)));
  if (elseFn)
    elseBlock = b().createBlock(
        thenBlock->getParent(), std::next(Region::iterator(thenBlock)));
  else
    elseBlock = endBlock;

  // Emit code for the IF block.
  b().setInsertionPointToEnd(ifBlock);
  Value condVal = cond(createLLVM);

  // Branch the block into the THEN and ELSE blocks.
  createLLVM.condBr(condVal, thenBlock, {}, elseBlock, {});

  // Emit code for the THEN block.
  b().setInsertionPointToStart(thenBlock);
  thenFn(createLLVM);
  if (thenBlock->hasNoSuccessors() && !isa<LLVM::ReturnOp>(thenBlock->back()))
    br({}, endBlock);

  // Emit code for the ELSE block if required.
  b().setInsertionPointToStart(elseBlock);
  if (elseFn) {
    elseFn(createLLVM);
    if (elseBlock->hasNoSuccessors() && !isa<LLVM::ReturnOp>(elseBlock->back()))
      br({}, endBlock);
  }

  // End if-then-else and return to the main body.
  b().setInsertionPointToStart(endBlock);
}

} // namespace onnx_mlir
