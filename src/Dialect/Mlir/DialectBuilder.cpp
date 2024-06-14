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

/* static */ bool MathBuilder::isVector(Type type) {
  return mlir::dyn_cast<VectorType>(type) != nullptr;
}

/* static */ Type MathBuilder::elementTypeWithVector(Type elementOrVectorType) {
  VectorType vectorType = mlir::dyn_cast<VectorType>(elementOrVectorType);
  if (vectorType)
    return vectorType.getElementType();
  return elementOrVectorType;
}

/* static */ Type MathBuilder::getTypeWithVector(
    VectorType vectorType, Type elementType) {
  if (vectorType)
    return VectorType::get(vectorType.getShape(), elementType);
  return elementType;
}

/* static */ bool MathBuilder::isIntegerWithVector(Type elementOrVectorType) {
  Type elementType = elementTypeWithVector(elementOrVectorType);
  return mlir::isa<IntegerType>(elementType) ||
         mlir::isa<IndexType>(elementType);
}

/* static */ bool MathBuilder::isUnsignedIntegerWithVector(
    Type elementOrVectorType) {
  Type elementType = elementTypeWithVector(elementOrVectorType);
  return elementType.isUnsignedInteger();
}

/* static */ bool MathBuilder::isFloatWithVector(Type elementOrVectorType) {
  Type elementType = elementTypeWithVector(elementOrVectorType);
  return mlir::isa<FloatType>(elementType);
}

Value MathBuilder::abs(Value val) const {
  if (isIntegerWithVector(val.getType()))
    return b().create<math::AbsIOp>(loc(), val);
  if (isFloatWithVector(val.getType()))
    return b().create<math::AbsFOp>(loc(), val);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::andi(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return b().create<arith::AndIOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::ori(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return b().create<arith::OrIOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::xori(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return b().create<arith::XOrIOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::add(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType())) {
    Type elemType = elementTypeWithVector(lhs.getType());
    if (elemType.isUnsignedInteger()) {
      unsigned elemWidth = mlir::cast<IntegerType>(elemType).getWidth();
      Value castLhs = castToSignless(lhs, elemWidth);
      Value castRhs = castToSignless(rhs, elemWidth);
      Value castAdd =
          b().create<arith::AddUIExtendedOp>(loc(), castLhs, castRhs).getSum();
      return castToUnsigned(castAdd, elemWidth);
    } else
      return b().create<arith::AddIOp>(loc(), lhs, rhs);
  }
  if (isFloatWithVector(lhs.getType()))
    return b().create<arith::AddFOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sub(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return b().create<arith::SubIOp>(loc(), lhs, rhs);
  if (isFloatWithVector(lhs.getType()))
    return b().create<arith::SubFOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::mul(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType())) {
    Type elemType = elementTypeWithVector(lhs.getType());
    if (elemType.isUnsignedInteger()) {
      unsigned elemWidth = mlir::cast<IntegerType>(elemType).getWidth();
      Value castLhs = castToSignless(lhs, elemWidth);
      Value castRhs = castToSignless(rhs, elemWidth);
      Value castMul =
          b().create<arith::MulUIExtendedOp>(loc(), castLhs, castRhs).getLow();
      return castToUnsigned(castMul, elemWidth);
    } else
      return b().create<arith::MulIOp>(loc(), lhs, rhs);
  }
  if (isFloatWithVector(lhs.getType()))
    return b().create<arith::MulFOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::div(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isFloatWithVector(lhs.getType()))
    return b().create<arith::DivFOp>(loc(), lhs, rhs);
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return b().create<arith::DivUIOp>(loc(), lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return b().create<arith::DivSIOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::rem(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isFloatWithVector(lhs.getType()))
    return b().create<arith::RemFOp>(loc(), lhs, rhs);
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return b().create<arith::RemUIOp>(loc(), lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return b().create<arith::RemSIOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::copySign(mlir::Value rem, mlir::Value dividend) const {
  assert(rem.getType() == dividend.getType() && "expected same type");
  if (isFloatWithVector(rem.getType()))
    return b().create<math::CopySignOp>(loc(), rem, dividend);
  llvm_unreachable("expected float");
}

Value MathBuilder::ceilDiv(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return b().create<arith::CeilDivUIOp>(loc(), lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return b().create<arith::CeilDivSIOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int");
}

Value MathBuilder::floorDiv(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    // Using regular unsigned div is ok as it rounds toward zero.
    return b().create<arith::DivUIOp>(loc(), lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return b().create<arith::FloorDivSIOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int");
}

// return (lhs * rhs) + acc
Value MathBuilder::fma(Value lhs, Value rhs, Value acc) const {
  assert((lhs.getType() == rhs.getType()) && (rhs.getType() == acc.getType()) &&
         "expected same type");
  if (isFloatWithVector(lhs.getType()) && !isa<FloatType>(lhs.getType()))
    return b().create<vector::FMAOp>(loc(), lhs, rhs, acc);
  return add(mul(lhs, rhs), acc);
}

Value MathBuilder::erf(Value val) const {
  return b().create<math::ErfOp>(loc(), val);
}

Value MathBuilder::exp(Value val) const {
  if (isFloatWithVector(val.getType()))
    return b().create<math::ExpOp>(loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::exp2(Value val) const {
  if (isFloatWithVector(val.getType()))
    return b().create<math::Exp2Op>(loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::log(Value val) const {
  if (isFloatWithVector(val.getType()))
    return b().create<math::LogOp>(loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::log2(Value val) const {
  if (isFloatWithVector(val.getType()))
    return b().create<math::Log2Op>(loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::sqrt(Value val) const {
  if (isFloatWithVector(val.getType()))
    return b().create<math::SqrtOp>(loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::pow(Value base, Value exp) const {
  if (isFloatWithVector(base.getType()))
    return b().create<math::PowFOp>(loc(), base, exp);
  llvm_unreachable("expected base float");
}

Value MathBuilder::neg(Value val) const {
  if (isIntegerWithVector(val.getType()))
    // Returns 0 - val.
    return sub(constant(val.getType(), 0), val);
  if (isFloatWithVector(val.getType()))
    return b().create<arith::NegFOp>(loc(), val);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::ceil(Value val) const {
  if (isFloatWithVector(val.getType()))
    return b().create<math::CeilOp>(loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::floor(Value val) const {
  if (isFloatWithVector(val.getType()))
    return b().create<math::FloorOp>(loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::tanh(Value val) const {
  if (isFloatWithVector(val.getType()))
    return b().create<math::TanhOp>(loc(), val);
  llvm_unreachable("expected float");
}

Value MathBuilder::min(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isFloatWithVector(lhs.getType()))
    return b().create<arith::MinNumFOp>(loc(), lhs, rhs);
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return b().create<arith::MinUIOp>(loc(), lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return b().create<arith::MinSIOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::max(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isFloatWithVector(lhs.getType()))
    return b().create<arith::MaxNumFOp>(loc(), lhs, rhs);
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return b().create<arith::MaxUIOp>(loc(), lhs, rhs);
  if (isIntegerWithVector(lhs.getType()))
    return b().create<arith::MaxSIOp>(loc(), lhs, rhs);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sgt(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sgt);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OGT);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sge(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sge);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OGE);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::slt(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::slt);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OLT);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::sle(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sle);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OLE);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::ugt(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ugt);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::uge(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::uge);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::ult(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ult);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::ule(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ule);
  llvm_unreachable("expected unsigned int");
}

Value MathBuilder::gt(Value lhs, Value rhs) const {
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return ugt(lhs, rhs);
  return sgt(lhs, rhs);
}

Value MathBuilder::ge(Value lhs, Value rhs) const {
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return uge(lhs, rhs);
  return sge(lhs, rhs);
}

Value MathBuilder::lt(Value lhs, Value rhs) const {
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return ult(lhs, rhs);
  return slt(lhs, rhs);
}

Value MathBuilder::le(Value lhs, Value rhs) const {
  if (isUnsignedIntegerWithVector(lhs.getType()))
    return ule(lhs, rhs);
  return sle(lhs, rhs);
}

Value MathBuilder::eq(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::eq);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::OEQ);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::neq(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (isIntegerWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ne);
  if (isFloatWithVector(lhs.getType()))
    return createArithCmp(lhs, rhs, arith::CmpFPredicate::ONE);
  llvm_unreachable("expected int or float");
}

Value MathBuilder::select(Value cmp, Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return b().create<arith::SelectOp>(loc(), cmp, lhs, rhs);
}

Value MathBuilder::constant(Type type, double val) const {
  Value constant = nullptr;
  // Could be a vector type; look at the element type.
  Type elementType = elementTypeWithVector(type);
  TypeSwitch<Type>(elementType)
      .Case<Float16Type>([&](Type) {
        constant =
            b().create<arith::ConstantOp>(loc(), b().getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant =
            b().create<arith::ConstantOp>(loc(), b().getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant =
            b().create<arith::ConstantOp>(loc(), b().getF64FloatAttr(val));
      })
      .Case<IntegerType>([&](IntegerType elementType) {
        assert(val == (int64_t)val && "value is ambiguous");
        unsigned width = elementType.getWidth();

        if (width == 1)
          constant =
              b().create<arith::ConstantOp>(loc(), b().getBoolAttr(val != 0));
        else {
          // If unsigned, create a signless constant, then cast it to unsigned.
          if (elementType.isUnsignedInteger()) {
            Type signlessTy = b().getIntegerType(width);
            constant = b().create<arith::ConstantOp>(loc(),
                b().getIntegerAttr(signlessTy, APInt(width, (int64_t)val)));
            constant = castToUnsigned(constant, width);
          } else {
            constant = b().create<arith::ConstantOp>(loc(),
                b().getIntegerAttr(elementType, APInt(width, (int64_t)val)));
          }
        }
      })
      .Case<IndexType>([&](Type elementType) {
        constant = b().create<arith::ConstantOp>(
            loc(), b().getIntegerAttr(elementType, val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  if (mlir::isa<VectorType>(type)) {
    // For vectors, need to splat the constant.
    MultiDialectBuilder<VectorBuilder> create(*this);
    VectorType vecType = mlir::dyn_cast<VectorType>(type);
    constant = create.vec.splat(vecType, constant);
  }
  return constant;
}

Value MathBuilder::constantIndex(int64_t val) const {
  IntegerAttr constantAttr = b().getIntegerAttr(b().getIndexType(), val);
  return b().create<arith::ConstantOp>(loc(), constantAttr);
}

TypedAttr MathBuilder::negativeInfAttr(mlir::Type type) const {
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
        attr = b().getIntegerAttr(type, APInt(width, value));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });
  assert(attr != nullptr && "Expecting valid attribute");
  return attr;
}

TypedAttr MathBuilder::positiveInfAttr(mlir::Type type) const {
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
        attr = b().getIntegerAttr(type, APInt(width, value));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });
  assert(attr != nullptr && "Expecting valid attribute");
  return attr;
}

Value MathBuilder::negativeInf(Type type) const {
  // Strip vector type if any.
  Type elementType = elementTypeWithVector(type);
  TypedAttr attr = negativeInfAttr(elementType);
  Value constant = b().create<arith::ConstantOp>(loc(), attr);
  assert(constant != nullptr && "Expecting valid constant value");
  if (mlir::isa<VectorType>(type)) {
    // For vectors, need to splat the constant.
    MultiDialectBuilder<VectorBuilder> create(*this);
    VectorType vecType = mlir::dyn_cast<VectorType>(type);
    constant = create.vec.splat(vecType, constant);
  }
  return constant;
}

Value MathBuilder::positiveInf(Type type) const {
  // Strip vector type if any.
  Type elementType = elementTypeWithVector(type);
  TypedAttr attr = positiveInfAttr(elementType);
  Value constant = b().create<arith::ConstantOp>(loc(), attr);
  assert(constant != nullptr && "Expecting valid constant value");
  if (mlir::isa<VectorType>(type)) {
    // For vectors, need to splat the constant.
    MultiDialectBuilder<VectorBuilder> create(*this);
    VectorType vecType = mlir::dyn_cast<VectorType>(type);
    constant = create.vec.splat(vecType, constant);
  }
  return constant;
}

Value MathBuilder::createArithCmp(
    Value lhs, Value rhs, arith::CmpIPredicate pred) const {
  Type type = lhs.getType();
  assert(type == rhs.getType() && "Operands should have the same type");
  assert(isIntegerWithVector(type) && "expected int");
  return b().create<arith::CmpIOp>(loc(), pred, lhs, rhs);
}

Value MathBuilder::createArithCmp(
    Value lhs, Value rhs, arith::CmpFPredicate pred) const {
  Type type = lhs.getType();
  assert(type == rhs.getType() && "Operands should have the same type");
  assert(isFloatWithVector(type) && "expected float");
  return b().create<arith::CmpFOp>(loc(), pred, lhs, rhs);
}

// Several operations in the arith dialect require signless integers. This
// cast remove the sign of integer types for successful processing, to the
// best of my understanding.
Value MathBuilder::castToSignless(Value val, int64_t width) const {
  Type valType = val.getType();
  VectorType vecType = mlir::dyn_cast<VectorType>(valType);
  Type valElemType = elementTypeWithVector(valType);
  assert(mlir::isa<IntegerType>(valElemType) &&
         !valElemType.isSignlessInteger() && "Expecting signed integer type");
  Type destType = getTypeWithVector(vecType, b().getIntegerType(width));
  return b()
      .create<UnrealizedConversionCastOp>(loc(), destType, val)
      .getResult(0);
}

Value MathBuilder::castToUnsigned(Value val, int64_t width) const {
  Type valType = val.getType();
  VectorType vecType = mlir::dyn_cast<VectorType>(valType);
  Type valElemType = elementTypeWithVector(valType);
  assert(mlir::isa<IntegerType>(valElemType) && "Expecting integer type");
  Type destType =
      getTypeWithVector(vecType, b().getIntegerType(width, false /*signed*/));
  return b()
      .create<UnrealizedConversionCastOp>(loc(), destType, val)
      .getResult(0);
}

// Methods inspired from MLIR TosaToLinalg CastOp.
Value MathBuilder::cast(Type destType, Value src) const {
  // Get element type and vector types (if any, i.e. possibly nullptr).
  Type srcType = src.getType();
  VectorType srcVecType = mlir::dyn_cast<VectorType>(srcType);
  VectorType destVecType = mlir::dyn_cast<VectorType>(destType);
  Type srcElemType = elementTypeWithVector(srcType);
  Type destElemType = elementTypeWithVector(destType);
  // Make sure we don't mix vector and scalars.
  assert(((srcVecType && destVecType) || (!srcVecType && !destVecType)) &&
         "expect both to be scalars or vectors");
  // Check if we even need a cast.
  if (srcType == destType)
    return src;

  // Process index types first.
  if (mlir::isa<IndexType>(srcElemType)) {
    // If the source is an index type, first convert it into a signless int of
    // size 64.
    srcElemType = b().getIntegerType(64);
    srcType = getTypeWithVector(srcVecType, srcElemType);
    src = b().create<arith::IndexCastOp>(loc(), srcType, src);
  }
  bool destIsIndex = false;
  Type savedDestType = destType; // Used when destIsIndex is true.
  if (mlir::isa<IndexType>(destElemType)) {
    // If the dest is an index type, pretend for now that we want it to be
    // converted to signless int of size 64.
    destElemType = b().getIntegerType(64);
    destType = getTypeWithVector(destVecType, destElemType);
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

  // Handle boolean first because they need special handling.
  // Boolean to int/float conversions. Boolean are unsigned.
  if (srcElemType.isInteger(1)) {
    if (mlir::isa<FloatType>(destElemType)) {
      return b().create<arith::UIToFPOp>(loc(), destType, src);
    } else {
      Value dest = b().create<arith::ExtUIOp>(loc(), destType, src);
      if (destIsIndex)
        dest = b().create<arith::IndexCastOp>(loc(), savedDestType, dest);
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
          srcVecType, IntegerType::get(srcElemType.getContext(), srcElemWidth));
      src = castToSignless(src, srcElemWidth);
    }
    Value zero = constant(constantType, 0);
    return neq(src, zero);
  }

  // Float to float conversions.
  if (mlir::isa<FloatType>(srcElemType) && mlir::isa<FloatType>(destElemType)) {
    assert((bitExtend || bitTrunc) && "expected extend or trunc");
    if (bitExtend)
      return b().create<arith::ExtFOp>(loc(), destType, src);
    else
      return b().create<arith::TruncFOp>(loc(), destType, src);
  }

  // Float to int conversions.
  if (mlir::isa<FloatType>(srcElemType) &&
      mlir::isa<IntegerType>(destElemType)) {
    // TosaToLinalg in MLIR uses a fancier algorithm that clamps values to
    // min/max signed/unsigned integer values.
    if (destType.isUnsignedInteger()) {
      Type castType = b().getIntegerType(destElemWidth);
      Value cast = b().create<arith::FPToUIOp>(loc(), castType, src);
      return castToUnsigned(cast, destElemWidth);
    } else {
      // Handle signed int.
      Value dest = b().create<arith::FPToSIOp>(loc(), destType, src);
      if (destIsIndex)
        dest = b().create<arith::IndexCastOp>(loc(), savedDestType, dest);
      return dest;
    }
  }

  // Int to float conversion.
  if (mlir::isa<IntegerType>(srcElemType) &&
      mlir::isa<FloatType>(destElemType)) {
    if (srcElemType.isUnsignedInteger()) {
      Value cast = castToSignless(src, srcElemWidth);
      return b().create<arith::UIToFPOp>(loc(), destType, cast);
    } else {
      // Handle signed int.
      return b().create<arith::SIToFPOp>(loc(), destType, src);
    }
  }

  // Int to int conversion.
  if (mlir::isa<IntegerType>(srcType) && mlir::isa<IntegerType>(destType)) {
    if (srcType.isUnsignedInteger()) {
      // Unsigned to unsigned/signed conversion.
      // Same bit width for unsigned to signed conversion.
      if ((srcElemWidth == destElemWidth) && destType.isSignlessInteger())
        return castToSignless(src, srcElemWidth);
      // Different bit width.
      assert((bitExtend || bitTrunc) && "expected extend or trunc");
      // Has to convert to signless first, and reconvert output to unsigned.
      Value cast = castToSignless(src, srcElemWidth);
      Type castType = b().getIntegerType(destElemWidth);
      if (bitExtend) {
        cast = b().create<arith::ExtUIOp>(loc(), castType, cast);
      } else {
        // TosaToLinalg use a clipping algo, not sure if needed.
        cast = b().create<arith::TruncIOp>(loc(), castType, cast);
      }
      if (destType.isUnsignedInteger()) {
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
      if ((srcElemWidth == destElemWidth) && destType.isUnsignedInteger())
        return castToUnsigned(src, srcElemWidth);
      // Different bit width.
      Value dest = src;
      if (bitExtend)
        dest = b().create<arith::ExtSIOp>(loc(), destType, src);
      if (bitTrunc)
        // TosaToLinalg use a clipping algo
        dest = b().create<arith::TruncIOp>(loc(), destType, src);
      if (destIsIndex)
        return b().create<arith::IndexCastOp>(loc(), b().getIndexType(), dest);
      if (destType.isUnsignedInteger()) {
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
void MathBuilder::addOffsetToLeastSignificant(mlir::ValueRange indices,
    mlir::ValueRange offsets,
    llvm::SmallVectorImpl<mlir::Value> &computedIndices) const {
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

void MathBuilder::addOffsetToLeastSignificant(mlir::ArrayRef<IndexExpr> indices,
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
  return b().create<shape::ShapeOfOp>(loc(), val);
}

Value ShapeBuilder::fromExtents(ValueRange extents) const {
  return b().create<shape::FromExtentsOp>(loc(), extents);
}

Value ShapeBuilder::toExtentTensor(Type type, Value shape) const {
  return b().create<shape::ToExtentTensorOp>(loc(), type, shape);
}

Value ShapeBuilder::getExtent(Value val, int64_t index) const {
  return b().create<shape::GetExtentOp>(loc(), val, index);
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
void MemRefBuilder::computeDynSymbols(MemRefType type,
    llvm::SmallVectorImpl<IndexExpr> &dims,
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
// Alloc functions without alignment.

memref::AllocOp MemRefBuilder::alloc(MemRefType type) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  return alloc(type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(
    MemRefType type, ValueRange dynSymbols) const {
  // Constant, ignore the dynamic symbols.
  if (dynSymbols.size() == 0)
    return b().create<memref::AllocOp>(loc(), type);
  return b().create<memref::AllocOp>(loc(), type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(
    Value operandOfSameType, MemRefType type) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(operandOfSameType, type, dynSymbols);
  return alloc(type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(
    MemRefType type, llvm::SmallVectorImpl<IndexExpr> &dims) const {
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
    return b().create<memref::AllocOp>(loc(), type, alignmentAttr);
  return b().create<memref::AllocOp>(loc(), type, dynSymbols, alignmentAttr);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    Value operandOfSameType, MemRefType type, int64_t alignment) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(operandOfSameType, type, dynSymbols);
  return alignedAlloc(type, dynSymbols, alignment);
}

memref::AllocOp MemRefBuilder::alignedAlloc(MemRefType type,
    llvm::SmallVectorImpl<IndexExpr> &dims, int64_t alignment) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return alignedAlloc(type, dynSymbols, alignment);
}

//===----------------------------------------------------------------------===//
// Info about memory size.

// Compute static and dynamic size of memref in elements. Return true if has
// static size.
bool MemRefBuilder::getStaticAndDynamicMemSize(MemRefType type,
    ValueRange dynSymbols, int64_t &staticSize, IndexExpr &dynSize,
    int64_t range) const {
  Type elementType = type.getElementType();
  assert(!(mlir::isa<VectorType>(elementType)) && "unsupported vector type");
  ArrayRef<int64_t> shape = type.getShape();
  staticSize = 1;                // Multiplication of static sizes.
  dynSize = LiteralIndexExpr(1); // Multiplication of dyn sizes.
  bool staticShape = true;       // Static until proven otherwise.
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
        dynSize = dynSize * SymbolIndexExpr(dynSymbols[iDim]);
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
    llvm::SmallVectorImpl<IndexExpr> &dims, int64_t &staticSize,
    IndexExpr &dynSize, int64_t range) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return getStaticAndDynamicMemSize(
      type, dynSymbols, staticSize, dynSize, range);
}

//===----------------------------------------------------------------------===//
// Alloc functions with alignment and padding for SIMD

Value MemRefBuilder::alignedAllocWithSimdPadding(
    mlir::MemRefType type, int64_t simdUnroll, int64_t alignment) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  return alignedAllocWithSimdPadding(type, dynSymbols, simdUnroll, alignment);
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
    IndexExpr totByteSize = LiteralIndexExpr(staticSize * byteWidth) * dynSize;
    totPaddedByteSize = totByteSize + LiteralIndexExpr(paddingSize * byteWidth);
  } else {
    // We have sub-byte element sizes. Need to do precise computations. Namely
    // first compute tot total number of bits (including static/dynamic
    // and padding bit sizes), and then doing a ceil division by
    // 8 (number of bits in a byte).
    IndexExpr totBitSize = LiteralIndexExpr(staticSize * bitWidth) * dynSize;
    IndexExpr totPaddedBitSize =
        totBitSize + LiteralIndexExpr(paddingSize * bitWidth);
    totPaddedByteSize = totPaddedBitSize.ceilDiv(LiteralIndexExpr(8));
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

Value MemRefBuilder::alignedAllocWithSimdPadding(MemRefType type,
    llvm::SmallVectorImpl<IndexExpr> &dims, int64_t VL,
    int64_t alignment) const {
  llvm::SmallVector<Value, 4> dynSymbols;
  computeDynSymbols(type, dims, dynSymbols);
  return alignedAllocWithSimdPadding(type, dynSymbols, VL, alignment);
}

//===----------------------------------------------------------------------===//
// Alloca

memref::AllocaOp MemRefBuilder::alloca(MemRefType type) const {
  return b().create<memref::AllocaOp>(loc(), type);
}

memref::AllocaOp MemRefBuilder::alignedAlloca(
    MemRefType type, int64_t alignment) const {
  // Drop align for scalars.
  if (type.getShape().size() == 0)
    return b().create<memref::AllocaOp>(loc(), type);
  // Has array, use alignment.
  IntegerAttr alignmentAttr = computeAlignment(alignment);
  return b().create<memref::AllocaOp>(loc(), type, alignmentAttr);
}

//===----------------------------------------------------------------------===//
// Dealloc.

memref::DeallocOp MemRefBuilder::dealloc(Value val) const {
  return b().create<memref::DeallocOp>(loc(), val);
}

//===----------------------------------------------------------------------===//
// Reshape.

memref::ReshapeOp MemRefBuilder::reshape(MemRefType destType,
    Value valToReshape, Value outputShapeStoredInMem) const {
  return b().create<memref::ReshapeOp>(
      loc(), destType, valToReshape, outputShapeStoredInMem);
}

memref::ReshapeOp MemRefBuilder::reshape(
    llvm::SmallVectorImpl<IndexExpr> &destDims, Value valToReshape) const {
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
    llvm::SmallVectorImpl<IndexExpr> &dims,
    llvm::SmallVectorImpl<IndexExpr> &flattenedDims,
    int64_t dimsToFlatten) const {
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
    flattenedDims = dims;
    return valToReshape;
  }
  // Compute the dimensions of the flattened array.
  int64_t axis = inputRank - dimsToFlatten;
  flattenedDims.clear();
  // Up to axis, flatten dims == input dims.
  for (int64_t d = 0; d < axis; ++d)
    flattenedDims.emplace_back(dims[d]);
  // Last flatten dim is the product of remaining input dims.
  IndexExpr numOfFlattenedElements = LiteralIndexExpr(1);
  for (int64_t d = axis; d < inputRank; ++d)
    numOfFlattenedElements = numOfFlattenedElements * dims[d];
  flattenedDims.emplace_back(numOfFlattenedElements);
  // Reshape.
  return reshape(flattenedDims, valToReshape);
}

Value MemRefBuilder::reshapeToFlat2D(Value valToReshape,
    llvm::SmallVectorImpl<IndexExpr> &dims,
    llvm::SmallVectorImpl<IndexExpr> &flattenedDims, int64_t axis) const {
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
    flattenedDims = dims;
    return valToReshape;
  }
  // Compute the dimensions of the flattened array.
  flattenedDims.clear();
  // First output dim: product of input dims until axis (exclusively).
  IndexExpr numElement1stDim = LiteralIndexExpr(1);
  for (int64_t d = 0; d < axis; ++d)
    numElement1stDim = numElement1stDim * dims[d];
  flattenedDims.emplace_back(numElement1stDim);
  // Second output dim: product of input dims after axis (inclusively).
  IndexExpr numElement2ndDim = LiteralIndexExpr(1);
  for (int64_t d = axis; d < inputRank; ++d)
    numElement2ndDim = numElement2ndDim * dims[d];
  flattenedDims.emplace_back(numElement2ndDim);
  // Reshape.
  return reshape(flattenedDims, valToReshape);
}

memref::ReshapeOp MemRefBuilder::reshapeFromFlat(Value valToReshape,
    llvm::SmallVectorImpl<IndexExpr> &outputDims, MemRefType outputType) const {
  assert(!hasNonIdentityLayout(outputType) && "MemRef is not normalized");
  return reshape(outputDims, valToReshape);
}

//===----------------------------------------------------------------------===//
// Casts and views.

memref::CastOp MemRefBuilder::cast(Value input, MemRefType outputType) const {
  return b().create<memref::CastOp>(loc(), outputType, input);
}

Value MemRefBuilder::reinterpretCast(
    Value input, SmallVectorImpl<IndexExpr> &outputDims) const {
  // IndexExpr zero = LiteralIndexExpr(0);
  return reinterpretCast(input, nullptr, outputDims);
}

Value MemRefBuilder::reinterpretCast(
    Value input, Value offset, SmallVectorImpl<IndexExpr> &outputDims) const {
  // Compute new sizes and strides.
  int64_t rank = outputDims.size();
  SmallVector<IndexExpr, 4> sizesIE, stridesIE;
  sizesIE.resize(rank);
  stridesIE.resize(rank);
  IndexExpr strideIE = LiteralIndexExpr(1);
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
    return b().create<memref::ReinterpretCastOp>(
        loc(), outputMemRefType, input, offset, sizes, strides);
  // Null offset: use zero attribute (remain compatible with old lit tests).
  return b().create<memref::ReinterpretCastOp>(loc(), outputMemRefType, input,
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
  return b().create<memref::CollapseShapeOp>(
      loc(), outputType, input, reassociation);
}

memref::ViewOp MemRefBuilder::view(Value input, int64_t byteOffset,
    MemRefType outputType, ValueRange outputDynSymbols) const {
  MultiDialectBuilder<MathBuilder> create(*this);
  Value offset = create.math.constantIndex(byteOffset);
  // auto offset = b().createOrFold<arith::ConstantIndexOp>(byteOffset);
  return b().create<memref::ViewOp>(
      loc(), outputType, input, offset, outputDynSymbols);
}

memref::SubViewOp MemRefBuilder::subView(Value val,
    llvm::SmallVectorImpl<int64_t> &offsets,
    llvm::SmallVectorImpl<int64_t> &sizes,
    llvm::SmallVectorImpl<int64_t> &strides) const {
  return b().create<memref::SubViewOp>(loc(), val, offsets, sizes, strides);
}

memref::SubViewOp MemRefBuilder::subView(MemRefType outputType, Value val,
    llvm::SmallVectorImpl<int64_t> &offsets,
    llvm::SmallVectorImpl<int64_t> &sizes,
    llvm::SmallVectorImpl<int64_t> &strides) const {
  return b().create<memref::SubViewOp>(
      loc(), outputType, val, offsets, sizes, strides);
}

memref::SubViewOp MemRefBuilder::subView(Value input,
    llvm::SmallVectorImpl<IndexExpr> &offsetsIE,
    llvm::SmallVectorImpl<IndexExpr> &sizesIE,
    llvm::SmallVectorImpl<IndexExpr> &stridesIE) const {
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
  return b().create<memref::SubViewOp>(
      loc(), outputType, input, offsets, sizes, strides);
}

//===----------------------------------------------------------------------===//
// Dims.

Value MemRefBuilder::dim(Value val, int64_t index) const {
  assert(index >= 0 && "Expecting a valid index");
  return dim(val, b().create<arith::ConstantIndexOp>(loc(), index));
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
  b().create<memref::PrefetchOp>(
      loc(), memref, indices, isWrite, locality, isData);
}

void MemRefBuilder::prefetchIE(Value memref,
    llvm::SmallVectorImpl<IndexExpr> &indices, bool isWrite, unsigned locality,
    bool isData) {
  SmallVector<Value, 4> indexVals;
  IndexExpr::getValues(indices, indexVals);
  prefetch(memref, indexVals, isWrite, locality, isData);
}

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF).
//===----------------------------------------------------------------------===//

void SCFBuilder::ifThenElse(Value cond,
    function_ref<void(SCFBuilder &createSCF)> thenFn,
    function_ref<void(SCFBuilder &createSCF)> elseFn) const {
  if (!elseFn) {
    b().create<scf::IfOp>(loc(), cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          thenFn(scfBuilder);
          yield();
        });
  } else {
    b().create<scf::IfOp>(
        loc(), cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          thenFn(scfBuilder);
          b().create<scf::YieldOp>(loc());
        },
        /*else*/
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          elseFn(scfBuilder);
          yield();
        });
  }
}

void SCFBuilder::forLoop(Value lowerBound, Value upperBound, int64_t step,
    function_ref<void(SCFBuilder &createSCF, Value)> bodyFn) const {
  MathBuilder createMath(*this);
  Value stepVal = createMath.constantIndex(step);
  b().create<scf::ForOp>(loc(), lowerBound, upperBound, stepVal, std::nullopt,
      [&](OpBuilder &childBuilder, Location childLoc, Value inductionVar,
          ValueRange args) {
        SCFBuilder builder(childBuilder, childLoc);
        bodyFn(builder, inductionVar);
        yield();
      });
}

void SCFBuilder::parallelLoop(ValueRange lowerBounds, ValueRange upperBounds,
    ValueRange steps,
    function_ref<void(SCFBuilder &createSCF, ValueRange)> bodyFn) const {
  // SmallVectorImpl<Value> ivStorage;
  b().create<scf::ParallelOp>(loc(), lowerBounds, upperBounds, steps,
      [&](OpBuilder &childBuilder, Location childLoc,
          ValueRange inductionVars) {
        SCFBuilder builder(childBuilder, childLoc);
        bodyFn(builder, inductionVars);
        yield();
      });
}

void SCFBuilder::yield() const { b().create<scf::YieldOp>(loc()); }

//===----------------------------------------------------------------------===//
// Vector Builder
//===----------------------------------------------------------------------===//

int64_t VectorBuilder::getMachineVectorLength(const Type &elementType) const {
  VectorMachineSupport *vms =
      VectorMachineSupport::getGlobalVectorMachineSupport();
  // Even if unsupported, we can always compute one result per vector.
  return std::max((int64_t)1, vms->getVectorLength(elementType));
}

int64_t VectorBuilder::getMachineVectorLength(const VectorType &vecType) const {
  return getMachineVectorLength(vecType.getElementType());
}

int64_t VectorBuilder::getMachineVectorLength(Value vecValue) const {
  VectorType vecType = mlir::dyn_cast_or_null<VectorType>(vecValue.getType());
  assert(vecType && "expected vector type");
  return getMachineVectorLength(vecType.getElementType());
}

Value VectorBuilder::load(
    VectorType vecType, Value memref, ValueRange indices) const {
  return b().create<vector::LoadOp>(loc(), vecType, memref, indices);
}
mlir::Value VectorBuilder::load(mlir::VectorType vecType, mlir::Value memref,
    mlir::ValueRange indices, mlir::ValueRange offsets) const {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(*this);
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  return load(vecType, memref, computedIndices);
}

mlir::Value VectorBuilder::loadIE(mlir::VectorType vecType, mlir::Value memref,
    llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) const {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(*this);
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  return load(vecType, memref, computedIndices);
}

void VectorBuilder::store(Value val, Value memref, ValueRange indices) const {
  b().create<vector::StoreOp>(loc(), val, memref, indices);
}

void VectorBuilder::store(mlir::Value val, mlir::Value memref,
    mlir::ValueRange indices, mlir::ValueRange offsets) const {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(*this);
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  store(val, memref, computedIndices);
}

void VectorBuilder::storeIE(mlir::Value val, mlir::Value memref,
    llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) const {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(*this);
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  store(val, memref, computedIndices);
}

Value VectorBuilder::fma(Value lhs, Value rhs, Value acc) const {
  return b().create<vector::FMAOp>(loc(), lhs, rhs, acc);
}

// Val is required to be a index/integer/float.
Value VectorBuilder::splat(VectorType vecType, Value val) const {
  return b().create<vector::SplatOp>(loc(), vecType, val);
}

Value VectorBuilder::broadcast(VectorType vecType, Value val) const {
  return b().create<vector::BroadcastOp>(loc(), vecType, val);
}

Value VectorBuilder::shuffle(
    Value lhs, Value rhs, SmallVectorImpl<int64_t> &mask) const {
  return b().create<vector::ShuffleOp>(loc(), lhs, rhs, mask);
}

Value VectorBuilder::typeCast(Type resTy, Value val) const {
  return b().create<vector::TypeCastOp>(loc(), resTy, val);
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
    return b().create<vector::ReductionOp>(
        loc(), vector::CombiningKind::ADD, value);
  }
  case CombiningKind::MUL: {
    return b().create<vector::ReductionOp>(
        loc(), vector::CombiningKind::MUL, value);
  }
  case CombiningKind::MAX: {
    if (MathBuilder::isUnsignedIntegerWithVector(type))
      return b().create<vector::ReductionOp>(
          loc(), vector::CombiningKind::MAXUI, value);
    if (MathBuilder::isIntegerWithVector(type))
      return b().create<vector::ReductionOp>(
          loc(), vector::CombiningKind::MAXSI, value);
    if (MathBuilder::isFloatWithVector(type))
      return b().create<vector::ReductionOp>(
          loc(), vector::CombiningKind::MAXNUMF, value);
    llvm_unreachable("unknown type in max");
  }
  case CombiningKind::MIN: {
    if (MathBuilder::isUnsignedIntegerWithVector(type))
      return b().create<vector::ReductionOp>(
          loc(), vector::CombiningKind::MINUI, value);
    if (MathBuilder::isIntegerWithVector(type))
      return b().create<vector::ReductionOp>(
          loc(), vector::CombiningKind::MINSI, value);
    if (MathBuilder::isFloatWithVector(type))
      return b().create<vector::ReductionOp>(
          loc(), vector::CombiningKind::MINNUMF, value);
    llvm_unreachable("unknown type in min");
  }
  case CombiningKind::AND: {
    if (MathBuilder::isIntegerWithVector(type))
      return b().create<vector::ReductionOp>(
          loc(), vector::CombiningKind::AND, value);
    llvm_unreachable("unknown type in and");
  }
  case CombiningKind::OR: {
    if (MathBuilder::isIntegerWithVector(type))
      return b().create<vector::ReductionOp>(
          loc(), vector::CombiningKind::OR, value);
    llvm_unreachable("unknown type in or");
  }
  case CombiningKind::XOR: {
    if (MathBuilder::isIntegerWithVector(type))
      return b().create<vector::ReductionOp>(
          loc(), vector::CombiningKind::XOR, value);
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

void VectorBuilder::multiReduction(SmallVectorImpl<Value> &inputVecArray,
    F2 reductionFct, SmallVectorImpl<Value> &outputVecArray) {
  uint64_t N = inputVecArray.size();
  assert(N > 0 && "expected at least one value to reduce");
  uint64_t VL = getLengthOf1DVector(inputVecArray[0]);
  uint64_t machineVL = getMachineVectorLength(inputVecArray[0]);
  // TODO alex, should relax this
  assert(VL == machineVL && "only natural sizes supported at this time");
  assert(N % machineVL == 0 &&
         "can only reduces multiple of VL vectors at this time");
  LLVM_DEBUG(llvm::dbgs() << "reduction with N " << N << ", VL " << VL
                          << ", mVL " << machineVL << "\n";);

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
  // Process each block of machineVL input vectors at a time.
  for (uint64_t r = 0; r < N; r += machineVL) {
    // Algorithm for the set of input arrays from tmp[r] to
    // tmp[r+machineVL-1].
    // With machineVL inputs, we have machineVL/2 initial pairs.
    uint64_t numPairs = machineVL / 2;
    // While we have pairs...
    for (uint64_t step = 1; step < machineVL; step = step * 2) {
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
    // Completed the machineVL x machineVL reduction, save it in the output.
    outputVecArray.emplace_back(tmpArray[r]);
  }
}

int64_t VectorBuilder::computeSuitableUnrollFactor(VectorMachineSupport *vms,
    MemRefType memRefType, llvm::SmallVectorImpl<IndexExpr> &memRefDims,
    int64_t collapsedInnermostLoops, int64_t maxSimdUnroll, bool canPad,
    int64_t &simdLoopStaticTripCount) const {
  assert(collapsedInnermostLoops > 0 && "expected at least one collapsed loop");
  assert(maxSimdUnroll > 0 && "expected positive max simd unroll");
  simdLoopStaticTripCount = 0; // Initially assume no SIMD.
  Type elementType = memRefType.getElementType();
  int64_t VL = vms->getVectorLength(elementType);
  LLVM_DEBUG(llvm::dbgs() << "  simd hw VL is " << VL << "\n");
  if (VL == 0) {
    LLVM_DEBUG(llvm::dbgs() << "  simd disabled: no simd\n");
    return 0;
  }
  MemRefBuilder createMem(*this);
  int64_t staticSize;
  IndexExpr dynSize;
  bool isStaticSize = createMem.getStaticAndDynamicMemSize(
      memRefType, memRefDims, staticSize, dynSize, -collapsedInnermostLoops);
  if (isStaticSize && staticSize < VL) {
    LLVM_DEBUG(llvm::dbgs() << "  simd disabled: trip count " << staticSize
                            << " too short for a VL of " << VL << "\n");
    return 0;
  }
  // Unless otherwise disabled, here is the estimated trip count.
  simdLoopStaticTripCount = staticSize > 1 ? staticSize : -1;
  if (canPad && collapsedInnermostLoops == (int64_t)memRefType.getRank()) {
    // Fully collapsed and can add padding to be fine
    return maxSimdUnroll * VL;
  }
  // We have a partially flattened operator. Since we do only simdize entire
  // loops (i.e. we don't support scalar epilogues at this time), make sure
  // the static size is a multiple of the VL. Get the VL of the store
  // (output's element type).
  if (staticSize % VL != 0) {
    LLVM_DEBUG(llvm::dbgs() << "  simd disabled: partial flattened dims "
                            << collapsedInnermostLoops << " with size "
                            << staticSize << " is not 0 mod VL " << VL << "\n");
    return 0;
  }
  // See if we can get a unroll factor.
  for (int64_t u = maxSimdUnroll; u > 0; --u) {
    if (staticSize % (u * VL) == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  partial flattened dims " << collapsedInnermostLoops
                 << " with size " << staticSize << " works with VL " << VL
                 << " and unroll " << u << "\n");
      return u * VL;
    }
  }
  llvm_unreachable("should always find u==1 feasible");
}

//===----------------------------------------------------------------------===//
// LLVM Builder
//===----------------------------------------------------------------------===//

Value LLVMBuilder::add(Value lhs, Value rhs) const {
  return b().create<LLVM::AddOp>(loc(), lhs, rhs);
}

Value LLVMBuilder::addressOf(LLVM::GlobalOp op) const {
  return b().create<LLVM::AddressOfOp>(loc(), op);
}

Value LLVMBuilder::_alloca(
    Type resultType, Type elementType, Value size, int64_t alignment) const {
  return b().create<LLVM::AllocaOp>(
      loc(), resultType, elementType, size, alignment);
}

Value LLVMBuilder::andi(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return b().create<LLVM::AndOp>(loc(), lhs, rhs);
}

Value LLVMBuilder::bitcast(Type type, Value val) const {
  return b().create<LLVM::BitcastOp>(loc(), type, val);
}

void LLVMBuilder::br(ArrayRef<Value> destOperands, Block *destBlock) const {
  b().create<LLVM::BrOp>(loc(), destOperands, destBlock);
}

Value LLVMBuilder::call(ArrayRef<Type> resultTypes, StringRef funcName,
    ArrayRef<Value> inputs) const {
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  LLVM::CallOp callOp =
      b().create<LLVM::CallOp>(loc(), resultTypes, funcName, inputs);
  // CallOp may return either 0 or 1 value.
  if (resultTypes.empty())
    return nullptr;
  return callOp.getResult();
}

Value LLVMBuilder::call(ArrayRef<Type> resultTypes,
    FlatSymbolRefAttr funcSymbol, ArrayRef<Value> inputs) const {
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  LLVM::CallOp callOp =
      b().create<LLVM::CallOp>(loc(), resultTypes, funcSymbol, inputs);
  // CallOp may return either 0 or 1 value.
  if (resultTypes.empty())
    return nullptr;
  return callOp.getResult();
}

void LLVMBuilder::condBr(Value cond, Block *trueBlock,
    llvm::ArrayRef<Value> trueOperands, Block *falseBlock,
    llvm::ArrayRef<Value> falseOperands) const {
  b().create<LLVM::CondBrOp>(
      loc(), cond, trueBlock, trueOperands, falseBlock, falseOperands);
}

Value LLVMBuilder::constant(Type type, int64_t val) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<IntegerType>([&](IntegerType type) {
        unsigned width = type.getWidth();
        if (width == 1)
          constant = b().create<LLVM::ConstantOp>(
              loc(), type, b().getBoolAttr(val != 0));
        else {
          assert(type.isSignless() &&
                 "LLVM::ConstantOp requires a signless type.");
          constant = b().create<LLVM::ConstantOp>(loc(), type,
              b().getIntegerAttr(type, APInt(width, (int64_t)val)));
        }
      })
      .Case<IndexType>([&](Type) {
        constant = b().create<LLVM::ConstantOp>(
            loc(), type, b().getIntegerAttr(type, val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value LLVMBuilder::constant(Type type, double val) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<Float16Type>([&](Type) {
        constant =
            b().create<LLVM::ConstantOp>(loc(), type, b().getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant =
            b().create<LLVM::ConstantOp>(loc(), type, b().getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant =
            b().create<LLVM::ConstantOp>(loc(), type, b().getF64FloatAttr(val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value LLVMBuilder::extractElement(
    Type resultType, Value container, int64_t position) const {
  Value posVal = constant(b().getI64Type(), position);
  return b().create<LLVM::ExtractElementOp>(
      loc(), resultType, container, posVal);
}

Value LLVMBuilder::extractValue(
    Type resultType, Value container, ArrayRef<int64_t> position) const {
  return b().create<LLVM::ExtractValueOp>(
      loc(), resultType, container, position);
}

LLVM::LLVMFuncOp LLVMBuilder::func(
    StringRef funcName, Type funcType, bool createUniqueFunc) const {
  // If createUniqueFunc, we create two functions: name and name_postfix.
  // They have the same signatures and `name` will call `name_postfix`.
  // `name_postfix` funtion is expected to be unique across all generated
  // modules, allowing to run multiple models at the same time.
  LLVM::LLVMFuncOp funcOp =
      b().create<LLVM::LLVMFuncOp>(loc(), funcName, funcType);
  if (!createUniqueFunc)
    return funcOp;

  // Create uniqueFuncOp if there exists a postfix.
  // Since `funcOp` calls `uniqueFuncOp`, put `uniqueFuncOp`'s definition before
  // `funcOp`.
  b().setInsertionPoint(funcOp);
  ModuleOp module = funcOp.getOperation()->getParentOfType<ModuleOp>();
  std::string uniqueFuncName =
      LLVMBuilder::SymbolPostfix(module, funcName.str());
  if (uniqueFuncName == funcName.str())
    return funcOp;

  auto uniqueFuncType = cast<LLVM::LLVMFunctionType>(funcType);
  LLVM::LLVMFuncOp uniqueFuncOp =
      b().create<LLVM::LLVMFuncOp>(loc(), uniqueFuncName, uniqueFuncType);

  // Call uniqueFuncOp inside funcOp.
  Block *entryBlock = funcOp.addEntryBlock(b());
  OpBuilder::InsertionGuard bodyGuard(b());
  b().setInsertionPointToStart(entryBlock);
  ValueRange args = entryBlock->getArguments();
  TypeRange resultTypes = uniqueFuncType.getReturnTypes();
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  if (resultTypes.size() == 0 || isa<LLVM::LLVMVoidType>(resultTypes[0])) {
    b().create<LLVM::CallOp>(loc(), ArrayRef<Type>({}), uniqueFuncName, args);
    b().create<LLVM::ReturnOp>(loc(), ArrayRef<Value>({}));
  } else {
    LLVM::CallOp callOp =
        b().create<LLVM::CallOp>(loc(), resultTypes, uniqueFuncName, args);
    b().create<LLVM::ReturnOp>(loc(), ArrayRef<Value>({callOp.getResult()}));
  }

  return uniqueFuncOp;
}

Value LLVMBuilder::getElemPtr(Type resultType, Type elemType, Value base,
    ArrayRef<LLVM::GEPArg> indices) const {
  return b().create<LLVM::GEPOp>(loc(), resultType, elemType, base, indices);
}

LLVM::GlobalOp LLVMBuilder::globalOp(Type resultType, bool isConstant,
    LLVM::Linkage linkage, StringRef name, Attribute valueAttr,
    uint64_t alignment, bool uniqueName) const {
  LLVM::GlobalOp gop = b().create<LLVM::GlobalOp>(loc(), resultType,
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
  return b().create<LLVM::ICmpOp>(loc(), cond, lhs, rhs);
}

Value LLVMBuilder::insertElement(Value vec, Value val, int64_t position) const {
  Value posVal = constant(b().getI64Type(), position);
  return b().create<LLVM::InsertElementOp>(loc(), vec, val, posVal);
}

Value LLVMBuilder::insertValue(Type resultType, Value container, Value val,
    llvm::ArrayRef<int64_t> position) const {
  return b().create<LLVM::InsertValueOp>(
      loc(), resultType, container, val, position);
}

Value LLVMBuilder::inttoptr(Type type, Value val) const {
  return b().create<LLVM::IntToPtrOp>(loc(), type, val);
}

Value LLVMBuilder::lshr(Value lhs, Value rhs) const {
  return b().create<LLVM::LShrOp>(loc(), lhs, rhs);
}

Value LLVMBuilder::load(Type elementType, Value addr) const {
  return b().create<LLVM::LoadOp>(loc(), elementType, addr);
}

Value LLVMBuilder::mul(Value lhs, Value rhs) const {
  return b().create<LLVM::MulOp>(loc(), lhs, rhs);
}

Value LLVMBuilder::null(Type type) const {
  return b().create<LLVM::ZeroOp>(loc(), type);
}

Value LLVMBuilder::ori(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return b().create<LLVM::OrOp>(loc(), lhs, rhs);
}

Value LLVMBuilder::ptrtoint(Type type, Value val) const {
  return b().create<LLVM::PtrToIntOp>(loc(), type, val);
}

void LLVMBuilder::_return() const {
  b().create<LLVM::ReturnOp>(loc(), ArrayRef<Value>{});
}

void LLVMBuilder::_return(Value val) const {
  b().create<LLVM::ReturnOp>(loc(), ArrayRef<Value>({val}));
}

Value LLVMBuilder::select(Value cmp, Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return b().create<LLVM::SelectOp>(loc(), cmp, lhs, rhs);
}

Value LLVMBuilder::sext(Type type, Value val) const {
  return b().create<LLVM::SExtOp>(loc(), type, val);
}

Value LLVMBuilder::shl(Value lhs, Value rhs) const {
  return b().create<LLVM::ShlOp>(loc(), lhs, rhs);
}

void LLVMBuilder::store(Value val, Value addr) const {
  b().create<LLVM::StoreOp>(loc(), val, addr);
}

Value LLVMBuilder::trunc(Type type, Value val) const {
  return b().create<LLVM::TruncOp>(loc(), type, val);
}

Value LLVMBuilder::zext(Type type, Value val) const {
  return b().create<LLVM::ZExtOp>(loc(), type, val);
}

FlatSymbolRefAttr LLVMBuilder::getOrInsertSymbolRef(ModuleOp module,
    StringRef funcName, Type resultType, ArrayRef<Type> operandTypes,
    bool isVarArg) const {
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    OpBuilder::InsertionGuard guard(b());
    b().setInsertionPointToStart(module.getBody());
    LLVM::LLVMFunctionType funcType =
        LLVM::LLVMFunctionType::get(resultType, operandTypes, isVarArg);
    b().create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);
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
