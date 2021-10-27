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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
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

Value MathBuilder::_and(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return b.create<arith::AndIOp>(loc, lhs, rhs);
}

Value MathBuilder::_or(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return b.create<arith::OrIOp>(loc, lhs, rhs);
}

Value MathBuilder::add(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<arith::AddIOp>(loc, lhs, rhs);
  return b.create<arith::AddFOp>(loc, lhs, rhs);
}
Value MathBuilder::sub(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<arith::SubIOp>(loc, lhs, rhs);
  return b.create<arith::SubFOp>(loc, lhs, rhs);
}
Value MathBuilder::mul(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<arith::MulIOp>(loc, lhs, rhs);
  return b.create<arith::MulFOp>(loc, lhs, rhs);
}

Value MathBuilder::div(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<FloatType>())
    return b.create<arith::DivFOp>(loc, lhs, rhs);
  else
    llvm_unreachable("Only support float type at this moment.");
}

Value MathBuilder::exp(Value val) const {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return b.create<math::ExpOp>(loc, val);
}

Value MathBuilder::exp2(Value val) const {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return b.create<math::Exp2Op>(loc, val);
}

Value MathBuilder::log2(Value val) const {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return b.create<math::Log2Op>(loc, val);
}

Value MathBuilder::min(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    if (lhs.getType().isSignedInteger())
      return b.create<MinSIOp>(loc, lhs, rhs);
    else
      return b.create<MinUIOp>(loc, lhs, rhs);
  else
    return b.create<MinFOp>(loc, lhs, rhs);
}

Value MathBuilder::max(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    if (lhs.getType().isSignedInteger())
      return b.create<MaxSIOp>(loc, lhs, rhs);
    else
      return b.create<MaxUIOp>(loc, lhs, rhs);
  else
    return b.create<MaxFOp>(loc, lhs, rhs);
}

Value MathBuilder::sgt(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs);
  return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, lhs, rhs);
}

Value MathBuilder::sge(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lhs, rhs);
  return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, lhs, rhs);
}

Value MathBuilder::slt(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs);
  return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, lhs, rhs);
}

Value MathBuilder::sle(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, lhs, rhs);
  return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE, lhs, rhs);
}

Value MathBuilder::eq(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, rhs);
  return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, lhs, rhs);
}

Value MathBuilder::neq(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, rhs);
  return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, lhs, rhs);
}

Value MathBuilder::select(Value cmp, Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return b.create<SelectOp>(loc, cmp, lhs, rhs);
}

Value MathBuilder::constant(Type type, double val) const {
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
  return b.create<arith::ConstantOp>(loc, constantAttr);
}

Value MathBuilder::constantIndex(int64_t val) const {
  Attribute constantAttr = b.getIntegerAttr(b.getIndexType(), val);
  return b.create<arith::ConstantOp>(loc, constantAttr);
}

// For some reason, operations on unsigned int are often unhappy because
// operations are mainly used on signless integers. So this cast remove the sign
// of unsigned int for successful processing, to the best of my understanding.
Value MathBuilder::castToSignless(Value val, int64_t width) const {
  printf("hi alex: creating unrealized cast\n");
  Value res =
      b.create<UnrealizedConversionCastOp>(loc, b.getIntegerType(width), val)
          .getResult(0);
  res.dump();
  return res;
}

Value MathBuilder::castToUnsigned(Value val, int64_t width) const {
  printf("hi alex: creating unrealized cast\n");
  Value res = b.create<UnrealizedConversionCastOp>(
                   loc, b.getIntegerType(width, /*is signed*/ false), val)
                  .getResult(0);
  res.dump();
  return res;
}

// Methods inspired from MLIR TosaToLinalg CastOp.
// Handle here either elementary types (such as int/float) or shaped type of
// elementary types. Does not handle Index Type here.
Value MathBuilder::cast(Value src, Type destType) const {
  // Get elementary types.
  Type srcType = src.getType();
  ShapedType srcShapedType = srcType.dyn_cast_or_null<ShapedType>();
  Type srcElementType =
      srcShapedType ? srcShapedType.getElementType() : srcType;
  ShapedType destShapedType = destType.dyn_cast_or_null<ShapedType>();
  Type destElementType =
      destShapedType ? destShapedType.getElementType() : destType;
  // Do we need a conversion? If not, we are done.
  if (srcElementType == destElementType)
    return src;
  // No index expressions allowed.
  assert(!srcElementType.isIndex() && !destElementType.isIndex() &&
         "does not handle IndexType");
  // Get sizes.
  int64_t srcWidth = srcElementType.getIntOrFloatBitWidth();
  int64_t destWidth = destElementType.getIntOrFloatBitWidth();
  bool bitExtend = srcWidth < destWidth;

  // Handle boolean first because they need special handling.
  // Boolean to float/int conversions. Boolean are unsigned.
  if (srcElementType.isInteger(1)) {
    if (arith::UIToFPOp::areCastCompatible(srcType, destType)) {
      // To float.
      return b.create<arith::UIToFPOp>(loc, destType, src, mlir::None);
    }
    // To larger int.
    assert(destElementType.isa<IntegerType>() && bitExtend &&
           "unknown cast from bit");
    assert(arith::ExtUIOp::areCastCompatible(srcType, destType) &&
           "expected compatible");
    return b.create<arith::ExtUIOp>(loc, destType, src, mlir::None);
  }
  // Int/Float to booleans, just compare value to be unequal zero.
  if (destElementType.isInteger(1)) {
    Value zero = constant(srcElementType, 0);
    return neq(src, zero);
  }

  // Float to float conversions.
  if (srcElementType.isa<FloatType>() && destElementType.isa<FloatType>()) {
    if (bitExtend) {
      // Extend.
      assert(arith::ExtFOp::areCastCompatible(srcType, destType) &&
             "expected compatible");
      return b.create<arith::ExtFOp>(loc, destType, src, mlir::None);
    }
    // Truncate.
    assert(arith::TruncFOp::areCastCompatible(srcType, destType) &&
           "expected compatible");
    return b.create<arith::TruncFOp>(loc, destType, src, mlir::None);
  }

  // Float to int conversions.
  if (srcElementType.isa<FloatType>() && destElementType.isa<IntegerType>()) {
    // TosaToLinalg in MLIR uses a fancier algorithm that clamps values to
    // min/max signed/unsigned integer values.
    if (destElementType.isUnsignedInteger()) {
      Value cast = castToSignless(src, srcWidth);
      return b.create<arith::FPToUIOp>(loc, destType, cast);
    }
    // Handle signed int.
    assert(arith::FPToSIOp::areCastCompatible(srcType, destType) &&
           "expected compatible");
    return b.create<arith::FPToSIOp>(loc, destType, src, mlir::None);
  }

  // Int to float conversion.
  if (srcElementType.isa<IntegerType>() && destElementType.isa<FloatType>()) {
    if (srcElementType.isUnsignedInteger()) {
      Value cast = castToSignless(src, srcWidth);
      return b.create<arith::UIToFPOp>(loc, destType, cast);
    }
    // Handle signed int.
    assert(arith::SIToFPOp::areCastCompatible(srcType, destType) &&
           "expected compatible");
    return b.create<arith::SIToFPOp>(loc, destType, src, mlir::None);
  }

  // Int to int conversion.
  if (srcElementType.isa<IntegerType>() && destElementType.isa<IntegerType>()) {
    if (srcElementType.isUnsignedInteger()) {
      assert(destElementType.isUnsignedInteger() &&
             "no unsigned to signed conversion");
      Value cast = castToSignless(src, srcWidth);
      if (bitExtend) {
        printf("hi alex, extend\n");
        Type castType = b.getIntegerType(destWidth);
        cast = b.create<arith::ExtUIOp>(loc, castType, cast, mlir::None);
        return castToUnsigned(cast, destWidth);
      }
      // TosaToLinalg use a cliping algo
      printf("hi alex, trunc\n");
      return b.create<arith::TruncIOp>(loc, destType, cast, mlir::None);
    }
    // handle signed ingeger
    assert(!srcElementType.isUnsignedInteger() &&
           !destElementType.isUnsignedInteger() &&
           "no signed to unsigned conversion");
    if (bitExtend) {
      return b.create<arith::ExtSIOp>(loc, destType, src, mlir::None);
    }
    // TosaToLinalg use a cliping algo
    return b.create<arith::TruncIOp>(loc, destType, src, mlir::None);
  }

  // Handled all the cases supported so far.
  llvm_unreachable("unsupported element type");
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Memref support, including inserting default alignment.
//===----------------------------------------------------------------------===//

memref::AllocOp MemRefBuilder::alloc(
    MemRefType type, ValueRange dynSymbols) const {
  return b.create<memref::AllocOp>(loc, type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(MemRefType type) const {
  return b.create<memref::AllocOp>(loc, type);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, int64_t alignment) const {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  IntegerAttr alignmentAttr = b.getI64IntegerAttr(alignment);
  if (type.getShape().size() == 0) // Drop align for scalars.
    return b.create<memref::AllocOp>(loc, type);
  return b.create<memref::AllocOp>(loc, type, alignmentAttr);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, ValueRange dynSymbols, int64_t alignment) const {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  IntegerAttr alignmentAttr = b.getI64IntegerAttr(alignment);
  if (type.getShape().size() == 0) // Drop align for scalars.
    return b.create<memref::AllocOp>(loc, type, dynSymbols);
  return b.create<memref::AllocOp>(loc, type, dynSymbols, alignmentAttr);
}

memref::AllocaOp MemRefBuilder::alloca(MemRefType type) const {
  return b.create<memref::AllocaOp>(loc, type);
}

memref::AllocaOp MemRefBuilder::alignedAlloca(
    MemRefType type, int64_t alignment) const {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  IntegerAttr alignmentAttr = b.getI64IntegerAttr(alignment);
  if (type.getShape().size() == 0) // Drop align for scalars.
    return b.create<memref::AllocaOp>(loc, type);
  return b.create<memref::AllocaOp>(loc, type, alignmentAttr);
}

memref::DeallocOp MemRefBuilder::dealloc(Value val) const {
  return b.create<memref::DeallocOp>(loc, val);
}

memref::CastOp MemRefBuilder::cast(Value input, MemRefType outputType) const {
  return b.create<memref::CastOp>(loc, input, outputType);
}

Value MemRefBuilder::dim(Value val, int64_t index) const {
  Value i = b.create<arith::ConstantIndexOp>(loc, index);
  return b.createOrFold<memref::DimOp>(loc, val, i);
}

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF).
//===----------------------------------------------------------------------===//

void SCFBuilder::ifThenElse(Value cond,
    function_ref<void(SCFBuilder &createSCF)> thenFn,
    function_ref<void(SCFBuilder &createSCF)> elseFn) const {
  if (!elseFn) {
    b.create<scf::IfOp>(loc, cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          thenFn(scfBuilder);
          yield();
        });
  } else {
    b.create<scf::IfOp>(
        loc, cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          thenFn(scfBuilder);
          b.create<scf::YieldOp>(loc);
        },
        /*else*/
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          elseFn(scfBuilder);
          yield();
        });
  }
}

void SCFBuilder::yield() const { b.create<scf::YieldOp>(loc); }
