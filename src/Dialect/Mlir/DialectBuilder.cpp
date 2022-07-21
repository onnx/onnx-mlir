/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ DialectBuilder.cpp - Helper functions for MLIR dialects -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"

#define DEBUG_TYPE "dialect_builder"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Original code for MathBuilder is copied from LLVM MLIR Utils.cpp
// Modified here to add operations, add super class.
// License added here for this class for completness.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

// Test for unsigned as signless are treated as signed. For reference, check in
// MLIR AffineToStandard where comparison of indices are done with slt and sgt,
// for example. Indices are signless. Also, in ONNX, we currently treat all
// ONNX Integers as MLIR signless, and only flag the ONNX Unsigned Integer as
// MLIR unsigned integer.

Value MathBuilder::andi(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return b.create<arith::AndIOp>(loc, lhs, rhs);
}

Value MathBuilder::ori(Value lhs, Value rhs) const {
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
  else if (lhs.getType().isUnsignedInteger())
    return b.create<arith::DivUIOp>(loc, lhs, rhs);
  else
    return b.create<arith::DivSIOp>(loc, lhs, rhs);
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

Value MathBuilder::sqrt(Value val) const {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return b.create<math::SqrtOp>(loc, val);
}

Value MathBuilder::pow(Value base, Value exp) const {
  assert(base.getType().isa<FloatType>() && "Data type must be float.");
  return b.create<math::PowFOp>(loc, base, exp);
}

Value MathBuilder::min(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    // Test for unsigned as signless are treated as signed.
    if (lhs.getType().isUnsignedInteger())
      return b.create<arith::MinUIOp>(loc, lhs, rhs);
    else
      return b.create<arith::MinSIOp>(loc, lhs, rhs);
  else
    return b.create<arith::MinFOp>(loc, lhs, rhs);
}

Value MathBuilder::max(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    // Test for unsigned as signless are treated as signed.
    if (lhs.getType().isUnsignedInteger())
      return b.create<arith::MaxUIOp>(loc, lhs, rhs);
    else
      return b.create<arith::MaxSIOp>(loc, lhs, rhs);
  else
    return b.create<arith::MaxFOp>(loc, lhs, rhs);
}

Value MathBuilder::sgt(Value lhs, Value rhs) const {
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sgt);
  return createArithCmp(lhs, rhs, arith::CmpFPredicate::OGT);
}

Value MathBuilder::sge(Value lhs, Value rhs) const {
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sge);
  return createArithCmp(lhs, rhs, arith::CmpFPredicate::OGE);
}

Value MathBuilder::slt(Value lhs, Value rhs) const {
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::slt);
  return createArithCmp(lhs, rhs, arith::CmpFPredicate::OLT);
}

Value MathBuilder::sle(Value lhs, Value rhs) const {
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::sle);
  return createArithCmp(lhs, rhs, arith::CmpFPredicate::OLE);
}

Value MathBuilder::eq(Value lhs, Value rhs) const {
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::eq);
  return createArithCmp(lhs, rhs, arith::CmpFPredicate::OEQ);
}

Value MathBuilder::neq(Value lhs, Value rhs) const {
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return createArithCmp(lhs, rhs, arith::CmpIPredicate::ne);
  return createArithCmp(lhs, rhs, arith::CmpFPredicate::ONE);
}

Value MathBuilder::select(Value cmp, Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return b.create<arith::SelectOp>(loc, cmp, lhs, rhs);
}

Value MathBuilder::constant(Type type, double val) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<Float16Type>([&](Type) {
        constant = b.create<arith::ConstantOp>(loc, b.getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant = b.create<arith::ConstantOp>(loc, b.getF64FloatAttr(val));
      })
      .Case<IntegerType>([&](IntegerType type) {
        assert(val == (int64_t)val && "value is ambiguous");
        unsigned width = type.getWidth();

        if (width == 1)
          constant = b.create<arith::ConstantOp>(loc, b.getBoolAttr(val != 0));
        else {
          assert(type.isSignless() &&
                 "arith::ConstantOp requires a signless type.");
          constant = b.create<arith::ConstantOp>(
              loc, b.getIntegerAttr(type, APInt(width, (int64_t)val)));
        }
      })
      .Case<IndexType>([&](Type) {
        constant =
            b.create<arith::ConstantOp>(loc, b.getIntegerAttr(type, val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value MathBuilder::constantIndex(int64_t val) const {
  Attribute constantAttr = b.getIntegerAttr(b.getIndexType(), val);
  return b.create<arith::ConstantOp>(loc, constantAttr);
}

Value MathBuilder::negativeInf(Type type) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<Float32Type>([&](Type) {
        constant = b.create<arith::ConstantOp>(
            loc, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
      })
      .Case<Float64Type>([&](Type) {
        constant = b.create<arith::ConstantOp>(
            loc, b.getF64FloatAttr(-std::numeric_limits<double>::infinity()));
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
        constant = b.create<arith::ConstantOp>(
            loc, b.getIntegerAttr(type, APInt(width, value)));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value MathBuilder::positiveInf(Type type) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<Float32Type>([&](Type) {
        constant = b.create<arith::ConstantOp>(
            loc, b.getF32FloatAttr(std::numeric_limits<float>::infinity()));
      })
      .Case<Float64Type>([&](Type) {
        constant = b.create<arith::ConstantOp>(
            loc, b.getF64FloatAttr(std::numeric_limits<double>::infinity()));
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
        constant = b.create<arith::ConstantOp>(
            loc, b.getIntegerAttr(type, APInt(width, value)));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value MathBuilder::createArithCmp(
    Value lhs, Value rhs, arith::CmpIPredicate pred) const {
  Type type = lhs.getType();
  assert(type == rhs.getType() && "Operands should have the same type");
  assert(((type.isa<IntegerType>() && type.isSignlessInteger()) ||
             type.isa<IndexType>()) &&
         "Expecting a signless IntegerType or an IndexType");
  return b.create<arith::CmpIOp>(loc, pred, lhs, rhs);
}

Value MathBuilder::createArithCmp(
    Value lhs, Value rhs, arith::CmpFPredicate pred) const {
  Type type = lhs.getType();
  assert(type == rhs.getType() && "Operands should have the same type");
  assert(type.isa<FloatType>() && "Expecting a FloatType");
  return b.create<arith::CmpFOp>(loc, pred, lhs, rhs);
}

// Several operations in the arith dialect require signless integers. This
// cast remove the sign of integer types for successful processing, to the
// best of my understanding.
Value MathBuilder::castToSignless(Value val, int64_t width) const {
  assert(val.getType().isa<IntegerType>() &&
         !val.getType().isSignlessInteger() && "Expecting signed integer type");
  return b.create<UnrealizedConversionCastOp>(loc, b.getIntegerType(width), val)
      .getResult(0);
}

Value MathBuilder::castToUnsigned(Value val, int64_t width) const {
  assert(val.getType().isa<IntegerType>() && "Expecting integer type");
  return b
      .create<UnrealizedConversionCastOp>(
          loc, b.getIntegerType(width, false /*signed*/), val)
      .getResult(0);
}

// Methods inspired from MLIR TosaToLinalg CastOp.
Value MathBuilder::cast(Type destType, Value src) const {
  // Get source type and check if we need a cast at all.
  Type srcType = src.getType();
  if (srcType == destType)
    return src;

  // Process index types first.
  if (srcType.isa<IndexType>()) {
    // If our source is an index type, first convert it into a signless int of
    // size 64.
    srcType = b.getIntegerType(64);
    src = b.create<arith::IndexCastOp>(loc, srcType, src);
  }
  bool destIsIndex = false;
  if (destType.isa<IndexType>()) {
    // If our dest is an index type, pretend for now that we want it to be
    // converted to.
    destType = b.getIntegerType(64);
    destIsIndex = true;
  }

  // Only support Integer or Float type at this stage. Index were transformed
  // to signless int.
  // TODO: add support for shaped tensor (MemRef, Vector, Tensor?) if needed.
  assert((srcType.isa<IntegerType>() || srcType.isa<FloatType>()) &&
         "support only float or int");
  assert((destType.isa<IntegerType>() || destType.isa<FloatType>()) &&
         "support only float or int");
  // Get source and dest type width.
  int64_t srcWidth = srcType.getIntOrFloatBitWidth();
  int64_t destWidth = destType.getIntOrFloatBitWidth();
  bool bitExtend = srcWidth < destWidth;
  bool bitTrunc = srcWidth > destWidth;

  LLVM_DEBUG(llvm::dbgs() << "srcType: " << srcType << "\n";
             llvm::dbgs() << "destType: " << destType << "\n";);

  // Handle boolean first because they need special handling.
  // Boolean to int/float conversions. Boolean are unsigned.
  if (srcType.isInteger(1)) {
    if (destType.isa<FloatType>()) {
      return b.create<arith::UIToFPOp>(loc, destType, src);
    } else {
      Value dest = b.create<arith::ExtUIOp>(loc, destType, src);
      if (destIsIndex)
        dest = b.create<arith::IndexCastOp>(loc, b.getIndexType(), dest);
      return dest;
    }
  }

  // Int/Float to booleans, just compare value to be unequal zero.
  if (destType.isInteger(1)) {
    Type constantType = srcType;
    if (srcType.isa<IntegerType>() && !srcType.isSignlessInteger()) {
      // An integer constant must be signless.
      unsigned srcWidth = srcType.cast<IntegerType>().getWidth();
      constantType = IntegerType::get(srcType.getContext(), srcWidth);
      src = castToSignless(src, srcWidth);
    }
    Value zero = constant(constantType, 0);
    return neq(src, zero);
  }

  // Float to float conversions.
  if (srcType.isa<FloatType>() && destType.isa<FloatType>()) {
    assert((bitExtend || bitTrunc) && "expected extend or trunc");
    if (bitExtend)
      return b.create<arith::ExtFOp>(loc, destType, src);
    else
      return b.create<arith::TruncFOp>(loc, destType, src);
  }

  // Float to int conversions.
  if (srcType.isa<FloatType>() && destType.isa<IntegerType>()) {
    // TosaToLinalg in MLIR uses a fancier algorithm that clamps values to
    // min/max signed/unsigned integer values.
    if (destType.isUnsignedInteger()) {
      Value cast = castToSignless(src, srcWidth);
      return b.create<arith::FPToUIOp>(loc, destType, cast);
    } else {
      // Handle signed int.
      Value dest = b.create<arith::FPToSIOp>(loc, destType, src);
      if (destIsIndex)
        dest = b.create<arith::IndexCastOp>(loc, b.getIndexType(), dest);
      return dest;
    }
  }

  // Int to float conversion.
  if (srcType.isa<IntegerType>() && destType.isa<FloatType>()) {
    if (srcType.isUnsignedInteger()) {
      Value cast = castToSignless(src, srcWidth);
      return b.create<arith::UIToFPOp>(loc, destType, cast);
    } else {
      // Handle signed int.
      return b.create<arith::SIToFPOp>(loc, destType, src);
    }
  }

  // Int to int conversion.
  if (srcType.isa<IntegerType>() && destType.isa<IntegerType>()) {
    if (srcType.isUnsignedInteger()) {
      // Unsigned to unsigned conversion. Has to convert to signless first,
      // and recovert output to unsigned.
      assert(destType.isUnsignedInteger() && "no unsigned/signed conversion");
      assert((bitExtend || bitTrunc) && "expected extend or trunc");
      Value cast = castToSignless(src, srcWidth);
      Type castType = b.getIntegerType(destWidth);
      if (bitExtend) {
        cast = b.create<arith::ExtUIOp>(loc, castType, cast);
      } else {
        // TosaToLinalg use a cliping algo, not sure if needed.
        cast = b.create<arith::TruncIOp>(loc, castType, cast);
      }
      return castToUnsigned(cast, destWidth);
    } else {
      // Handle signed ingeger
      assert(!destType.isUnsignedInteger() && "no signed/unsigned conversion");
      Value dest = src;
      if (bitExtend)
        dest = b.create<arith::ExtSIOp>(loc, destType, src);
      if (bitTrunc)
        // TosaToLinalg use a cliping algo
        dest = b.create<arith::TruncIOp>(loc, destType, src);
      if (destIsIndex)
        dest = b.create<arith::IndexCastOp>(loc, b.getIndexType(), dest);
      return dest;
    }
  }

  // Handled all the cases supported so far.
  llvm_unreachable("unsupported element type");
  return nullptr;
}

Value MathBuilder::castToIndex(Value src) const {
  return cast(b.getIndexType(), src);
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
  return b.create<memref::CastOp>(loc, outputType, input);
}

Value MemRefBuilder::reinterpretCast(
    Value input, SmallVectorImpl<IndexExpr> &outputDims) const {
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
  Type elementType = input.getType().cast<ShapedType>().getElementType();
  MemRefType outputMemRefType = MemRefType::get(outputShape, elementType);

  return b.create<memref::ReinterpretCastOp>(loc, outputMemRefType, input,
      /*offset=*/b.getIndexAttr(0), sizes, strides);
}

Value MemRefBuilder::dim(Value val, int64_t index) const {
  assert((val.getType().isa<MemRefType>() ||
             val.getType().isa<UnrankedMemRefType>()) &&
         "memref::DimOp expects input operand to have MemRefType or "
         "UnrankedMemRefType");
  assert(index >= 0 && "Expecting a valid index");
  return dim(val, b.create<arith::ConstantIndexOp>(loc, index));
}

Value MemRefBuilder::dim(Value val, Value index) const {
  assert((val.getType().isa<MemRefType>() ||
             val.getType().isa<UnrankedMemRefType>()) &&
         "memref::DimOp expects input operand to have MemRefType or "
         "UnrankedMemRefType");
  return b.createOrFold<memref::DimOp>(loc, val, index);
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

//===----------------------------------------------------------------------===//
// Vector Builder
//===----------------------------------------------------------------------===//

int64_t VectorBuilder::getMachineVectorLength(const Type &elementType) const {
  unsigned typeBitSize = elementType.getIntOrFloatBitWidth();
  unsigned simdBitSize;
  // TODO: use march and mcpu to determine the right size, right now assume
  // 4*32=128 bits.
  simdBitSize = 128;
  assert(simdBitSize >= typeBitSize && simdBitSize % typeBitSize == 0 &&
         "bad machine vector length");
  return (simdBitSize / typeBitSize);
}

int64_t VectorBuilder::getMachineVectorLength(const VectorType &vecType) const {
  return getMachineVectorLength(vecType.getElementType());
}

int64_t VectorBuilder::getMachineVectorLength(Value vecValue) const {
  VectorType vecType = vecValue.getType().dyn_cast_or_null<VectorType>();
  assert(vecType && "expected vector type");
  return getMachineVectorLength(vecType.getElementType());
}

Value VectorBuilder::load(
    VectorType vecType, Value memref, ValueRange indices) const {
  return b.create<vector::LoadOp>(loc, vecType, memref, indices);
}
mlir::Value VectorBuilder::load(mlir::VectorType vecType, mlir::Value memref,
    mlir::ValueRange indices, mlir::ValueRange offsets) const {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MathBuilder createMath(*this);
  createMath.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  return load(vecType, memref, computedIndices);
}

mlir::Value VectorBuilder::loadIE(mlir::VectorType vecType, mlir::Value memref,
    llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) const {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MathBuilder createMath(*this);
  createMath.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  return load(vecType, memref, computedIndices);
}

void VectorBuilder::store(Value val, Value memref, ValueRange indices) const {
  b.create<vector::StoreOp>(loc, val, memref, indices);
}

void VectorBuilder::store(mlir::Value val, mlir::Value memref,
    mlir::ValueRange indices, mlir::ValueRange offsets) const {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MathBuilder createMath(*this);
  createMath.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  store(val, memref, computedIndices);
}

void VectorBuilder::storeIE(mlir::Value val, mlir::Value memref,
    llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) const {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MathBuilder createMath(*this);
  createMath.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  store(val, memref, computedIndices);
}

Value VectorBuilder::fma(Value lhs, Value rhs, Value acc) const {
  return b.create<vector::FMAOp>(loc, lhs, rhs, acc);
}

Value VectorBuilder::broadcast(VectorType vecType, Value val) const {
  return b.create<vector::BroadcastOp>(loc, vecType, val);
}

Value VectorBuilder::shuffle(
    Value lhs, Value rhs, SmallVectorImpl<int64_t> &mask) const {
  return b.create<vector::ShuffleOp>(loc, lhs, rhs, mask);
}

// Private vector utilities.
bool VectorBuilder::isPowerOf2(uint64_t num) const {
  return (num & (num - 1)) == 0;
}

uint64_t VectorBuilder::getLengthOf1DVector(Value vec) const {
  VectorType vecType = vec.getType().dyn_cast_or_null<VectorType>();
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

// Do a parallel-simd reduction of N vectors of SIMD length VL.
// Restrictions:
// *  VL is the vector length of the machine SIMD vectors.
// *  N is a multiple of VL as we can perform consecutive VL x VL
//    reductions.
void VectorBuilder::multiReduction(SmallVectorImpl<Value> &inputVecArray,
    SmallVectorImpl<Value> &outputVecArray) {
  uint64_t N = inputVecArray.size();
  assert(N > 0 && "expected at least one value to reduce");
  uint64_t VL = getLengthOf1DVector(inputVecArray[0]);
  uint64_t machineVL = getMachineVectorLength(inputVecArray[0]);
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
  MathBuilder createMath(*this);
  for (uint64_t r = 0; r < N; r += machineVL) {
    // Algorithm for the set of input arrays from tmp[r] to tmp[r+machineVL-1].
    uint64_t numPairs = machineVL / 2; // Pair number decrease by power of 2.
    for (uint64_t step = 1; step < machineVL; step = step * 2) {
      for (uint64_t p = 0; p < numPairs; ++p) {
        Value highVal =
            mergeHigh(tmpArray[r + 2 * p], tmpArray[r + 2 * p + 1], step);
        Value lowVal =
            mergeLow(tmpArray[r + 2 * p], tmpArray[r + 2 * p + 1], step);
        Value red = createMath.add(highVal, lowVal);
        tmpArray[r + p] = red;
      }
      numPairs = numPairs / 2; // Pair number decrease by power of 2.
    }
    // Completed the machineVL x machineVL reduction, save it in the output.
    outputVecArray.emplace_back(tmpArray[r]);
  }
}

//===----------------------------------------------------------------------===//
// LLVM Builder
//===----------------------------------------------------------------------===//

Value LLVMBuilder::addressOf(LLVM::GlobalOp op) const {
  return b.create<LLVM::AddressOfOp>(loc, op);
}

Value LLVMBuilder::_alloca(
    Type resultType, Value size, int64_t alignment) const {
  return b.create<LLVM::AllocaOp>(loc, resultType, size, alignment);
}

Value LLVMBuilder::bitcast(Type type, Value val) const {
  return b.create<LLVM::BitcastOp>(loc, type, val);
}

Value LLVMBuilder::bitcastI8Ptr(Value val) const {
  return b.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMPointerType::get(b.getI8Type()), val);
}

Value LLVMBuilder::bitcastI8PtrPtr(Value val) const {
  return b.create<LLVM::BitcastOp>(loc,
      LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(b.getI8Type())),
      val);
}

void LLVMBuilder::br(ArrayRef<Value> destOperands, Block *destBlock) const {
  b.create<LLVM::BrOp>(loc, destOperands, destBlock);
}

Value LLVMBuilder::call(ArrayRef<Type> resultTypes, StringRef funcName,
    ArrayRef<Value> inputs) const {
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  LLVM::CallOp callOp =
      b.create<LLVM::CallOp>(loc, resultTypes, funcName, inputs);
  // CallOp may return either 0 or 1 value.
  if (resultTypes.empty())
    return nullptr;
  return callOp.getResult(0);
}

Value LLVMBuilder::call(ArrayRef<Type> resultTypes,
    FlatSymbolRefAttr funcSymbol, ArrayRef<Value> inputs) const {
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  LLVM::CallOp callOp =
      b.create<LLVM::CallOp>(loc, resultTypes, funcSymbol, inputs);
  // CallOp may return either 0 or 1 value.
  if (resultTypes.empty())
    return nullptr;
  return callOp.getResult(0);
}

void LLVMBuilder::condBr(Value cond, Block *trueBlock,
    llvm::ArrayRef<Value> trueOperands, Block *falseBlock,
    llvm::ArrayRef<Value> falseOperands) const {
  b.create<LLVM::CondBrOp>(
      loc, cond, trueBlock, trueOperands, falseBlock, falseOperands);
}

Value LLVMBuilder::constant(Type type, int64_t val) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<IntegerType>([&](IntegerType type) {
        unsigned width = type.getWidth();
        if (width == 1)
          constant =
              b.create<LLVM::ConstantOp>(loc, type, b.getBoolAttr(val != 0));
        else {
          assert(type.isSignless() &&
                 "LLVM::ConstantOp requires a signless type.");
          constant = b.create<LLVM::ConstantOp>(
              loc, type, b.getIntegerAttr(type, APInt(width, (int64_t)val)));
        }
      })
      .Case<IndexType>([&](Type) {
        constant =
            b.create<LLVM::ConstantOp>(loc, type, b.getIntegerAttr(type, val));
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
            b.create<LLVM::ConstantOp>(loc, type, b.getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant =
            b.create<LLVM::ConstantOp>(loc, type, b.getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant =
            b.create<LLVM::ConstantOp>(loc, type, b.getF64FloatAttr(val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value LLVMBuilder::extractValue(
    Type resultType, Value container, ArrayRef<int64_t> position) const {
  ArrayAttr posAttr = b.getI64ArrayAttr(position);
  return b.create<LLVM::ExtractValueOp>(loc, resultType, container, posAttr);
}

LLVM::LLVMFuncOp LLVMBuilder::func(StringRef name, Type type) const {
  return b.create<LLVM::LLVMFuncOp>(loc, name, type);
}

Value LLVMBuilder::getElemPtr(
    Type resultType, Value base, ArrayRef<Value> indices) const {
  return b.create<LLVM::GEPOp>(loc, resultType, base, indices);
}

LLVM::GlobalOp LLVMBuilder::globalOp(Type resultType, bool isConstant,
    LLVM::Linkage linkage, StringRef name, Attribute valueAttr,
    uint64_t alignment) const {
  return b.create<LLVM::GlobalOp>(loc, resultType,
      /*isConstant=*/isConstant, linkage, name, valueAttr);
}

Value LLVMBuilder::icmp(LLVM::ICmpPredicate cond, Value lhs, Value rhs) const {
  return b.create<LLVM::ICmpOp>(loc, cond, lhs, rhs);
}

Value LLVMBuilder::insertValue(Type resultType, Value container, Value val,
    llvm::ArrayRef<int64_t> position) const {
  ArrayAttr posAttr = b.getI64ArrayAttr(position);
  return b.create<LLVM::InsertValueOp>(
      loc, resultType, container, val, posAttr);
}

Value LLVMBuilder::load(Value addr) const {
  return b.create<LLVM::LoadOp>(loc, addr);
}

Value LLVMBuilder::null(Type type) const {
  return b.create<LLVM::NullOp>(loc, type);
}

Value LLVMBuilder::nullI8Ptr() const {
  Type I8PtrTy = LLVM::LLVMPointerType::get(b.getI8Type());
  return b.create<LLVM::NullOp>(loc, I8PtrTy);
}

void LLVMBuilder::_return(Value val) const {
  b.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({val}));
}

void LLVMBuilder::store(Value val, Value addr) const {
  b.create<LLVM::StoreOp>(loc, val, addr);
}

FlatSymbolRefAttr LLVMBuilder::getOrInsertSymbolRef(ModuleOp module,
    StringRef funcName, Type resultType, ArrayRef<Type> operandTypes,
    bool isVarArg) const {
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    LLVM::LLVMFunctionType funcType =
        LLVM::LLVMFunctionType::get(resultType, operandTypes, isVarArg);
    b.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);
  }
  return SymbolRefAttr::get(b.getContext(), funcName);
}

void LLVMBuilder::ifThenElse(
    valueFuncRef cond, voidFuncRef thenFn, voidFuncRef elseFn) const {
  LLVMBuilder createLLVM(b, loc);

  // Split the current block into IF, THEN, ELSE and END blocks.
  Block *ifBlock, *thenBlock, *elseBlock, *endBlock;
  ifBlock = b.getInsertionBlock();
  thenBlock = ifBlock->splitBlock(b.getInsertionPoint());
  elseBlock = b.createBlock(
      thenBlock->getParent(), std::next(Region::iterator(thenBlock)));
  if (elseFn)
    endBlock = b.createBlock(
        elseBlock->getParent(), std::next(Region::iterator(elseBlock)));
  else
    endBlock = elseBlock;

  // Emit code for the IF block.
  b.setInsertionPointToEnd(ifBlock);
  Value condVal = cond(createLLVM);

  // Branch the block into the THEN and ELSE blocks.
  createLLVM.condBr(condVal, thenBlock, {}, elseBlock, {});

  // Emit code for the THEN block.
  b.setInsertionPointToStart(thenBlock);
  thenFn(createLLVM);
  if (thenBlock->hasNoSuccessors() && !isa<LLVM::ReturnOp>(thenBlock->back()))
    br({}, endBlock);

  // Emit code for the ELSE block if required.
  b.setInsertionPointToStart(elseBlock);
  if (elseFn) {
    elseFn(createLLVM);
    if (elseBlock->hasNoSuccessors() && !isa<LLVM::ReturnOp>(elseBlock->back()))
      br({}, endBlock);
  }

  // End if-then-else and return to the main body.
  b.setInsertionPointToStart(endBlock);
}

} // namespace onnx_mlir
