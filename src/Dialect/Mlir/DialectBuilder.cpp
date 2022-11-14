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

// hi alex: todo, we should not have krnl dependences here.
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

#define DEBUG_TYPE "dialect_builder"

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

Value MathBuilder::abs(Value val) const {
  if (val.getType().isa<IntegerType>() || val.getType().isa<IndexType>())
    return bbbb().create<math::AbsIOp>(llll(), val);
  return bbbb().create<math::AbsFOp>(llll(), val);
}

Value MathBuilder::andi(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return bbbb().create<arith::AndIOp>(llll(), lhs, rhs);
}

Value MathBuilder::ori(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  return bbbb().create<arith::OrIOp>(llll(), lhs, rhs);
}

Value MathBuilder::add(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return bbbb().create<arith::AddIOp>(llll(), lhs, rhs);
  return bbbb().create<arith::AddFOp>(llll(), lhs, rhs);
}

Value MathBuilder::sub(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return bbbb().create<arith::SubIOp>(llll(), lhs, rhs);
  return bbbb().create<arith::SubFOp>(llll(), lhs, rhs);
}

Value MathBuilder::mul(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    return bbbb().create<arith::MulIOp>(llll(), lhs, rhs);
  return bbbb().create<arith::MulFOp>(llll(), lhs, rhs);
}

Value MathBuilder::div(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<FloatType>())
    return bbbb().create<arith::DivFOp>(llll(), lhs, rhs);
  else if (lhs.getType().isUnsignedInteger())
    return bbbb().create<arith::DivUIOp>(llll(), lhs, rhs);
  else
    return bbbb().create<arith::DivSIOp>(llll(), lhs, rhs);
}

Value MathBuilder::exp(Value val) const {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return bbbb().create<math::ExpOp>(llll(), val);
}

Value MathBuilder::exp2(Value val) const {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return bbbb().create<math::Exp2Op>(llll(), val);
}

Value MathBuilder::log2(Value val) const {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return bbbb().create<math::Log2Op>(llll(), val);
}

Value MathBuilder::sqrt(Value val) const {
  assert(val.getType().isa<FloatType>() && "Data type must be float.");
  return bbbb().create<math::SqrtOp>(llll(), val);
}

Value MathBuilder::pow(Value base, Value exp) const {
  assert(base.getType().isa<FloatType>() && "Data type must be float.");
  return bbbb().create<math::PowFOp>(llll(), base, exp);
}

Value MathBuilder::min(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    // Test for unsigned as signless are treated as signed.
    if (lhs.getType().isUnsignedInteger())
      return bbbb().create<arith::MinUIOp>(llll(), lhs, rhs);
    else
      return bbbb().create<arith::MinSIOp>(llll(), lhs, rhs);
  else
    return bbbb().create<arith::MinFOp>(llll(), lhs, rhs);
}

Value MathBuilder::max(Value lhs, Value rhs) const {
  assert(lhs.getType() == rhs.getType() && "expected same type");
  if (lhs.getType().isa<IntegerType>() || lhs.getType().isa<IndexType>())
    // Test for unsigned as signless are treated as signed.
    if (lhs.getType().isUnsignedInteger())
      return bbbb().create<arith::MaxUIOp>(llll(), lhs, rhs);
    else
      return bbbb().create<arith::MaxSIOp>(llll(), lhs, rhs);
  else
    return bbbb().create<arith::MaxFOp>(llll(), lhs, rhs);
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
  return bbbb().create<arith::SelectOp>(llll(), cmp, lhs, rhs);
}

Value MathBuilder::constant(Type type, double val) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<Float16Type>([&](Type) {
        constant = bbbb().create<arith::ConstantOp>(llll(), bbbb().getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant = bbbb().create<arith::ConstantOp>(llll(), bbbb().getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant = bbbb().create<arith::ConstantOp>(llll(), bbbb().getF64FloatAttr(val));
      })
      .Case<IntegerType>([&](IntegerType type) {
        assert(val == (int64_t)val && "value is ambiguous");
        unsigned width = type.getWidth();

        if (width == 1)
          constant = bbbb().create<arith::ConstantOp>(llll(), bbbb().getBoolAttr(val != 0));
        else {
          assert(type.isSignless() &&
                 "arith::ConstantOp requires a signless type.");
          constant = bbbb().create<arith::ConstantOp>(
              llll(), bbbb().getIntegerAttr(type, APInt(width, (int64_t)val)));
        }
      })
      .Case<IndexType>([&](Type) {
        constant =
            bbbb().create<arith::ConstantOp>(llll(), bbbb().getIntegerAttr(type, val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value MathBuilder::constantIndex(int64_t val) const {
  Attribute constantAttr = bbbb().getIntegerAttr(bbbb().getIndexType(), val);
  return bbbb().create<arith::ConstantOp>(llll(), constantAttr);
}

Value MathBuilder::negativeInf(Type type) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<Float32Type>([&](Type) {
        constant = bbbb().create<arith::ConstantOp>(
            llll(), bbbb().getF32FloatAttr(-std::numeric_limits<float>::infinity()));
      })
      .Case<Float64Type>([&](Type) {
        constant = bbbb().create<arith::ConstantOp>(
            llll(), bbbb().getF64FloatAttr(-std::numeric_limits<double>::infinity()));
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
        constant = bbbb().create<arith::ConstantOp>(
            llll(), bbbb().getIntegerAttr(type, APInt(width, value)));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value MathBuilder::positiveInf(Type type) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<Float32Type>([&](Type) {
        constant = bbbb().create<arith::ConstantOp>(
            llll(), bbbb().getF32FloatAttr(std::numeric_limits<float>::infinity()));
      })
      .Case<Float64Type>([&](Type) {
        constant = bbbb().create<arith::ConstantOp>(
            llll(), bbbb().getF64FloatAttr(std::numeric_limits<double>::infinity()));
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
        constant = bbbb().create<arith::ConstantOp>(
            llll(), bbbb().getIntegerAttr(type, APInt(width, value)));
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
  return bbbb().create<arith::CmpIOp>(llll(), pred, lhs, rhs);
}

Value MathBuilder::createArithCmp(
    Value lhs, Value rhs, arith::CmpFPredicate pred) const {
  Type type = lhs.getType();
  assert(type == rhs.getType() && "Operands should have the same type");
  assert(type.isa<FloatType>() && "Expecting a FloatType");
  return bbbb().create<arith::CmpFOp>(llll(), pred, lhs, rhs);
}

// Several operations in the arith dialect require signless integers. This
// cast remove the sign of integer types for successful processing, to the
// best of my understanding.
Value MathBuilder::castToSignless(Value val, int64_t width) const {
  assert(val.getType().isa<IntegerType>() &&
         !val.getType().isSignlessInteger() && "Expecting signed integer type");
  return bbbb().create<UnrealizedConversionCastOp>(llll(), bbbb().getIntegerType(width), val)
      .getResult(0);
}

Value MathBuilder::castToUnsigned(Value val, int64_t width) const {
  assert(val.getType().isa<IntegerType>() && "Expecting integer type");
  return bbbb()
      .create<UnrealizedConversionCastOp>(
          llll(), bbbb().getIntegerType(width, false /*signed*/), val)
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
    srcType = bbbb().getIntegerType(64);
    src = bbbb().create<arith::IndexCastOp>(llll(), srcType, src);
  }
  bool destIsIndex = false;
  if (destType.isa<IndexType>()) {
    // If our dest is an index type, pretend for now that we want it to be
    // converted to.
    destType = bbbb().getIntegerType(64);
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
      return bbbb().create<arith::UIToFPOp>(llll(), destType, src);
    } else {
      Value dest = bbbb().create<arith::ExtUIOp>(llll(), destType, src);
      if (destIsIndex)
        dest = bbbb().create<arith::IndexCastOp>(llll(), bbbb().getIndexType(), dest);
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
      return bbbb().create<arith::ExtFOp>(llll(), destType, src);
    else
      return bbbb().create<arith::TruncFOp>(llll(), destType, src);
  }

  // Float to int conversions.
  if (srcType.isa<FloatType>() && destType.isa<IntegerType>()) {
    // TosaToLinalg in MLIR uses a fancier algorithm that clamps values to
    // min/max signed/unsigned integer values.
    if (destType.isUnsignedInteger()) {
      Value cast = castToSignless(src, srcWidth);
      return bbbb().create<arith::FPToUIOp>(llll(), destType, cast);
    } else {
      // Handle signed int.
      Value dest = bbbb().create<arith::FPToSIOp>(llll(), destType, src);
      if (destIsIndex)
        dest = bbbb().create<arith::IndexCastOp>(llll(), bbbb().getIndexType(), dest);
      return dest;
    }
  }

  // Int to float conversion.
  if (srcType.isa<IntegerType>() && destType.isa<FloatType>()) {
    if (srcType.isUnsignedInteger()) {
      Value cast = castToSignless(src, srcWidth);
      return bbbb().create<arith::UIToFPOp>(llll(), destType, cast);
    } else {
      // Handle signed int.
      return bbbb().create<arith::SIToFPOp>(llll(), destType, src);
    }
  }

  // Int to int conversion.
  if (srcType.isa<IntegerType>() && destType.isa<IntegerType>()) {
    if (srcType.isUnsignedInteger()) {
      // Unsigned to unsigned conversion. Has to convert to signless first,
      // and reconvert output to unsigned.
      assert(destType.isUnsignedInteger() && "no unsigned/signed conversion");
      assert((bitExtend || bitTrunc) && "expected extend or trunc");
      Value cast = castToSignless(src, srcWidth);
      Type castType = bbbb().getIntegerType(destWidth);
      if (bitExtend) {
        cast = bbbb().create<arith::ExtUIOp>(llll(), castType, cast);
      } else {
        // TosaToLinalg use a clipping algo, not sure if needed.
        cast = bbbb().create<arith::TruncIOp>(llll(), castType, cast);
      }
      return castToUnsigned(cast, destWidth);
    } else {
      // Handle signed integer
      assert(!destType.isUnsignedInteger() && "no signed/unsigned conversion");
      Value dest = src;
      if (bitExtend)
        dest = bbbb().create<arith::ExtSIOp>(llll(), destType, src);
      if (bitTrunc)
        // TosaToLinalg use a clipping algo
        dest = bbbb().create<arith::TruncIOp>(llll(), destType, src);
      if (destIsIndex)
        dest = bbbb().create<arith::IndexCastOp>(llll(), bbbb().getIndexType(), dest);
      return dest;
    }
  }

  // Handled all the cases supported so far.
  llvm_unreachable("unsupported element type");
  return nullptr;
}

Value MathBuilder::castToIndex(Value src) const {
  return cast(bbbb().getIndexType(), src);
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
  return bbbb().create<memref::AllocOp>(llll(), type, dynSymbols);
}

memref::AllocOp MemRefBuilder::alloc(MemRefType type) const {
  return bbbb().create<memref::AllocOp>(llll(), type);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, int64_t alignment) const {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  IntegerAttr alignmentAttr = bbbb().getI64IntegerAttr(alignment);
  if (type.getShape().size() == 0) // Drop align for scalars.
    return bbbb().create<memref::AllocOp>(llll(), type);
  return bbbb().create<memref::AllocOp>(llll(), type, alignmentAttr);
}

memref::AllocOp MemRefBuilder::alignedAlloc(
    MemRefType type, ValueRange dynSymbols, int64_t alignment) const {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  IntegerAttr alignmentAttr = bbbb().getI64IntegerAttr(alignment);
  if (type.getShape().size() == 0) // Drop align for scalars.
    return bbbb().create<memref::AllocOp>(llll(), type, dynSymbols);
  return bbbb().create<memref::AllocOp>(llll(), type, dynSymbols, alignmentAttr);
}

memref::AllocaOp MemRefBuilder::alloca(MemRefType type) const {
  return bbbb().create<memref::AllocaOp>(llll(), type);
}

memref::AllocaOp MemRefBuilder::alignedAlloca(
    MemRefType type, int64_t alignment) const {
  alignment = (alignment > gDefaultAllocAlign ? alignment : gDefaultAllocAlign);
  IntegerAttr alignmentAttr = bbbb().getI64IntegerAttr(alignment);
  if (type.getShape().size() == 0) // Drop align for scalars.
    return bbbb().create<memref::AllocaOp>(llll(), type);
  return bbbb().create<memref::AllocaOp>(llll(), type, alignmentAttr);
}

memref::DeallocOp MemRefBuilder::dealloc(Value val) const {
  return bbbb().create<memref::DeallocOp>(llll(), val);
}

memref::CastOp MemRefBuilder::cast(Value input, MemRefType outputType) const {
  return bbbb().create<memref::CastOp>(llll(), outputType, input);
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

  return bbbb().create<memref::ReinterpretCastOp>(llll(), outputMemRefType, input,
      /*offset=*/bbbb().getIndexAttr(0), sizes, strides);
}

Value MemRefBuilder::dim(Value val, int64_t index) const {
  assert((val.getType().isa<MemRefType>() ||
             val.getType().isa<UnrankedMemRefType>()) &&
         "memref::DimOp expects input operand to have MemRefType or "
         "UnrankedMemRefType");
  assert(index >= 0 && "Expecting a valid index");
  return dim(val, bbbb().create<arith::ConstantIndexOp>(llll(), index));
}

Value MemRefBuilder::dim(Value val, Value index) const {
  assert((val.getType().isa<MemRefType>() ||
             val.getType().isa<UnrankedMemRefType>()) &&
         "memref::DimOp expects input operand to have MemRefType or "
         "UnrankedMemRefType");
  return bbbb().createOrFold<memref::DimOp>(llll(), val, index);
}

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF).
//===----------------------------------------------------------------------===//

void SCFBuilder::ifThenElse(Value cond,
    function_ref<void(SCFBuilder &createSCF)> thenFn,
    function_ref<void(SCFBuilder &createSCF)> elseFn) const {
  if (!elseFn) {
    bbbb().create<scf::IfOp>(llll(), cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          thenFn(scfBuilder);
          yield();
        });
  } else {
    bbbb().create<scf::IfOp>(
        llll(), cond,
        /* then */
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          thenFn(scfBuilder);
          bbbb().create<scf::YieldOp>(llll());
        },
        /*else*/
        [&](OpBuilder &childBuilder, Location childLoc) {
          SCFBuilder scfBuilder(childBuilder, childLoc);
          elseFn(scfBuilder);
          yield();
        });
  }
}

void SCFBuilder::parallelLoop(ValueRange lowerBounds, ValueRange upperBounds,
    ValueRange steps,
    function_ref<void(DialectBuilder &createKrnl, ValueRange)> bodyFn) const {
  // SmallVectorImpl<Value> ivStorage;
  bbbb().create<scf::ParallelOp>(llll(), lowerBounds, upperBounds, steps,
      [&](OpBuilder &childBuilder, Location childLoc,
          ValueRange inductionVars) {
        KrnlBuilder builder(childBuilder, childLoc);
        bodyFn(builder, inductionVars);
        yield();
      });
}

void SCFBuilder::yield() const { bbbb().create<scf::YieldOp>(llll()); }

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
  return bbbb().create<vector::LoadOp>(llll(), vecType, memref, indices);
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
  bbbb().create<vector::StoreOp>(llll(), val, memref, indices);
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
  return bbbb().create<vector::FMAOp>(llll(), lhs, rhs, acc);
}

Value VectorBuilder::broadcast(VectorType vecType, Value val) const {
  return bbbb().create<vector::BroadcastOp>(llll(), vecType, val);
}

Value VectorBuilder::shuffle(
    Value lhs, Value rhs, SmallVectorImpl<int64_t> &mask) const {
  return bbbb().create<vector::ShuffleOp>(llll(), lhs, rhs, mask);
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
  return bbbb().create<LLVM::AddressOfOp>(llll(), op);
}

Value LLVMBuilder::_alloca(
    Type resultType, Value size, int64_t alignment) const {
  return bbbb().create<LLVM::AllocaOp>(llll(), resultType, size, alignment);
}

Value LLVMBuilder::bitcast(Type type, Value val) const {
  return bbbb().create<LLVM::BitcastOp>(llll(), type, val);
}

Value LLVMBuilder::bitcastI8Ptr(Value val) const {
  return bbbb().create<LLVM::BitcastOp>(
      llll(), LLVM::LLVMPointerType::get(bbbb().getI8Type()), val);
}

Value LLVMBuilder::bitcastI8PtrPtr(Value val) const {
  return bbbb().create<LLVM::BitcastOp>(llll(),
      LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(bbbb().getI8Type())),
      val);
}

void LLVMBuilder::br(ArrayRef<Value> destOperands, Block *destBlock) const {
  bbbb().create<LLVM::BrOp>(llll(), destOperands, destBlock);
}

Value LLVMBuilder::call(ArrayRef<Type> resultTypes, StringRef funcName,
    ArrayRef<Value> inputs) const {
  assert((resultTypes.size() == 0 || resultTypes.size() == 1) &&
         "LLVM:CallOp must return either 0 or 1 value");
  LLVM::CallOp callOp =
      bbbb().create<LLVM::CallOp>(llll(), resultTypes, funcName, inputs);
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
      bbbb().create<LLVM::CallOp>(llll(), resultTypes, funcSymbol, inputs);
  // CallOp may return either 0 or 1 value.
  if (resultTypes.empty())
    return nullptr;
  return callOp.getResult();
}

void LLVMBuilder::condBr(Value cond, Block *trueBlock,
    llvm::ArrayRef<Value> trueOperands, Block *falseBlock,
    llvm::ArrayRef<Value> falseOperands) const {
  bbbb().create<LLVM::CondBrOp>(
      llll(), cond, trueBlock, trueOperands, falseBlock, falseOperands);
}

Value LLVMBuilder::constant(Type type, int64_t val) const {
  Value constant = nullptr;
  TypeSwitch<Type>(type)
      .Case<IntegerType>([&](IntegerType type) {
        unsigned width = type.getWidth();
        if (width == 1)
          constant =
              bbbb().create<LLVM::ConstantOp>(llll(), type, bbbb().getBoolAttr(val != 0));
        else {
          assert(type.isSignless() &&
                 "LLVM::ConstantOp requires a signless type.");
          constant = bbbb().create<LLVM::ConstantOp>(
              llll(), type, bbbb().getIntegerAttr(type, APInt(width, (int64_t)val)));
        }
      })
      .Case<IndexType>([&](Type) {
        constant =
            bbbb().create<LLVM::ConstantOp>(llll(), type, bbbb().getIntegerAttr(type, val));
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
            bbbb().create<LLVM::ConstantOp>(llll(), type, bbbb().getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        constant =
            bbbb().create<LLVM::ConstantOp>(llll(), type, bbbb().getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        constant =
            bbbb().create<LLVM::ConstantOp>(llll(), type, bbbb().getF64FloatAttr(val));
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  assert(constant != nullptr && "Expecting valid constant value");
  return constant;
}

Value LLVMBuilder::extractValue(
    Type resultType, Value container, ArrayRef<int64_t> position) const {
  return bbbb().create<LLVM::ExtractValueOp>(llll(), resultType, container, position);
}

LLVM::LLVMFuncOp LLVMBuilder::func(StringRef name, Type type) const {
  return bbbb().create<LLVM::LLVMFuncOp>(llll(), name, type);
}

Value LLVMBuilder::getElemPtr(
    Type resultType, Value base, ArrayRef<Value> indices) const {
  return bbbb().create<LLVM::GEPOp>(llll(), resultType, base, indices);
}

LLVM::GlobalOp LLVMBuilder::globalOp(Type resultType, bool isConstant,
    LLVM::Linkage linkage, StringRef name, Attribute valueAttr,
    uint64_t alignment) const {
  return bbbb().create<LLVM::GlobalOp>(llll(), resultType,
      /*isConstant=*/isConstant, linkage, name, valueAttr);
}

Value LLVMBuilder::icmp(LLVM::ICmpPredicate cond, Value lhs, Value rhs) const {
  return bbbb().create<LLVM::ICmpOp>(llll(), cond, lhs, rhs);
}

Value LLVMBuilder::insertValue(Type resultType, Value container, Value val,
    llvm::ArrayRef<int64_t> position) const {
  return bbbb().create<LLVM::InsertValueOp>(
      llll(), resultType, container, val, position);
}

Value LLVMBuilder::load(Value addr) const {
  return bbbb().create<LLVM::LoadOp>(llll(), addr);
}

Value LLVMBuilder::null(Type type) const {
  return bbbb().create<LLVM::NullOp>(llll(), type);
}

Value LLVMBuilder::nullI8Ptr() const {
  Type I8PtrTy = LLVM::LLVMPointerType::get(bbbb().getI8Type());
  return bbbb().create<LLVM::NullOp>(llll(), I8PtrTy);
}

void LLVMBuilder::_return(Value val) const {
  bbbb().create<LLVM::ReturnOp>(llll(), ArrayRef<Value>({val}));
}

void LLVMBuilder::store(Value val, Value addr) const {
  bbbb().create<LLVM::StoreOp>(llll(), val, addr);
}

FlatSymbolRefAttr LLVMBuilder::getOrInsertSymbolRef(ModuleOp module,
    StringRef funcName, Type resultType, ArrayRef<Type> operandTypes,
    bool isVarArg) const {
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    OpBuilder::InsertionGuard guard(bbbb());
    bbbb().setInsertionPointToStart(module.getBody());
    LLVM::LLVMFunctionType funcType =
        LLVM::LLVMFunctionType::get(resultType, operandTypes, isVarArg);
    bbbb().create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);
  }
  return SymbolRefAttr::get(bbbb().getContext(), funcName);
}

void LLVMBuilder::ifThenElse(
    valueFuncRef cond, voidFuncRef thenFn, voidFuncRef elseFn) const {
  LLVMBuilder createLLVM(bbbb(), llll());

  // Split the current block into IF, THEN, ELSE and END blocks.
  Block *ifBlock, *thenBlock, *elseBlock, *endBlock;
  ifBlock = bbbb().getInsertionBlock();
  thenBlock = ifBlock->splitBlock(bbbb().getInsertionPoint());
  elseBlock = bbbb().createBlock(
      thenBlock->getParent(), std::next(Region::iterator(thenBlock)));
  if (elseFn)
    endBlock = bbbb().createBlock(
        elseBlock->getParent(), std::next(Region::iterator(elseBlock)));
  else
    endBlock = elseBlock;

  // Emit code for the IF block.
  bbbb().setInsertionPointToEnd(ifBlock);
  Value condVal = cond(createLLVM);

  // Branch the block into the THEN and ELSE blocks.
  createLLVM.condBr(condVal, thenBlock, {}, elseBlock, {});

  // Emit code for the THEN block.
  bbbb().setInsertionPointToStart(thenBlock);
  thenFn(createLLVM);
  if (thenBlock->hasNoSuccessors() && !isa<LLVM::ReturnOp>(thenBlock->back()))
    br({}, endBlock);

  // Emit code for the ELSE block if required.
  bbbb().setInsertionPointToStart(elseBlock);
  if (elseFn) {
    elseFn(createLLVM);
    if (elseBlock->hasNoSuccessors() && !isa<LLVM::ReturnOp>(elseBlock->back()))
      br({}, endBlock);
  }

  // End if-then-else and return to the main body.
  bbbb().setInsertionPointToStart(endBlock);
}

} // namespace onnx_mlir
