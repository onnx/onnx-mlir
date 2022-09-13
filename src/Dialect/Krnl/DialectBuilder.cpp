/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-------------- DialectBuilder.cpp - Krnl Dialect Builder ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file declares helper methods to build Krnl Dialect Ops.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/Krnl/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

static StringRef getFormat(const Type &inputType) {
  StringRef format;
  TypeSwitch<Type>(inputType)
      .Case<Float16Type>([&](Float16Type) { format = "%g"; })
      .Case<Float32Type>([&](Float32Type) { format = "%f"; })
      .Case<Float64Type>([&](Float64Type) { format = "%f"; })
      .Case<IntegerType>([&](IntegerType type) {
        switch (type.getWidth()) {
        case 1:
        case 8:
        case 16:
        case 32:
          format = type.isUnsigned() ? "%u" : "%d";
          break;
        case 64:
          format = type.isUnsigned() ? "%llu" : "%lld";
          break;
        }
      })
      .Case<IndexType>([&](IndexType) { format = "%lld"; })
      .Case<onnx_mlir::krnl::StringType>(
          [&](onnx_mlir::krnl::StringType) { format = "%s"; })
      .Case<LLVM::LLVMPointerType>(
          [&](LLVM::LLVMPointerType) { format = "%s"; })
      .Default([&](Type type) {
        llvm::errs() << "type: " << type << "\n";
        llvm_unreachable("Unhandled type");
      });

  return format;
}

//====---------------- Support for Krnl Builder ----------------------===//

Value KrnlBuilder::load(Value memref, ValueRange indices) const {
  return b.create<KrnlLoadOp>(loc, memref, indices);
}

mlir::Value KrnlBuilder::load(mlir::Value memref, mlir::ValueRange indices,
    mlir::ValueRange offsets) const {
  SmallVector<Value, 4> computedIndices;
  MathBuilder createMath(*this);
  createMath.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  return load(memref, computedIndices);
}

Value KrnlBuilder::loadIE(Value memref, ArrayRef<IndexExpr> indices) const {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  return b.create<KrnlLoadOp>(loc, memref, indexValues);
}

void KrnlBuilder::store(Value val, Value memref, ValueRange indices) const {
  b.create<KrnlStoreOp>(loc, val, memref, indices);
}

void KrnlBuilder::store(mlir::Value val, mlir::Value memref,
    mlir::ValueRange indices, mlir::ValueRange offsets) const {
  SmallVector<Value, 4> computedIndices;
  MathBuilder createMath(*this);
  createMath.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  store(val, memref, computedIndices);
}

void KrnlBuilder::storeIE(
    Value val, Value memref, ArrayRef<IndexExpr> indices) const {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  b.create<KrnlStoreOp>(loc, val, memref, indexValues);
}

Value KrnlBuilder::vectorTypeCast(Value sourceMemref, int64_t vectorLen) const {
  return b.create<KrnlVectorTypeCastOp>(loc, sourceMemref, vectorLen);
}

ValueRange KrnlBuilder::defineLoops(int64_t originalLoopNum) const {
  return b.create<KrnlDefineLoopsOp>(loc, originalLoopNum).getResults();
}

ValueRange KrnlBuilder::block(Value loop, int64_t blockSize) const {
  return b.create<KrnlBlockOp>(loc, loop, blockSize).getResults();
}

void KrnlBuilder::permute(ValueRange loops, ArrayRef<int64_t> map) const {
  b.create<KrnlPermuteOp>(loc, loops, map);
}

ValueRange KrnlBuilder::getInductionVarValue(ValueRange loops) const {
  return b.create<KrnlGetInductionVariableValueOp>(loc, loops).getResults();
}

void KrnlBuilder::iterate(ValueRange originalLoops, ValueRange optimizedLoops,
    ValueRange lbs, ValueRange ubs,
    function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
        bodyBuilderFn) const {
  // Check that originalLoops, lbs, and ubs have the same rank.
  assert(originalLoops.size() == lbs.size() && "expected same rank");
  assert(originalLoops.size() == ubs.size() && "expected same rank");
  ValueRange empty;
  b.create<KrnlIterateOp>(loc, originalLoops, optimizedLoops, lbs, ubs, empty,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        KrnlBuilder createKrnl(builder, loc);
        ValueRange indices = createKrnl.getInductionVarValue(optimizedLoops);
        bodyBuilderFn(createKrnl, indices);
      });
}

KrnlIterateOp KrnlBuilder::iterate(
    const krnl::KrnlIterateOperandPack &operands) const {
  return b.create<KrnlIterateOp>(loc, operands);
}

void KrnlBuilder::iterateIE(ValueRange originalLoops, ValueRange optimizedLoops,
    ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
    function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
        bodyBuilderFn) const {
  // Check that originalLoops, lbs, and ubs have the same rank.
  assert(originalLoops.size() == lbs.size() && "expected same rank");
  assert(originalLoops.size() == ubs.size() && "expected same rank");
  ValueRange empty;
  b.create<KrnlIterateOp>(loc, originalLoops, optimizedLoops, lbs, ubs, empty,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        KrnlBuilder createKrnl(builder, loc);
        ValueRange indices = createKrnl.getInductionVarValue(optimizedLoops);
        bodyBuilderFn(createKrnl, indices);
      });
}

void KrnlBuilder::copyToBuffer(Value bufferMemref, Value sourceMemref,
    ValueRange starts, Value padValue, ArrayRef<int64_t> tileSize,
    ArrayRef<int64_t> padToNext, bool transpose) const {
  b.create<KrnlCopyToBufferOp>(loc, bufferMemref, sourceMemref, starts,
      padValue, tileSize, padToNext, transpose);
}

void KrnlBuilder::copyToBuffer(Value bufferMemref, Value sourceMemref,
    ValueRange starts, Value padValue, bool transpose) const {
  b.create<KrnlCopyToBufferOp>(
      loc, bufferMemref, sourceMemref, starts, padValue, transpose);
}

void KrnlBuilder::copyFromBuffer(Value bufferMemref, Value memref,
    ValueRange starts, ArrayRef<int64_t> tileSize) const {
  b.create<KrnlCopyFromBufferOp>(loc, bufferMemref, memref, starts, tileSize);
}

void KrnlBuilder::copyFromBuffer(
    Value bufferMemref, Value memref, ValueRange starts) const {
  b.create<KrnlCopyFromBufferOp>(loc, bufferMemref, memref, starts);
}

void KrnlBuilder::matmul(Value A, ValueRange aStart, Value B, ValueRange bStart,
    Value C, ValueRange cStart, ValueRange loops, ValueRange computeStarts,
    ValueRange globalUBs, ArrayRef<int64_t> computeTileSize,
    ArrayRef<int64_t> aTileSize, ArrayRef<int64_t> bTileSize,
    ArrayRef<int64_t> cTileSize, bool simdize, bool unroll,
    bool overCompute) const {
  b.create<KrnlMatMulOp>(loc, A, aStart, B, bStart, C, cStart, loops,
      computeStarts[0], computeStarts[1], computeStarts[2], globalUBs[0],
      globalUBs[1], globalUBs[2], computeTileSize, aTileSize, bTileSize,
      cTileSize, simdize, unroll, overCompute);
}

void KrnlBuilder::matmul(Value A, ValueRange aStart, Value B, ValueRange bStart,
    Value C, ValueRange cStart, ValueRange loops, ValueRange computeStarts,
    ValueRange globalUBs, bool simdize, bool unroll, bool overCompute) const {
  b.create<KrnlMatMulOp>(loc, A, aStart, B, bStart, C, cStart, loops,
      computeStarts[0], computeStarts[1], computeStarts[2], globalUBs[0],
      globalUBs[1], globalUBs[2], simdize, unroll, overCompute);
}

Value KrnlBuilder::dim(Type type, Value alloc, Value index) const {
  return b.create<KrnlDimOp>(loc, type, alloc, index);
}

KrnlMovableOp KrnlBuilder::movable() const {
  return b.create<KrnlMovableOp>(loc);
}

KrnlGetRefOp KrnlBuilder::getRef(
    Type type, Value memref, Value offset, ValueRange indices) const {
  return b.create<KrnlGetRefOp>(loc, type, memref, offset, indices);
}

Value KrnlBuilder::constant(MemRefType type, StringRef name,
    Optional<Attribute> value, Optional<IntegerAttr> offset,
    Optional<IntegerAttr> alignment) const {
  static int32_t constantID = 0;
  return b.create<KrnlGlobalOp>(loc, type, b.getI64ArrayAttr(type.getShape()),
      b.getStringAttr(name + std::to_string(constantID++)),
      value.value_or(nullptr), offset.value_or(nullptr),
      alignment.value_or(nullptr));
}

void KrnlBuilder::memcpy(Value dest, Value src, Value size) const {
  b.create<KrnlMemcpyOp>(loc, dest, src, size);
}

void KrnlBuilder::memset(Value dest, Value val, bool delayed) const {
  b.create<KrnlMemsetOp>(loc, dest, val, b.getBoolAttr(delayed));
}

Value KrnlBuilder::strncmp(Value str1, Value str2, Value len) const {
  return b.create<KrnlStrncmpOp>(loc, b.getI32Type(), str1, str2, len);
}

Value KrnlBuilder::strlen(Value str) const {
  return b.create<KrnlStrlenOp>(loc, b.getI64Type(), str);
}

void KrnlBuilder::randomNormal(Value alloc, Value numberOfRandomValues,
    Value mean, Value scale, Value seed) const {
  b.create<KrnlRandomNormalOp>(
      loc, alloc, numberOfRandomValues, mean, scale, seed);
}

Value KrnlBuilder::findIndex(Value input, Value G, Value V, Value len) const {
  return b.create<KrnlFindIndexOp>(loc, b.getIndexType(), input, G, V, len);
}

void KrnlBuilder::printTensor(StringRef msg, Value input) const {
  b.create<KrnlPrintTensorOp>(loc, msg, input);
}

void KrnlBuilder::printf(StringRef msg) const {
  Value noneValue;
  b.create<KrnlPrintOp>(loc, msg, noneValue);
}

void KrnlBuilder::printf(
    StringRef msg, Value input, Type inputType, bool endsWithNewLine) const {
  StringRef format = getFormat(inputType);
  std::string concat(msg.str() + format.str() + (endsWithNewLine ? "\n" : ""));
  StringRef newFormat(concat);
  b.create<KrnlPrintOp>(loc, newFormat, input);
}

void KrnlBuilder::printf(Value input, Type inputType) const {
  StringRef format = getFormat(inputType);
  b.create<KrnlPrintOp>(loc, format, input);
}

} // namespace onnx_mlir
