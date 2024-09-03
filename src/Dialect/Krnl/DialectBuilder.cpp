/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-------------- DialectBuilder.cpp - Krnl Dialect Builder ------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file declares helper methods to build Krnl Dialect Ops.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

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

Value KrnlBuilder::load(
    Value memref, ValueRange indices, ValueRange offsets) const {
  return onnx_mlir::impl::load<KrnlBuilder, KrnlLoadOp>(
      *this, memref, indices, offsets);
}

Value KrnlBuilder::loadIE(
    Value memref, ArrayRef<IndexExpr> indices, ValueRange offsets) const {
  return onnx_mlir::impl::loadIE<KrnlBuilder, KrnlLoadOp>(
      *this, memref, indices, offsets);
}

void KrnlBuilder::store(
    Value val, Value memref, ValueRange indices, ValueRange offsets) const {
  onnx_mlir::impl::store<KrnlBuilder, KrnlStoreOp>(
      *this, val, memref, indices, offsets);
}

void KrnlBuilder::storeIE(Value val, Value memref, ArrayRef<IndexExpr> indices,
    ValueRange offsets) const {
  onnx_mlir::impl::storeIE<KrnlBuilder, KrnlStoreOp>(
      *this, val, memref, indices, offsets);
}

Value KrnlBuilder::getLinearOffsetIndex(
    Value memref, ValueRange indices) const {
  return b().create<KrnlGetLinearOffsetIndexOp>(loc(), memref, indices);
}

Value KrnlBuilder::getLinearOffsetIndexIE(
    Value memref, ArrayRef<IndexExpr> indices) const {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  return b().create<KrnlGetLinearOffsetIndexOp>(loc(), memref, indexValues);
}

void KrnlBuilder::prefetch(Value memref, ValueRange indices, bool isWrite,
    unsigned localityHint, bool isDataCache) {
  if (disableMemRefPrefetch)
    return;
  b().create<KrnlPrefetchOp>(
      loc(), memref, indices, isWrite, localityHint, isDataCache);
}

void KrnlBuilder::prefetchIE(Value memref, ArrayRef<IndexExpr> indices,
    bool isWrite, unsigned localityHint, bool isDataCache) {
  if (disableMemRefPrefetch)
    return;
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  b().create<KrnlPrefetchOp>(
      loc(), memref, indexValues, isWrite, localityHint, isDataCache);
}

void KrnlBuilder::seqstore(Value element, Value seq, Value index) const {
  b().create<KrnlSeqStoreOp>(loc(), element, seq, index);
}

void KrnlBuilder::seqstore(Value element, Value seq, IndexExpr index) const {
  b().create<KrnlSeqStoreOp>(loc(), element, seq, index.getValue());
}

Value KrnlBuilder::vectorTypeCast(Value sourceMemref, int64_t vectorLen) const {
  return b().create<KrnlVectorTypeCastOp>(loc(), sourceMemref, vectorLen);
}

void KrnlBuilder::region(
    function_ref<void(KrnlBuilder &createKrnl)> bodyBuilderFn) const {
  KrnlBuilder createKrnl(b(), loc());
  KrnlRegionOp regionOp = b().create<KrnlRegionOp>(loc());
  {
    OpBuilder::InsertionGuard guard(b());
    b().setInsertionPointToStart(&regionOp.getBodyRegion().front());
    bodyBuilderFn(createKrnl);
  }
}

ValueRange KrnlBuilder::defineLoops(int64_t originalLoopNum) const {
  return b()
      .template create<KrnlDefineLoopsOp>(loc(), originalLoopNum)
      .getResults();
}

ValueRange KrnlBuilder::block(Value loop, int64_t blockSize) const {
  return b().create<KrnlBlockOp>(loc(), loop, blockSize).getResults();
}

void KrnlBuilder::permute(ValueRange loops, ArrayRef<int64_t> map) const {
  b().create<KrnlPermuteOp>(loc(), loops, map);
}

ValueRange KrnlBuilder::getInductionVarValue(ValueRange loops) const {
  return b()
      .template create<KrnlGetInductionVariableValueOp>(loc(), loops)
      .getResults();
}

void KrnlBuilder::parallel(ValueRange loops) const {
  b().template create<KrnlParallelOp>(loc(), loops);
}

void KrnlBuilder::iterate(ValueRange originalLoops, ValueRange optimizedLoops,
    ValueRange lbs, ValueRange ubs,
    function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
        bodyBuilderFn) const {
  auto bodyBuilderFnWrapper = [&](KrnlBuilder &createKrnl, ValueRange indices,
                                  ValueRange iterArgs) {
    bodyBuilderFn(createKrnl, indices);
  };
  iterate(originalLoops, optimizedLoops, lbs, ubs, {}, bodyBuilderFnWrapper);
}

KrnlIterateOp KrnlBuilder::iterate(ValueRange originalLoops,
    ValueRange optimizedLoops, ValueRange lbs, ValueRange ubs, ValueRange inits,
    function_ref<void(
        KrnlBuilder &createKrnl, ValueRange indices, ValueRange iterArgs)>
        bodyBuilderFn) const {
  // Check that originalLoops, lbs, and ubs have the same rank.
  assert(originalLoops.size() == lbs.size() && "expected same rank");
  assert(originalLoops.size() == ubs.size() && "expected same rank");
  return b().create<KrnlIterateOp>(loc(), originalLoops, optimizedLoops, lbs,
      ubs, inits,
      [&](OpBuilder &builder, Location loc, ValueRange args,
          ValueRange iterArgs) {
        KrnlBuilder createKrnl(builder, loc);
        ValueRange indices = createKrnl.getInductionVarValue(optimizedLoops);
        bodyBuilderFn(createKrnl, indices, iterArgs);
      });
}

KrnlIterateOp KrnlBuilder::iterate(
    const krnl::KrnlIterateOperandPack &operands) const {
  return b().create<KrnlIterateOp>(loc(), operands);
}

void KrnlBuilder::iterateIE(ValueRange originalLoops, ValueRange optimizedLoops,
    ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
    function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
        bodyBuilderFn) const {
  auto bodyBuilderFnWrapper = [&](KrnlBuilder &createKrnl, ValueRange indices,
                                  ValueRange iterArgs) {
    bodyBuilderFn(createKrnl, indices);
  };
  iterateIE(originalLoops, optimizedLoops, lbs, ubs, {}, bodyBuilderFnWrapper);
}

KrnlIterateOp KrnlBuilder::iterateIE(ValueRange originalLoops,
    ValueRange optimizedLoops, ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
    ValueRange inits,
    function_ref<void(
        KrnlBuilder &createKrnl, ValueRange indices, ValueRange iterArgs)>
        bodyBuilderFn) const {
  // Check that originalLoops, lbs, and ubs have the same rank.
  assert(originalLoops.size() == lbs.size() && "expected same rank");
  assert(originalLoops.size() == ubs.size() && "expected same rank");
  return b().create<KrnlIterateOp>(loc(), originalLoops, optimizedLoops, lbs,
      ubs, inits,
      [&](OpBuilder &builder, Location loc, ValueRange args,
          ValueRange iterArgs) {
        KrnlBuilder createKrnl(builder, loc);
        ValueRange indices = createKrnl.getInductionVarValue(optimizedLoops);
        bodyBuilderFn(createKrnl, indices, iterArgs);
      });
}

void KrnlBuilder::simdIterateIE(IndexExpr lb, IndexExpr ub, int64_t VL,
    bool fullySimd, bool useParallel, ArrayRef<Value> inputs,
    ArrayRef<DimsExpr> inputAFs, ArrayRef<Value> outputs,
    ArrayRef<DimsExpr> outputAFs,
    function_ref<void(KrnlBuilder &b, ArrayRef<Value> inputVals,
        llvm::SmallVectorImpl<Value> &resultVals, int64_t VL)>
        bodyBuilderFn) const {
  onnx_mlir::impl::simdIterateIE<KrnlBuilder, KrnlBuilder>(*this, lb, ub, VL,
      fullySimd, useParallel, inputs, inputAFs, outputs, outputAFs,
      bodyBuilderFn);
}

void KrnlBuilder::simdReduceIE(IndexExpr lb, IndexExpr ub, int64_t VL,
    bool fullySimd, ArrayRef<Value> inputs, ArrayRef<DimsExpr> inputAFs,
    ArrayRef<Value> tmps, ArrayRef<DimsExpr> tmpAFs, ArrayRef<Value> outputs,
    ArrayRef<DimsExpr> outputAFs, ArrayRef<Value> initVals,
    /* reduction function (simd or scalar) */
    function_ref<void(const KrnlBuilder &b, ArrayRef<Value> inputVals,
        ArrayRef<Value> tmpVals, llvm::SmallVectorImpl<Value> &resultVals,
        int64_t VL)>
        reductionBuilderFn,
    /* post reduction function (simd to scalar + post processing)*/
    function_ref<void(const KrnlBuilder &b, ArrayRef<Value> tmpVals,
        llvm::SmallVectorImpl<Value> &scalarOutputs, int64_t VL)>
        postProcessingBuilderFn) const {
  onnx_mlir::impl::simdReduceIE<KrnlBuilder, KrnlBuilder>(*this, lb, ub, VL,
      fullySimd, inputs, inputAFs, tmps, tmpAFs, outputs, outputAFs, initVals,
      reductionBuilderFn, postProcessingBuilderFn);
}

void KrnlBuilder::yield(mlir::ValueRange iterArgs) const {
  b().create<KrnlYieldOp>(loc(), iterArgs);
}

void KrnlBuilder::copyToBuffer(Value bufferMemref, Value sourceMemref,
    ValueRange starts, Value padValue, ArrayRef<int64_t> tileSize,
    ArrayRef<int64_t> padToNext, bool transpose) const {
  b().create<KrnlCopyToBufferOp>(loc(), bufferMemref, sourceMemref, starts,
      padValue, tileSize, padToNext, transpose);
}

void KrnlBuilder::copyToBuffer(Value bufferMemref, Value sourceMemref,
    ValueRange starts, Value padValue, bool transpose) const {
  b().create<KrnlCopyToBufferOp>(
      loc(), bufferMemref, sourceMemref, starts, padValue, transpose);
}

void KrnlBuilder::copyFromBuffer(Value bufferMemref, Value memref,
    ValueRange starts, ArrayRef<int64_t> tileSize) const {
  b().create<KrnlCopyFromBufferOp>(
      loc(), bufferMemref, memref, starts, tileSize);
}

void KrnlBuilder::copyFromBuffer(
    Value bufferMemref, Value memref, ValueRange starts) const {
  b().create<KrnlCopyFromBufferOp>(loc(), bufferMemref, memref, starts);
}

void KrnlBuilder::matmul(Value A, ValueRange aStart, Value B, ValueRange bStart,
    Value C, ValueRange cStart, ValueRange loops, ValueRange computeStarts,
    ValueRange globalUBs, ArrayRef<int64_t> computeTileSize,
    ArrayRef<int64_t> aTileSize, ArrayRef<int64_t> bTileSize,
    ArrayRef<int64_t> cTileSize, bool simdize, bool unroll,
    bool overCompute) const {
  b().create<KrnlMatMulOp>(loc(), A, aStart, B, bStart, C, cStart, loops,
      computeStarts[0], computeStarts[1], computeStarts[2], globalUBs[0],
      globalUBs[1], globalUBs[2], computeTileSize, aTileSize, bTileSize,
      cTileSize, simdize, unroll, overCompute);
}

void KrnlBuilder::matmul(Value A, ValueRange aStart, Value B, ValueRange bStart,
    Value C, ValueRange cStart, ValueRange loops, ValueRange computeStarts,
    ValueRange globalUBs, bool simdize, bool unroll, bool overCompute) const {
  b().create<KrnlMatMulOp>(loc(), A, aStart, B, bStart, C, cStart, loops,
      computeStarts[0], computeStarts[1], computeStarts[2], globalUBs[0],
      globalUBs[1], globalUBs[2], simdize, unroll, overCompute);
}

KrnlMovableOp KrnlBuilder::movable() const {
  return b().create<KrnlMovableOp>(loc());
}

Value KrnlBuilder::constant(MemRefType type, StringRef name,
    std::optional<Attribute> value, std::optional<IntegerAttr> offset,
    std::optional<IntegerAttr> alignment) const {
  static int32_t constantID = 0;
  return b().create<KrnlGlobalOp>(loc(), type,
      b().getI64ArrayAttr(type.getShape()),
      b().getStringAttr(name + std::to_string(constantID++)),
      value.value_or(nullptr), offset.value_or(nullptr),
      alignment.value_or(nullptr));
}

void KrnlBuilder::memcpy(Value dest, Value src, Value numElems) const {
  MultiDialectBuilder<MathBuilder> create(*this);
  Value zero = create.math.constantIndex(0);
  b().create<KrnlMemcpyOp>(loc(), dest, src, numElems,
      /*dest_offset=*/zero, /*src_offset=*/zero);
}

void KrnlBuilder::memcpy(Value dest, Value src, Value numElems,
    Value destOffset, Value srcOffset) const {
  b().create<KrnlMemcpyOp>(loc(), dest, src, numElems, destOffset, srcOffset);
}

void KrnlBuilder::memset(Value dest, Value val, bool delayed) const {
  b().create<KrnlMemsetOp>(loc(), dest, val, b().getBoolAttr(delayed));
}

Value KrnlBuilder::strncmp(Value str1, Value str2, Value len) const {
  return b().create<KrnlStrncmpOp>(loc(), b().getI32Type(), str1, str2, len);
}

Value KrnlBuilder::strlen(Value str) const {
  return b().create<KrnlStrlenOp>(loc(), b().getI64Type(), str);
}

void KrnlBuilder::randomNormal(Value alloc, Value numberOfRandomValues,
    Value mean, Value scale, Value seed) const {
  b().create<KrnlRandomNormalOp>(
      loc(), alloc, numberOfRandomValues, mean, scale, seed);
}

Value KrnlBuilder::findIndex(Value input, Value G, Value V, Value len) const {
  return b().create<KrnlFindIndexOp>(
      loc(), b().getIndexType(), input, G, V, len);
}

void KrnlBuilder::printTensor(StringRef msg, Value input) const {
  b().create<KrnlPrintTensorOp>(loc(), msg, input);
}

void KrnlBuilder::printf(StringRef msg) const {
  Value noneValue;
  b().create<KrnlPrintOp>(loc(), msg, noneValue);
}

void KrnlBuilder::printf(
    StringRef msg, Value input, Type inputType, bool endsWithNewLine) const {
  StringRef format = getFormat(inputType);
  std::string concat(msg.str() + format.str() + (endsWithNewLine ? "\n" : ""));
  StringRef newFormat(concat);
  b().create<KrnlPrintOp>(loc(), newFormat, input);
}

void KrnlBuilder::printf(
    StringRef msg, Value input, bool endsWithNewLine) const {
  KrnlBuilder::printf(msg, input, input.getType(), endsWithNewLine);
}

void KrnlBuilder::printf(
    StringRef msg, IndexExpr input, bool endsWithNewLine) const {
  KrnlBuilder::printf(msg, input.getValue(), endsWithNewLine);
}

// =============================================================================
// IndexExpr Builder for Analysis
// =============================================================================

// Return null if none is found.
ElementsAttr IndexExprBuilderForKrnl::getConst(Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto globalOp = dyn_cast_or_null<KrnlGlobalOp>(definingOp)) {
    if (globalOp.getValue().has_value())
      return mlir::dyn_cast<ElementsAttr>(globalOp.getValueAttr());
  } else if (auto globalOp = dyn_cast_or_null<ONNXConstantOp>(definingOp)) {
    if (globalOp.getValue().has_value())
      return mlir::dyn_cast<ElementsAttr>(globalOp.getValueAttr());
  }
  return nullptr;
}

Value IndexExprBuilderForKrnl::getVal(Value intArrayVal, uint64_t i) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(*this);
  uint64_t rank = getShapedTypeRank(intArrayVal);
  if (rank == 0)
    return create.krnl.load(intArrayVal);
  uint64_t size = getArraySize(intArrayVal);
  assert(i < size && "out of bound reference");
  Value iVal = create.math.constantIndex(i);
  return create.krnl.load(intArrayVal, {iVal});
}

Value IndexExprBuilderForKrnl::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  MemRefBuilder createMemRef(*this);
  return createMemRef.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
