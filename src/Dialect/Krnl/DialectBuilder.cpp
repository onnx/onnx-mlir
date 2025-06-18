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
    function_ref<void(const KrnlBuilder &createKrnl)> bodyBuilderFn) const {
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
  Value noneValue;
  StringAttr noneStrAttr;
  b().template create<KrnlParallelOp>(loc(), loops, noneValue, noneStrAttr);
}

void KrnlBuilder::parallel(
    ValueRange loops, Value numThreads, StringAttr procBind) const {
  if (procBind.getValue().size() > 0) {
    std::string str = procBind.getValue().str();
    assert((str == "primary" || str == "close" || str == "spread") &&
           "expected primary, close, or spread for proc_bind");
  }
  b().template create<KrnlParallelOp>(loc(), loops, numThreads, procBind);
}

void KrnlBuilder::parallelClause(
    Value parallelLoopIndex, Value numThreads, StringAttr procBind) const {
  // No need to check procBind as its value are derived from parallel(...).
  b().template create<KrnlParallelClauseOp>(
      loc(), parallelLoopIndex, numThreads, procBind);
}

void KrnlBuilder::iterate(ValueRange originalLoops, ValueRange optimizedLoops,
    ValueRange lbs, ValueRange ubs,
    function_ref<void(const KrnlBuilder &createKrnl, ValueRange indices)>
        bodyBuilderFn) const {
  auto bodyBuilderFnWrapper = [&](const KrnlBuilder &createKrnl,
                                  ValueRange indices, ValueRange iterArgs) {
    bodyBuilderFn(createKrnl, indices);
  };
  iterate(originalLoops, optimizedLoops, lbs, ubs, {}, bodyBuilderFnWrapper);
}

// Deprecated
KrnlIterateOp KrnlBuilder::iterate(ValueRange originalLoops,
    ValueRange optimizedLoops, ValueRange lbs, ValueRange ubs, ValueRange inits,
    KrnlLoopBody2Fn bodyBuilderFn) const {
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
    KrnlLoopBodyFn bodyBuilderFn) const {
  auto bodyBuilderFnWrapper = [&](const KrnlBuilder &createKrnl,
                                  ValueRange indices, ValueRange iterArgs) {
    bodyBuilderFn(createKrnl, indices);
  };
  iterateIE(originalLoops, optimizedLoops, lbs, ubs, {}, bodyBuilderFnWrapper);
}

// Deprecated.
KrnlIterateOp KrnlBuilder::iterateIE(ValueRange originalLoops,
    ValueRange optimizedLoops, ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
    ValueRange inits, KrnlLoopBody2Fn bodyBuilderFn) const {
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

void KrnlBuilder::forLoopIE(IndexExpr lb, IndexExpr ub, int64_t step,
    bool useParallel, KrnlLoopBodyFn builderFn) const {
  ValueRange originalLoopDef = defineLoops(1);
  llvm::SmallVector<Value, 1> optLoopDef(1, originalLoopDef[0]);
  if (step > 1) {
    // Block loop by step.
    ValueRange blockedLoopDef = block(originalLoopDef[0], step);
    optLoopDef[0] = blockedLoopDef[0];
  }
  if (useParallel)
    parallel(optLoopDef[0]);
  iterateIE(originalLoopDef, optLoopDef, {lb}, {ub}, builderFn);
}

void KrnlBuilder::forLoopsIE(ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
    ArrayRef<int64_t> steps, ArrayRef<bool> useParallel,
    KrnlLoopBodyFn builderFn) const {
  impl::forLoopsIE(*this, lbs, ubs, steps, useParallel, builderFn);
}

void KrnlBuilder::forExplicitParallelLoopIE(IndexExpr lb, IndexExpr ub,
    IndexExpr threadNum, KrnlLoopBodyFn builderFn) const {
  IndexExpr zero = LitIE(0);
  if (threadNum.isLiteralAndIdenticalTo(1)) {
    // Sequential case as we have only 1 thread (parallel disabled statically).
    llvm::SmallVector<Value, 4> params = {
        zero.getValue(), lb.getValue(), ub.getValue()};
    builderFn(*this, params);
    return;
  }
  // Compute blockSize: the number of elements of (lb...ub) per thread.
  IndexExpr trip = ub - lb; // Expected to be positive, aka ub>lb.
  IndexExpr blockSize = trip.ceilDiv(threadNum);
  // Explicit parallelism: iterate over all threads 0..threadNum in parallel.
  forLoopIE(zero, threadNum, /*step*/ 1, /*parallel*/ true,
      [&](const KrnlBuilder &ck, ValueRange loopInd) {
        IndexExprScope scope(ck);
        IndexExpr t = DimIE(loopInd[0]);
        IndexExpr tTimesBlockSize = t * SymIE(blockSize);
        IndexExpr currLB = SymIE(lb) + tTimesBlockSize;
        IndexExpr currUB = currLB + SymIE(blockSize);
        currUB = IndexExpr::min(currUB, SymIE(ub));
        // Passes the thread ID, its lower bound, and its upper bound.
        llvm::SmallVector<Value, 4> params = {
            t.getValue(), currLB.getValue(), currUB.getValue()};
        builderFn(ck, params);
      });
}

void KrnlBuilder::simdIterateIE(IndexExpr lb, IndexExpr ub, int64_t VL,
    bool fullySimd, bool useParallel, ArrayRef<Value> inputs,
    ArrayRef<DimsExpr> inputAFs, ArrayRef<Value> outputs,
    ArrayRef<DimsExpr> outputAFs,
    ArrayRef<KrnlSimdIterateBodyFn> iterateBodyFnList) const {
  onnx_mlir::impl::simdIterateIE<KrnlBuilder, KrnlBuilder>(*this, lb, ub, VL,
      fullySimd, useParallel, inputs, inputAFs, outputs, outputAFs,
      iterateBodyFnList);
}

void KrnlBuilder::simdReduceIE(IndexExpr lb, IndexExpr ub, int64_t VL,
    bool fullySimd, ArrayRef<Value> inputs, ArrayRef<DimsExpr> inputAFs,
    ArrayRef<Value> tmps, ArrayRef<DimsExpr> tmpAFs, ArrayRef<Value> outputs,
    ArrayRef<DimsExpr> outputAFs, ArrayRef<Value> initVals,
    /* reduction function (simd or scalar) */
    ArrayRef<KrnlSimdReductionBodyFn> reductionBodyFnList,
    /* post reduction function (simd to scalar + post processing)*/
    ArrayRef<KrnlSimdPostReductionBodyFn> postReductionBodyFnList) const {
  onnx_mlir::impl::simdReduceIE<KrnlBuilder, KrnlBuilder>(*this, lb, ub, VL,
      fullySimd, inputs, inputAFs, tmps, tmpAFs, outputs, outputAFs, initVals,
      reductionBodyFnList, postReductionBodyFnList);
}

void KrnlBuilder::simdReduce2DIE(IndexExpr lb, IndexExpr ub, int64_t VL,
    bool fullySimd, Value input, DimsExpr inputAF, Value tmp, DimsExpr tmpAF,
    Value output, DimsExpr outputAF, Value initVal,
    /* reduction functions (simd or scalar) */
    KrnlSimdReductionBodyFn reductionBodyFn,
    /* post reduction functions (post processing ONLY)*/
    KrnlSimdPostReductionBodyFn postReductionBodyFn) const {
  onnx_mlir::impl::simdReduce2DIE<KrnlBuilder, KrnlBuilder>(*this, lb, ub, VL,
      fullySimd, input, inputAF, tmp, tmpAF, output, outputAF, initVal,
      reductionBodyFn, postReductionBodyFn);
}

void KrnlBuilder::yield(ValueRange iterArgs) const {
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

//===----------------------------------------------------------------------===//
// Math style functions.

// Keep code gen here in sync with Elementwise.cpp GenOpMix
// getGenOpMix<ONNXRoundOp>
Value KrnlBuilder::roundEven(Value input) const {
  Type elementType = getElementTypeOrSelf(input.getType());
  MultiDialectBuilder<VectorBuilder, MathBuilder> create(*this);
  VectorType vecType = mlir::dyn_cast<VectorType>(input.getType());
  if (VectorMachineSupport::requireCustomASM(
          GenericOps::roundEvenGop, elementType)) {
    // Use Krnl round even op as LLVM does not support roundEven.
    if (!vecType)
      // Scalar.
      return b().create<KrnlRoundEvenOp>(loc(), input.getType(), input);

    // Vector, enable unrolling of multiple archVL.
    int64_t archVL = VectorMachineSupport::getArchVectorLength(
        GenericOps::roundEvenGop, elementType);
    assert(archVL > 1 && "expected vector with archVL>1");
    assert(vecType.getRank() == 1 && "1D vec only");
    int64_t vecSize = vecType.getShape()[0];
    assert(vecSize % archVL == 0 && "expected multiple of archVL");
    int64_t numArchVec = vecSize / archVL;
    VectorType vecType2D = VectorType::get({numArchVec, archVL}, elementType);
    // Cast input vector to a vector of chunks (archVL values that can be
    // handled by one hardware SIMD instruction).
    Value input2D = create.vec.shapeCast(vecType2D, input);
    Value output2D = input2D;
    // Iterates over all hardware SIMD chunks.
    for (int64_t i = 0; i < numArchVec; ++i) {
      // Extract one chunk, compute new value, insert result in corresponding
      // output 2D vector.
      Value subInput = create.vec.extractFrom2D(input2D, i);
      Value subOutput =
          b().create<KrnlRoundEvenOp>(loc(), subInput.getType(), subInput);
      output2D = create.vec.insertInto2D(subOutput, output2D, i);
    }
    // Recast output 2D vector into the flat vector (same shape as input).
    return create.vec.shapeCast(vecType, output2D);
  }
  // No need for custom support, use math roundEven. May want to evaluate
  // whether to use the mlir roundEven or our own emulation.
  // Note: MacOS CI has an issue with the roundEven instruction, thus continue
  // to use emulation. May change in the future.
  return create.math.roundEvenEmulation(input);
}

//===----------------------------------------------------------------------===//
// C library functions.

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
