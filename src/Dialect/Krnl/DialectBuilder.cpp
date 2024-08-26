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

Value KrnlBuilder::load(Value memref, ValueRange indices) const {
  if (indices.size() == 0) {
    // case memref<1xdtype>
    MemRefType type = dyn_cast_or_null<MemRefType>(memref.getType());
    assert(type && "Not MemRefType");
    if (type.getRank() == 1 && type.getShape()[0] == 1) {
      MultiDialectBuilder<MathBuilder> create(*this);
      Value iZero = create.math.constantIndex(0);
      return b().create<KrnlLoadOp>(loc(), memref, ValueRange({iZero}));
    }
  }
  return b().create<KrnlLoadOp>(loc(), memref, indices);
}

mlir::Value KrnlBuilder::load(mlir::Value memref, mlir::ValueRange indices,
    mlir::ValueRange offsets) const {
  SmallVector<Value, 4> computedIndices;
  MathBuilder createMath(*this);
  createMath.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  return load(memref, computedIndices);
}

Value KrnlBuilder::loadIE(Value memref, ArrayRef<IndexExpr> indices) const {
  if (indices.size() == 0) {
    // case memref<1xdtype>
    MemRefType type = dyn_cast_or_null<MemRefType>(memref.getType());
    assert(type && "Not MemRefType");
    if (type.getRank() == 1 && type.getShape()[0] == 1) {
      MultiDialectBuilder<MathBuilder> create(*this);
      Value iZero = create.math.constantIndex(0);
      return b().create<KrnlLoadOp>(loc(), memref, ValueRange({iZero}));
    }
  }
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  return b().create<KrnlLoadOp>(loc(), memref, indexValues);
}

void KrnlBuilder::store(Value val, Value memref, ValueRange indices) const {
  if (indices.size() == 0) {
    // case memref<1xdtype>
    MemRefType type = dyn_cast_or_null<MemRefType>(memref.getType());
    assert(type && "Not MemRefType");
    if (type.getRank() == 1 && type.getShape()[0] == 1) {
      MultiDialectBuilder<MathBuilder> create(*this);
      Value iZero = create.math.constantIndex(0);
      b().create<KrnlStoreOp>(loc(), val, memref, ValueRange({iZero}));
      return;
    }
  }
  b().create<KrnlStoreOp>(loc(), val, memref, indices);
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
  if (indices.size() == 0) {
    // case memref<1xdtype>
    MemRefType type = dyn_cast_or_null<MemRefType>(memref.getType());
    assert(type && "Not MemRefType");
    if (type.getRank() == 1 && type.getShape()[0] == 1) {
      MultiDialectBuilder<MathBuilder> create(*this);
      Value iZero = create.math.constantIndex(0);
      b().create<KrnlStoreOp>(loc(), val, memref, ValueRange({iZero}));
      return;
    }
  }
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  b().create<KrnlStoreOp>(loc(), val, memref, indexValues);
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
  b().create<KrnlPrefetchOp>(
      loc(), memref, indices, isWrite, localityHint, isDataCache);
}

void KrnlBuilder::prefetchIE(Value memref, ArrayRef<IndexExpr> indices,
    bool isWrite, unsigned localityHint, bool isDataCache) {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  b().create<KrnlPrefetchOp>(
      loc(), memref, indexValues, isWrite, localityHint, isDataCache);
}

void KrnlBuilder::seqstore(
    mlir::Value element, mlir::Value seq, mlir::Value index) const {
  b().create<KrnlSeqStoreOp>(loc(), element, seq, index);
}

void KrnlBuilder::seqstore(
    mlir::Value element, mlir::Value seq, IndexExpr index) const {
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

mlir::KrnlIterateOp KrnlBuilder::iterate(ValueRange originalLoops,
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
    mlir::ValueRange inits,
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

/*
Example of how to use the interface:
Say you have a loop of i=0..256, j=0..128 and want to exploit r[i,j] = a[i,j] +
b[j] + c. For the loops, we will need access functions for a, b, and r.

Say we already have the loop for the outer loop of i

krnl.iterate(loop i from 0 to 256) {
  ii is the loop index.

  // 1) compute access function for a, b, c
  // 2) launch simd loop with
  //     3) simd kernel
}

1) Access functions
   Assuming here that we are not blocking the j loop, namely the simd iteration
   goes over all j values, the access functions should be defined as follows.

   aAF = {ii, 0}
   bAF = {0}
   rAF = {ii, 0}

   If the j loop was blocked (say j=0 to 128 by 16), then instead of `0` in the
   last dim, we would have 'blocked_jj'

2) Launch simd loop

   create.krnl.simdIterateIE(
     lb=LitIE(0), ub=litIE(128), totVL=8, // loop params
     fullySimd=true, useParallel=false,   // loop options
     inputs={A, B}, inputAFs={aAF, bAF},  // inputs
     outputs={R}, outputAFs={rAF},        // outputs
     krnl)                                // lambda function for kernel

3) Krnl for SIMD loop

   The kernel functions has 4 inputs:
   a) krnl builder to further build code
   b) list of loaded input values, in the same order as in inputs
   c) list of results values, that must be enqueued by the kernel
   d) totVL used for the loop (VL for simd, 1 for scalar)

   The same kernel will be used in a SIMD context, in which the inputs and
   outputs must be vectors of VL elements, or in a scalar context, in which the
   inputs and outputs must be scalars.

   In our example, the kernel is as follows

   [&](KrnlBuilder &kb, ArrayRef<Value> inputVals,
       SmallVectorImpl<Value> &resVals, int64_t VL) {
      MultiDialectBuilder<KrnlBuilder, MathBuilder> create(kb);
      Value aVal = inputVals[0];            // simd or scalar
      Value bVal = inputVals[1];            // simd or scalar
      Value cVal = create.krnl.load(C, {}); // scalar always
      Value newVal = create.math.add(aVal, bVal); // simd or scalar
      newVal = create.math.add(newVal, cVal); // if newVal is simd, cVal is
                                              // splatted
      res.emplace_back(newVal); // Save simd or scalar result.
    }

    The krnl.simdIterateIE will be in charge of loading and saving the values in
    memory. The create.math functions have been extended so that when a SIMD
    value is computed with a scalar, that scalar will be automaticaly splatted
    (aka promoted to a vector of identical values). As a result, the kernel can
    be written in a SIMD agnostic value. However, in rare situations, we may
    want to know if we are in SIMD mode or not. VL will give the totVL used here
    (either totVL>1 or 1).
*/

// Determine if an access has one element from the innermost dimensions up to
// innerDim.
bool static hasOneElementInInnermostDims(Value value, int64_t innerDim) {
  // Get info.
  ShapedType type = mlir::dyn_cast<ShapedType>(value.getType());
  assert(type && "expected shaped type");
  int64_t rank = type.getRank();
  mlir::ArrayRef<int64_t> shape = type.getShape();
  for (int64_t i = std::max((int64_t)0, rank - innerDim); i < rank; ++i)
    if (shape[i] != 1)
      return false;
  return true;
}

void KrnlBuilder::simdIterateIE(IndexExpr lb, IndexExpr ub, int64_t VL,
    bool fullySimd, bool useParallel, ArrayRef<Value> inputs,
    ArrayRef<DimsExpr> inputAFs, ArrayRef<Value> outputs,
    ArrayRef<DimsExpr> outputAFs,
    function_ref<void(KrnlBuilder &kb, ArrayRef<Value> inputVals,
        llvm::SmallVectorImpl<Value> &resultVals, int64_t VL)>
        bodyBuilderFn) {
  int64_t inputNum = inputs.size();
  assert(inputAFs.size() == inputs.size() && "expected same size");
  int64_t outputNum = outputs.size();
  assert(outputAFs.size() == outputs.size() && "expected same size");
  MultiDialectBuilder<VectorBuilder> create(*this);

  if (VL > 1) {
    // Want SIMD, execute full SIMD loops blocked by VL.
    ValueRange loopDef = defineLoops(1);
    ValueRange blockedLoopDef = block(loopDef[0], VL);
    if (useParallel)
      parallel({blockedLoopDef[0]});

    // If we are not guaranteed that every iterations are SIMD iterations, then
    // we need to reduce the trip count by a bit so as to not over compute.
    // If we are not guaranteed that every iterations are SIMD iterations, then
    IndexExpr simdUb = ub;
    if (!fullySimd)
      simdUb = simdUb - (VL - 1);
    iterateIE(loopDef, {blockedLoopDef[0]}, {lb}, {simdUb},
        [&](KrnlBuilder &ck, ValueRange loopInd) {
          IndexExprScope scope(ck);
          MultiDialectBuilder<KrnlBuilder, VectorBuilder> create(ck);
          IndexExpr ind = DimIE(loopInd[0]);
          // Load all the inputs as vectors of VL values, with a few exceptions.
          // One is if the value is a "none value", leave as is. Another one is
          // if the innermost dim is a scalar (ie dim[rank-1] == 1), then we
          // just load the scalar.
          llvm::SmallVector<Value, 4> vecInputVals;
          for (int64_t i = 0; i < inputNum; ++i) {
            Value input = inputs[i];
            if (isNoneValue(input)) {
              // Simply enqueue the none value.
              vecInputVals.emplace_back(input);
              continue;
            }
            MemRefType type = mlir::cast<MemRefType>(input.getType());
            int64_t rank = type.getRank();
            DimsExpr AF = SymListIE(inputAFs[i]);
            assert(rank == (int64_t)AF.size() && "AF expected input rank refs");
            if (hasOneElementInInnermostDims(input, 1)) {
              // Has a reference with a scalar innermost dim, just load as a
              // scalar. No need to add the induction variable.
              Value scalarVal = create.krnl.loadIE(input, AF);
              vecInputVals.emplace_back(scalarVal);
            } else {
              // Have a vector.
              VectorType vecType = VectorType::get({VL}, type.getElementType());
              AF[rank - 1] = AF[rank - 1] + ind; // Add induction var.
              Value vecVal = create.vec.loadIE(vecType, input, AF, {});
              vecInputVals.emplace_back(vecVal);
            }
          }
          // Call the method to compute the values.
          llvm::SmallVector<Value, 4> vecResVals;
          bodyBuilderFn(create.krnl, vecInputVals, vecResVals, VL);
          assert((int64_t)vecResVals.size() == outputNum &&
                 "loop body with incorrect number of results");
          // Store all the outputs as vectors of VL values,
          for (int64_t i = 0; i < outputNum; ++i) {
            MemRefType type = mlir::cast<MemRefType>(outputs[i].getType());
            DimsExpr AF = SymListIE(outputAFs[i]);
            int64_t rank = type.getRank();
            assert(rank == (int64_t)AF.size() && "AF expected ouput rank refs");
            AF[rank - 1] = AF[rank - 1] + ind;
            create.vec.storeIE(vecResVals[i], outputs[i], AF, {});
          }
        });
    if (fullySimd)
      // Asserted that we only have SIMD iterations, we are done.
      return;
    // Account for the loop iterations performed above.
    IndexExpr tripCount = ub - lb;
    IndexExpr missingIters = tripCount % VL;
    IndexExpr completedIters = tripCount - missingIters;
    if (missingIters.isLiteralAndIdenticalTo(0)) {
      // Detect that we only have SIMD iterations, we are also done.
      return;
    }
    // We may have additional iterations to perform, adjust lb to skip the
    // completed iterations.
    lb = lb + completedIters;
  }
  // Handle remaining scalar values (from lb to ub without unrolling).
  ValueRange loopDef = defineLoops(1);
  iterateIE(
      loopDef, loopDef, {lb}, {ub}, [&](KrnlBuilder &ck, ValueRange loopInd) {
        IndexExprScope scope(ck);
        MultiDialectBuilder<KrnlBuilder> create(ck);
        IndexExpr ind = DimIE(loopInd[0]);
        // Load all the inputs as scalar values,
        llvm::SmallVector<Value, 4> scalarInputVals;
        for (int64_t i = 0; i < inputNum; ++i) {
          Value input = inputs[i];
          if (isNoneValue(input)) {
            // Simply enqueue the none value.
            scalarInputVals.emplace_back(input);
            continue;
          }
          MemRefType type = mlir::cast<MemRefType>(input.getType());
          int64_t rank = type.getRank();
          DimsExpr AF = SymListIE(inputAFs[i]);
          if (hasOneElementInInnermostDims(input, 1)) {
            // Has a reference with a scalar innermost dim, just load as a
            // scalar. No need to add the induction variable.
            Value scalarVal = create.krnl.loadIE(input, AF);
            scalarInputVals.emplace_back(scalarVal);
          } else {
            AF[rank - 1] = AF[rank - 1] + ind;
            Value scalarVal = create.krnl.loadIE(input, AF);
            scalarInputVals.emplace_back(scalarVal);
          }
        }
        // Call the method to compute the values.
        llvm::SmallVector<Value, 4> scalarResVals;
        bodyBuilderFn(create.krnl, scalarInputVals, scalarResVals, /*VL*/ 1);
        assert((int64_t)scalarResVals.size() == outputNum &&
               "loop body with incorrect number of results");
        // Store all the outputs as vectors of VL values,
        for (int64_t i = 0; i < outputNum; ++i) {
          MemRefType type = mlir::cast<MemRefType>(outputs[i].getType());
          DimsExpr AF = SymListIE(outputAFs[i]);
          int64_t rank = type.getRank();
          assert(rank == (int64_t)AF.size() && "AF expected ouput rank refs");
          AF[rank - 1] = AF[rank - 1] + ind;
          create.krnl.storeIE(scalarResVals[i], outputs[i], AF);
        }
      });
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
  b().create<KrnlMemcpyOp>(
      loc(), dest, src, numElems, /*dest_offset=*/zero, /*src_offset=*/zero);
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
ElementsAttr IndexExprBuilderForKrnl::getConst(mlir::Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto globalOp = dyn_cast_or_null<mlir::KrnlGlobalOp>(definingOp)) {
    if (globalOp.getValue().has_value())
      return mlir::dyn_cast<ElementsAttr>(globalOp.getValueAttr());
  } else if (auto globalOp =
                 dyn_cast_or_null<mlir::ONNXConstantOp>(definingOp)) {
    if (globalOp.getValue().has_value())
      return mlir::dyn_cast<ElementsAttr>(globalOp.getValueAttr());
  }
  return nullptr;
}

Value IndexExprBuilderForKrnl::getVal(Value intArrayVal, uint64_t i) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(*this);
  uint64_t rank = getShapedTypeRank(intArrayVal);
  if (rank == 0)
    return create.krnl.load(intArrayVal, {});
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
