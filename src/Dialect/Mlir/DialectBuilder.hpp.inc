//===---- DialectBuilder.hpp.inc - Helper functions for MLIR dialects -----===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains template helper functions for building MLIR operations.
//
// Note on usage of template keyword. Since the GenericAffineBuilder is
// templated, and we use templated functions (such as create<OP>), we must add
// the "template" keyword before the "create" function to indicate what is being
// templated.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DIALECT_BUILDER_MLIR_H
// This include is only here to include builder in the editors. Will be skipped
// when actually compiling.
#define ONNX_MLIR_DIALECT_BUILDER_MLIR_INC 1
#include "DialectBuilder.hpp"
#undef ONNX_MLIR_DIALECT_BUILDER_MLIR_INC
#endif

//===----------------------------------------------------------------------===//
// Templates for load / store
//===----------------------------------------------------------------------===//

namespace impl {

template <class BUILDER, class LOAD_OP>
mlir::Value load(const BUILDER &b, mlir::Value memref, mlir::ValueRange indices,
    mlir::ValueRange offsets) {
  // Handle offsets.
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(b);
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  // Perform load.
  if (computedIndices.size() == 0) {
    // case memref<1xdtype>
    auto type = mlir::cast<mlir::MemRefType>(memref.getType());
    if (type.getRank() == 1 && type.getShape()[0] == 1) {
      mlir::Value iZero = create.math.constantIndex(0);
      return b.getBuilder().template create<LOAD_OP>(
          b.getLoc(), memref, mlir::ValueRange({iZero}));
    }
  }
  return b.getBuilder().template create<LOAD_OP>(
      b.getLoc(), memref, computedIndices);
}

template <class BUILDER, class LOAD_OP>
mlir::Value loadIE(const BUILDER &b, mlir::Value memref,
    mlir::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) {
  llvm::SmallVector<mlir::Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  return load<BUILDER, LOAD_OP>(b, memref, indexValues, offsets);
}

template <class BUILDER, class STORE_OP>
void store(const BUILDER &b, mlir::Value val, mlir::Value memref,
    mlir::ValueRange indices, mlir::ValueRange offsets) {
  llvm::SmallVector<mlir::Value, 4> computedIndices;
  MultiDialectBuilder<MathBuilder> create(b);
  create.math.addOffsetToLeastSignificant(indices, offsets, computedIndices);
  if (computedIndices.size() == 0) {
    // case memref<1xdtype>
    auto type = mlir::cast<mlir::MemRefType>(memref.getType());
    if (type.getRank() == 1 && type.getShape()[0] == 1) {
      mlir::Value iZero = create.math.constantIndex(0);
      b.getBuilder().template create<STORE_OP>(
          b.getLoc(), val, memref, mlir::ValueRange({iZero}));
      return;
    }
  }
  b.getBuilder().template create<STORE_OP>(
      b.getLoc(), val, memref, computedIndices);
}

template <class BUILDER, class STORE_OP>
void storeIE(const BUILDER &b, mlir::Value val, mlir::Value memref,
    mlir::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) {
  llvm::SmallVector<mlir::Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  store<BUILDER, STORE_OP>(b, val, memref, indexValues, offsets);
}
} // namespace impl

//===----------------------------------------------------------------------===//
// Templates for GenericAffineBuilder
//===----------------------------------------------------------------------===//

template <class LOAD_OP, class STORE_OP>
mlir::Value GenericAffineBuilder<LOAD_OP, STORE_OP>::load(mlir::Value memref,
    mlir::ValueRange indices, mlir::ValueRange offsets) const {
  return onnx_mlir::impl::load<GenericAffineBuilder, LOAD_OP>(
      *this, memref, indices, offsets);
}

template <class LOAD_OP, class STORE_OP>
mlir::Value GenericAffineBuilder<LOAD_OP, STORE_OP>::loadIE(mlir::Value memref,
    mlir::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) const {
  return onnx_mlir::impl::loadIE<GenericAffineBuilder, LOAD_OP>(
      *this, memref, indices, offsets);
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::store(mlir::Value val,
    mlir::Value memref, mlir::ValueRange indices,
    mlir::ValueRange offsets) const {
  onnx_mlir::impl::store<GenericAffineBuilder, STORE_OP>(
      *this, val, memref, indices, offsets);
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::storeIE(mlir::Value val,
    mlir::Value memref, mlir::ArrayRef<IndexExpr> indices,
    mlir::ValueRange offsets) const {
  onnx_mlir::impl::storeIE<GenericAffineBuilder, STORE_OP>(
      *this, val, memref, indices, offsets);
}

template <class LOAD_OP, class STORE_OP>
inline mlir::Operation *GenericAffineBuilder<LOAD_OP, STORE_OP>::prefetch(
    mlir::Value memref, mlir::AffineMap map, mlir::ValueRange indices,
    bool isWrite, unsigned localityHint, bool isDataCache) {
  llvm::SmallVector<mlir::Value> indexArray(indices);
  return b().template create<mlir::affine::AffinePrefetchOp>(
      loc(), memref, map, indexArray, isWrite, localityHint, isDataCache);
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::forLoopIE(IndexExpr lb,
    IndexExpr ub, int64_t step,
    mlir::function_ref<void(GenericAffineBuilder &, mlir::ValueRange)>
        builderFn) const {
  // Transform IndexExpressions into value maps and list of operands.
  mlir::AffineMap lbMap, ubMap;
  llvm::SmallVector<mlir::Value, 8> lbOperands, ubOperands;
  lb.getAffineMapAndOperands(lbMap, lbOperands);
  ub.getAffineMapAndOperands(ubMap, ubOperands);
  // Create affine for.
  b().template create<mlir::affine::AffineForOp>(loc(), lbOperands, lbMap,
      ubOperands, ubMap, step, mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value index,
          mlir::ValueRange args) {
        GenericAffineBuilder createAffine(b, loc);
        builderFn(createAffine, {index});
        createAffine.yield();
      });
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::forLoopsIE(
    mlir::ArrayRef<IndexExpr> lbs, mlir::ArrayRef<IndexExpr> ubs,
    mlir::ArrayRef<int64_t> steps,
    mlir::function_ref<void(GenericAffineBuilder &, mlir::ValueRange)>
        builderFn) const {
  assert(lbs.size() == ubs.size() && "expected identical sizes");
  assert(lbs.size() == steps.size() && "expected identical sizes");
  llvm::SmallVector<mlir::Value> loopIndices;
  recursionForLoopsIE(lbs, ubs, steps, loopIndices, builderFn);
}

// This if then else construct has no arguments to the blocks.
template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::ifThenElseIE(
    IndexExprScope &scope, mlir::ArrayRef<IndexExpr> conditions,
    mlir::function_ref<void(GenericAffineBuilder &createAffine)> thenFn,
    mlir::function_ref<void(GenericAffineBuilder &createAffine)> elseFn) const {
  int64_t rank = conditions.size();
  llvm::SmallVector<mlir::AffineExpr, 4> affineCond;
  bool allTrue = true;
  bool allFalse = true;
  for (IndexExpr c : conditions) {
    assert(c.isAffine() && "conditions expected to be affine");
    affineCond.emplace_back(c.getAffineExpr());
    if (c.isLiteral()) {
      if (c.getLiteral() < 0) // Inequality is expr >= 0, test if false.
        allTrue = false;
      if (c.getLiteral() >= 0) // Inequality is expr >= 0, test if true.
        allFalse = false;
    } else {
      allTrue = allFalse = false;
    }
  }
  llvm::SmallVector<bool, 4> isEq(rank, false);
  auto inset = mlir::IntegerSet::get(
      scope.getNumDims(), scope.getNumSymbols(), affineCond, isEq);
  llvm::SmallVector<mlir::Value, 8> dimAndSymbolList;
  scope.getDimAndSymbolList(dimAndSymbolList);
  auto ifOp = b().template create<mlir::affine::AffineIfOp>(
      loc(), inset, dimAndSymbolList, true);
  mlir::Block *thenBlock = ifOp.getThenBlock();
  mlir::Block *elseBlock = ifOp.getElseBlock();
  if (!allFalse) {
    appendToBlock(thenBlock, [&](mlir::ValueRange args) {
      GenericAffineBuilder createAffine(b(), loc());
      thenFn(createAffine);
    });
  }
  if (!allTrue) {
    appendToBlock(elseBlock, [&](mlir::ValueRange args) {
      GenericAffineBuilder createAffine(b(), loc());
      elseFn(createAffine);
    });
  }
}

template <class LOAD_OP, class STORE_OP>
mlir::Value GenericAffineBuilder<LOAD_OP, STORE_OP>::apply(
    mlir::AffineMap map, mlir::ValueRange operands) const {
  return b().template create<mlir::affine::AffineApplyOp>(loc(), map, operands);
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::yield() const {
  b().template create<mlir::affine::AffineYieldOp>(loc());
}

// Support for multiple forLoopIE loops.
template <class LOAD_OP, class STORE_OP>
void GenericAffineBuilder<LOAD_OP, STORE_OP>::recursionForLoopsIE(
    mlir::ArrayRef<IndexExpr> lbs, mlir::ArrayRef<IndexExpr> ubs,
    mlir::ArrayRef<int64_t> steps,
    llvm::SmallVectorImpl<mlir::Value> &loopIndices,
    mlir::function_ref<void(GenericAffineBuilder &, mlir::ValueRange)>
        builderFn) const {
  int d = loopIndices.size();
  if (d < (int)lbs.size()) {
    // Issue a loop and recurse again.
    forLoopIE(lbs[d], ubs[d], steps[d],
        [&](GenericAffineBuilder &createAffine, mlir::ValueRange loopInd) {
          loopIndices.emplace_back(loopInd[0]);
          recursionForLoopsIE(lbs, ubs, steps, loopIndices, builderFn);
        });
  } else {
    // Call lambda function
    GenericAffineBuilder createAffine(b(), loc());
    builderFn(createAffine, loopIndices);
  }
}

// Support for adding blocks.
template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::appendToBlock(
    mlir::Block *block,
    mlir::function_ref<void(mlir::ValueRange)> builderFn) const {
  mlir::OpBuilder::InsertionGuard guard(b());
  if (block->empty() ||
      !block->back().mightHaveTrait<mlir::OpTrait::IsTerminator>()) {
    b().setInsertionPointToEnd(block);
  } else
    b().setInsertionPoint(&block->back());
  builderFn(block->getArguments());
}

//===----------------------------------------------------------------------===//
// Templates for SIMD code gen (instantiated for KRNL and SCF builders)
//===----------------------------------------------------------------------===//

// Forward declaration to keep template testing happy.
struct KrnlBuilder;

namespace impl { // Hide support for SIMD iterate/reduce in impl namespace.

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
     {krnl})                              // lambda function for kernel

3) Krnl for SIMD loop

   The kernel functions has 4 inputs:
   a) krnl builder to further build code
   b) list of loaded input values, in the same order as in inputs
   c) list of results values, that must be enqueued by the kernel
   d) totVL used for the loop (VL for simd, 1 for scalar)

   The same kernels will be used in a SIMD context, in which the inputs and
   outputs must be vectors of VL elements, or in a scalar context, in which the
   inputs and outputs must be scalars.

   In our example, the kernel is as follows

   [&](KrnlBuilder &kb, ArrayRef<Value> inputVals, int64_t VL) {
      MultiDialectBuilder<KrnlBuilder, MathBuilder> create(kb);
      Value aVal = inputVals[0];            // simd or scalar
      Value bVal = inputVals[1];            // simd or scalar
      Value cVal = create.krnl.load(C); // scalar always
      Value newVal = create.math.add(aVal, bVal); // simd or scalar
      newVal = create.math.add(newVal, cVal); // if newVal is simd, cVal is
                                              // splatted
      return newVal; // Save simd or scalar result.
    }

    The krnl.simdIterateIE will be in charge of loading and saving the values in
    memory. The create.math functions have been extended so that when a SIMD
    value is computed with a scalar, that scalar will be automaticaly splatted
    (aka promoted to a vector of identical values). As a result, the kernel can
    be written in a SIMD agnostic value. However, in rare situations, we may
    want to know if we are in SIMD mode or not. VL will give the totVL used here
    (either totVL>1 or 1).
*/

template <class BUILDER>
using SimdIterateBodyFn = std::function<mlir::Value(
    const BUILDER &b, mlir::ArrayRef<mlir::Value> inputVals, int64_t VL)>;

template <class BUILDER, class MEM_BUILDER>
void simdIterateIE(const BUILDER &builder, IndexExpr lb, IndexExpr ub,
    int64_t VL, bool fullySimd, bool useParallel,
    mlir::ArrayRef<mlir::Value> inputs, mlir::ArrayRef<DimsExpr> inputAFs,
    mlir::ArrayRef<mlir::Value> outputs, mlir::ArrayRef<DimsExpr> outputAFs,
    mlir::ArrayRef<SimdIterateBodyFn<BUILDER>> iterateBodyList) {
  int64_t inputNum = inputs.size();
  assert(inputAFs.size() == inputs.size() && "expected same size");
  int64_t outputNum = outputs.size();
  assert(outputAFs.size() == outputs.size() && "expected same size");
  int64_t fnNum = iterateBodyList.size();
  assert((int64_t)fnNum == outputNum && "expect 1 loop function per output");

  if (VL > 1) {
    // Want SIMD, execute full SIMD loops blocked by VL.

    // If we are not guaranteed that every iterations are SIMD iterations,
    // then we need to reduce the trip count by a bit so as to not over
    // compute. If we are not guaranteed that every iterations are SIMD
    // iterations, then
    IndexExpr simdUb = ub;
    if (!fullySimd)
      simdUb = simdUb - (VL - 1);

    // Define the loop block
    auto simdLoopBody = [&](BUILDER b, mlir::ValueRange loopInd) {
      IndexExprScope scope(b);
      VectorBuilder createVec(b);
      MEM_BUILDER createMem(b);
      IndexExpr ind = DimIE(loopInd[0]);
      llvm::SmallVector<mlir::Value, 4> vecInputVals;
      for (int64_t i = 0; i < inputNum; ++i) {
        mlir::Value input = inputs[i];
        if (MemRefBuilder::isNoneValue(input)) {
          // Simply enqueue the none value.
          vecInputVals.emplace_back(input);
          continue;
        }
        auto type = mlir::cast<mlir::MemRefType>(input.getType());
        int64_t rank = type.getRank();
        DimsExpr AF = SymListIE(inputAFs[i]);
        assert(rank == (int64_t)AF.size() && "AF expected input rank refs");
        if (MemRefBuilder::hasOneElementInInnermostDims(input, 1)) {
          // Has a reference with a scalar innermost dim, just load as a
          // scalar. No need to add the induction variable.
          vecInputVals.emplace_back(createMem.loadIE(input, AF));
        } else {
          // Have a vector.
          auto vecType = mlir::VectorType::get({VL}, type.getElementType());
          AF[rank - 1] = AF[rank - 1] + ind; // Add induction var.
          vecInputVals.emplace_back(createVec.loadIE(vecType, input, AF));
        }
      }
      // Call the method to compute the values.
      llvm::SmallVector<mlir::Value, 4> vecResVals;
      for (int64_t f = 0; f < outputNum; ++f) {
        vecResVals.emplace_back(iterateBodyList[f](b, vecInputVals, VL));
      }
      // Store all the outputs as vectors of VL values,
      for (int64_t i = 0; i < outputNum; ++i) {
        auto type = mlir::cast<mlir::MemRefType>(outputs[i].getType());
        DimsExpr AF = SymListIE(outputAFs[i]);
        int64_t rank = type.getRank();
        assert(rank == (int64_t)AF.size() && "AF expected ouput rank refs");
        AF[rank - 1] = AF[rank - 1] + ind;
        createVec.storeIE(vecResVals[i], outputs[i], AF);
      }
    };

    // Invocation of the  (possibly parallel) SIMD loop.
    if constexpr (std::is_same<BUILDER, KrnlBuilder>::value) {
      // Use KRNL interface
      mlir::ValueRange loopDef = builder.defineLoops(1);
      mlir::ValueRange blockedLoopDef = builder.block(loopDef[0], VL);
      if (useParallel)
        builder.parallel({blockedLoopDef[0]});
      builder.iterateIE(
          loopDef, {blockedLoopDef[0]}, {lb}, {simdUb}, simdLoopBody);
    } else if constexpr (std::is_same<BUILDER, SCFBuilder>::value) {
      if (useParallel) {
        IndexExpr litVL = LitIE(VL);
        builder.parallelLoops({lb.getValue()}, {simdUb.getValue()},
            {litVL.getValue()}, simdLoopBody);
      } else {
        builder.forLoop(lb.getValue(), simdUb.getValue(), VL, simdLoopBody);
      }
    } else {
      llvm_unreachable("BUILDER type not supported\n");
    }

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
  auto scalarLoopBody = [&](BUILDER b, mlir::ValueRange loopInd) {
    IndexExprScope scope(b);
    MEM_BUILDER createMem(b);

    IndexExpr ind = DimIE(loopInd[0]);
    // Load all the inputs as scalar values,
    llvm::SmallVector<mlir::Value, 4> scalarInputVals;
    for (int64_t i = 0; i < inputNum; ++i) {
      mlir::Value input = inputs[i];
      if (MemRefBuilder::isNoneValue(input)) {
        // Simply enqueue the none value.
        scalarInputVals.emplace_back(input);
        continue;
      }
      auto type = mlir::cast<mlir::MemRefType>(input.getType());
      int64_t rank = type.getRank();
      DimsExpr AF = SymListIE(inputAFs[i]);
      if (MemRefBuilder::hasOneElementInInnermostDims(input, 1)) {
        // Has a reference with a scalar innermost dim, just load as a
        // scalar. No need to add the induction variable.
        scalarInputVals.emplace_back(createMem.loadIE(input, AF));
      } else {
        AF[rank - 1] = AF[rank - 1] + ind;
        scalarInputVals.emplace_back(createMem.loadIE(input, AF));
      }
    }
    // Call the method to compute the values.
    llvm::SmallVector<mlir::Value, 4> scalarResVals;
    for (int64_t f = 0; f < outputNum; ++f) {
      scalarResVals.emplace_back(iterateBodyList[f](b, scalarInputVals, 1));
    }
    // Store all the outputs as vectors of VL values,
    for (int64_t i = 0; i < outputNum; ++i) {
      auto type = mlir::cast<mlir::MemRefType>(outputs[i].getType());
      DimsExpr AF = SymListIE(outputAFs[i]);
      int64_t rank = type.getRank();
      assert(rank == (int64_t)AF.size() && "AF expected ouput rank refs");
      AF[rank - 1] = AF[rank - 1] + ind;
      createMem.storeIE(scalarResVals[i], outputs[i], AF);
    }
  };

  // Invocation of the scalar loop.
  if constexpr (std::is_same<BUILDER, KrnlBuilder>::value) {
    // Use KRNL dialect.
    mlir::ValueRange loopDef = builder.defineLoops(1);
    builder.iterateIE(loopDef, loopDef, {lb}, {ub}, scalarLoopBody);
  } else if constexpr (std::is_same<BUILDER, SCFBuilder>::value) {
    builder.forLoop(lb.getValue(), ub.getValue(), 1, scalarLoopBody);
  } else {
    llvm_unreachable("BUILDER type not supported\n");
  }
}

/*
  Note that because reductions are always between 2 values, the reduction
  function takes 1 input and one temp value, where the temp contains the partial
  result. So if we have 2 reductions (aka 2 outputs), we also need 2 inputs and
  2 temp. A call to function reductionBodyFnList[k] (namely the kth entry in the
  list) will be instantiated with the kth input value and the kth temp, and its
  result is ultimately saved into the kth output.

  This was not the case for simdIterateIE, where all of the inputs are provided
  to each of the functions computing one output. Here we only pass a pair of
  input & temp value to each function.

  This is reflected in the Body types below.
*/

template <class BUILDER>
using SimdReductionBodyFn = std::function<mlir::Value(
    const BUILDER &b, mlir::Value inputVal, mlir::Value tmpVal, int64_t VL)>;

template <class BUILDER>
using SimdPostReductionBodyFn = std::function<mlir::Value(
    const BUILDER &b, mlir::Value tmpVal, int64_t VL)>;

template <class BUILDER, class MEM_BUILDER>
void simdReduceIE(const BUILDER &builder, IndexExpr lb, IndexExpr ub,
    int64_t VL, bool fullySimd, mlir::ArrayRef<mlir::Value> inputs,
    mlir::ArrayRef<DimsExpr> inputAFs, mlir::ArrayRef<mlir::Value> tmps,
    mlir::ArrayRef<DimsExpr> tmpAFs, mlir::ArrayRef<mlir::Value> outputs,
    mlir::ArrayRef<DimsExpr> outputAFs, mlir::ArrayRef<mlir::Value> initVals,
    /* reduction functions (simd or scalar) */
    mlir::ArrayRef<SimdReductionBodyFn<BUILDER>> reductionBodyFnList,
    /* post reduction functions (simd to scalar + post processing)*/
    mlir::ArrayRef<SimdPostReductionBodyFn<BUILDER>> postReductionBodyFnList) {

  MultiDialectBuilder<VectorBuilder> create(builder);
  MEM_BUILDER createMem(builder);

  int64_t inputSize = inputs.size();
  int64_t outputSize = outputs.size();
  int64_t reductionFnSize = reductionBodyFnList.size();
  int64_t postReductionFnSize = postReductionBodyFnList.size();
  assert((int64_t)inputAFs.size() == inputSize && "expect same input size");
  assert(tmps.size() == tmpAFs.size() && "expect same tmp size");
  assert((int64_t)outputAFs.size() == outputSize && "expect output same size");
  assert((int64_t)tmps.size() == outputSize && "expect 1 tmp per output");
  assert((int64_t)initVals.size() == outputSize && "expect 1 init per output");
  assert(reductionFnSize == outputSize && "expect 1 red fn per output");
  assert(postReductionFnSize == outputSize && "expect 1 red fn per output");
  // Gather element and vector types and perform the inits. Do it in SIMD mode
  // regardless.
  llvm::SmallVector<mlir::VectorType, 4> vectorTypes;
  for (int64_t o = 0; o < outputSize; ++o) {
    mlir::Value initVal = initVals[o];
    mlir::Type elementType = initVal.getType();
    auto vectorType = mlir::VectorType::get({VL}, elementType);
    vectorTypes.emplace_back(vectorType);
    mlir::Value initVec = create.vec.splat(vectorType, initVal);
    create.vec.storeIE(initVec, tmps[o], tmpAFs[o], {});
  }
  if (VL > 1) {
    // Logic: see simdIterateIE.
    IndexExpr simdUb = ub;
    if (!fullySimd)
      simdUb = simdUb - (VL - 1);

    auto simdLoopBody = [&](BUILDER &b, mlir::ValueRange loopInd) {
      IndexExprScope scope(b);
      MultiDialectBuilder<VectorBuilder> create(b);
      // Load inputs in SIMD mode, indexed by loopInd[0] in innermost dim.
      llvm::SmallVector<mlir::Value, 4> inputVals;
      for (int64_t i = 0; i < inputSize; ++i) {
        auto inputType = mlir::cast<mlir::MemRefType>(inputs[i].getType());
        auto vecType = mlir::VectorType::get({VL}, inputType.getElementType());
        inputVals.emplace_back(
            create.vec.loadIE(vecType, inputs[i], inputAFs[i], {loopInd[0]}));
      }
      // Load tmp value in SIMD mode  (no indexing, same value over & over).
      llvm::SmallVector<mlir::Value, 4> tmpVals;
      for (int64_t o = 0; o < outputSize; ++o) {
        tmpVals.emplace_back(
            create.vec.loadIE(vectorTypes[o], tmps[o], tmpAFs[o], {}));
      }
      // Call reduction, one per function each with their input and tmp value.
      llvm::SmallVector<mlir::Value, 4> resultVals;
      for (int64_t f = 0; f < outputSize; ++f) {
        resultVals.emplace_back(
            reductionBodyFnList[f](b, inputVals[f], tmpVals[f], VL));
      }
      // Save tmp values in SIMD mode.
      for (int64_t o = 0; o < outputSize; ++o) {
        create.vec.storeIE(resultVals[o], tmps[o], tmpAFs[o], {});
      }
    };

    // Want SIMD, execute full SIMD loops reductions blocked by VL.
    // Perform SIMD reduction: iterates over all SIMD vectors.

    if constexpr (std::is_same<BUILDER, KrnlBuilder>::value) {
      // Implementation with Krnl.
      mlir::ValueRange loopDef = builder.defineLoops(1);
      mlir::ValueRange blockedLoopDef = builder.block(loopDef[0], VL);
      builder.iterateIE(
          loopDef, {blockedLoopDef[0]}, {lb}, {simdUb}, simdLoopBody);
    } else if constexpr (std::is_same<BUILDER, SCFBuilder>::value) {
      // Implementation with SCF.
      builder.forLoop(lb.getValue(), simdUb.getValue(), VL, simdLoopBody);
    } else {
      llvm_unreachable("BUILDER type not supported");
    }

    if (fullySimd) {
      // No leftovers, no additional iterations to be done.
    } else {
      // Account for the loop iterations performed above.
      IndexExpr tripCount = ub - lb;
      IndexExpr missingIters = tripCount % VL;
      IndexExpr completedIters = tripCount - missingIters;
      if (missingIters.isLiteralAndIdenticalTo(0)) {
        // Detected that we have no missing iterations. Ee are done, namely
        // fullySimd is true.
        fullySimd = true;
      } else {
        // We may have additional iterations to perform, adjust lb to skip the
        // completed iterations.
        lb = lb + completedIters;
      }
    }
  } else {
    // VL was 1, set fullySimd to false so that we execute all iterations
    // sequentially.
    fullySimd = false;
  }
  if (!fullySimd) {
    // We have leftover iterations to be done in sequential mode.
    // Handle remaining scalar values (from lb to ub without unrolling).

    auto scalarLoopBody = [&](BUILDER &b, mlir::ValueRange loopInd) {
      IndexExprScope scope(b);
      MEM_BUILDER createMem(b);
      IndexExpr ind = DimIE(loopInd[0]);
      // We now perform sequential reduction in the tmps 1st element. Load
      // inputs in sequential mode indexed by loopInd[0] in innermost dim.
      llvm::SmallVector<mlir::Value, 4> inputVals;
      for (int64_t i = 0; i < inputSize; ++i) {
        inputVals.emplace_back(
            createMem.loadIE(inputs[i], inputAFs[i], {loopInd[0]}));
      }
      // Load tmps in scalar mode (no indexing, same value over & over).
      llvm::SmallVector<mlir::Value, 4> tmpVals;
      for (int64_t o = 0; o < outputSize; ++o) {
        tmpVals.emplace_back(createMem.loadIE(tmps[o], tmpAFs[o], {}));
      }
      // Call reduction.
      llvm::SmallVector<mlir::Value, 4> resultVals;
      for (int64_t f = 0; f < outputSize; ++f) {
        resultVals.emplace_back(
            reductionBodyFnList[f](b, inputVals[f], tmpVals[f], 1));
      }
      // Save tmp values in sequential mode.
      for (int64_t o = 0; o < outputSize; ++o) {
        createMem.storeIE(resultVals[o], tmps[o], tmpAFs[o], {});
      }
    };

    // Perform scalar loop.
    if constexpr (std::is_same<BUILDER, KrnlBuilder>::value) {
      // Implementation with Krnl.
      mlir::ValueRange loopDef = builder.defineLoops(1);
      builder.iterateIE(loopDef, loopDef, {lb}, {ub}, scalarLoopBody);
    } else if constexpr (std::is_same<BUILDER, SCFBuilder>::value) {
      // Implementation with SCF.
      builder.forLoop(lb.getValue(), ub.getValue(), 1, scalarLoopBody);
    } else {
      llvm_unreachable("BUILDER type not supported");
    }
  }

  // Now perform post processing. Load all tmps.
  llvm::SmallVector<mlir::Value, 4> tmpVals;
  for (int64_t o = 0; o < outputSize; ++o) {
    // Load tmp in vector mode.
    tmpVals.emplace_back(
        create.vec.loadIE(vectorTypes[o], tmps[o], tmpAFs[o], {}));
  }
  llvm::SmallVector<mlir::Value, 4> scalarOutputs;
  // Invoke the post processing operations, which takes each tmp vector and
  // reduces it to a scalar.
  for (int64_t f = 0; f < outputSize; ++f) {
    scalarOutputs.emplace_back(
        postReductionBodyFnList[f](builder, tmpVals[f], 1));
  }
  // Store the scalar reductions.
  for (int64_t o = 0; o < outputSize; ++o) {
    createMem.storeIE(scalarOutputs[o], outputs[o], outputAFs[o]);
  }
}

} // namespace impl