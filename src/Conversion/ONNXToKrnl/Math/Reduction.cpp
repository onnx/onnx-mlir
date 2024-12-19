/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Reduction.cpp - Lowering Reduction Ops ----------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reduction Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include <functional>

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/SmallVectorHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"
#define DEBUG_FORCE_SHUFFLE_REDUCTION 0 /* should be 0 in repo */
#define REDUCTION_MULTIPLE_OF_VL_ONLY 0 /* 0: improved;1: old, for debug */

using namespace mlir;

namespace onnx_mlir {

enum RLegacy { Latest, UpTo13 };

//===----------------------------------------------------------------------===//
// Defaults

// Defines the VectorBuilder's CombiningKind associated with a given Op.
template <typename OP>
VectorBuilder::CombiningKind getCombiningKind() {
  llvm_unreachable("illegal combination kind");
}

// Defines if the OP requires a divide by mean; false by default.
template <typename OP>
bool divideByMean() {
  return false;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceProdOp
//===----------------------------------------------------------------------===//

template <>
Value getIdentityValue<ONNXReduceProdOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 1);
}
template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceProdOp>() {
  return VectorBuilder::CombiningKind::MUL;
}
template <>
GenOpMix getGenOpMix<ONNXReduceProdOp>(Type t, Operation *op) {
  return {{GenericOps::MulGop, 1}};
}

template <>
struct ScalarOp<ONNXReduceProdOp> {
  using FOp = arith::MulFOp;
  using IOp = arith::MulIOp;
};

template <>
Value getIdentityValue<ONNXReduceProdV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return getIdentityValue<ONNXReduceProdOp>(rewriter, loc, type);
}
template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceProdV13Op>() {
  return getCombiningKind<ONNXReduceProdOp>();
}
template <>
GenOpMix getGenOpMix<ONNXReduceProdV13Op>(Type t, Operation *op) {
  return getGenOpMix<ONNXReduceProdOp>(t, op);
}
template <>
struct ScalarOp<ONNXReduceProdV13Op> {
  using FOp = arith::MulFOp;
  using IOp = arith::MulIOp;
};

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceSumOp
//===----------------------------------------------------------------------===//

template <>
Value getIdentityValue<ONNXReduceSumOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 0);
}
template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceSumOp>() {
  return VectorBuilder::CombiningKind::ADD;
}
template <>
GenOpMix getGenOpMix<ONNXReduceSumOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}};
}
template <>
struct ScalarOp<ONNXReduceSumOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

template <>
Value getIdentityValue<ONNXReduceSumV11Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return getIdentityValue<ONNXReduceSumOp>(rewriter, loc, type);
}
template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceSumV11Op>() {
  return getCombiningKind<ONNXReduceSumOp>();
}
template <>
GenOpMix getGenOpMix<ONNXReduceSumV11Op>(Type t, Operation *op) {
  return getGenOpMix<ONNXReduceSumOp>(t, op);
}
template <>
struct ScalarOp<ONNXReduceSumV11Op> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMeanOp
//===----------------------------------------------------------------------===//

template <>
Value getIdentityValue<ONNXReduceMeanOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 0);
}
template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceMeanOp>() {
  return VectorBuilder::CombiningKind::ADD;
}
template <>
GenOpMix getGenOpMix<ONNXReduceMeanOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}};
}
template <>
bool divideByMean<ONNXReduceMeanOp>() {
  return true;
}
template <>
struct ScalarOp<ONNXReduceMeanOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

template <>
Value getIdentityValue<ONNXReduceMeanV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return getIdentityValue<ONNXReduceMeanOp>(rewriter, loc, type);
}
template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceMeanV13Op>() {
  return getCombiningKind<ONNXReduceMeanOp>();
}
template <>
GenOpMix getGenOpMix<ONNXReduceMeanV13Op>(Type t, Operation *op) {
  return getGenOpMix<ONNXReduceMeanOp>(t, op);
}
template <>
bool divideByMean<ONNXReduceMeanV13Op>() {
  return divideByMean<ONNXReduceMeanOp>();
}
template <>
struct ScalarOp<ONNXReduceMeanV13Op> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMaxOp
//===----------------------------------------------------------------------===//

template <>
Value getIdentityValue<ONNXReduceMaxOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.negativeInf(type);
}
template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceMaxOp>() {
  return VectorBuilder::CombiningKind::MAX;
}
template <>
GenOpMix getGenOpMix<ONNXReduceMaxOp>(Type t, Operation *op) {
  return {{GenericOps::MinMaxGop, 1}};
}
template <>
Value emitScalarOpFor<ONNXReduceMaxOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  MathBuilder createMath(rewriter, loc);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  return createMath.max(lhs, rhs);
}

template <>
Value getIdentityValue<ONNXReduceMaxV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return getIdentityValue<ONNXReduceMaxOp>(rewriter, loc, type);
}
template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceMaxV13Op>() {
  return getCombiningKind<ONNXReduceMaxOp>();
}
template <>
GenOpMix getGenOpMix<ONNXReduceMaxV13Op>(Type t, Operation *op) {
  return getGenOpMix<ONNXReduceMaxOp>(t, op);
}
template <>
Value emitScalarOpFor<ONNXReduceMaxV13Op>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  return emitScalarOpFor<ONNXReduceMaxOp>(
      rewriter, loc, op, elementType, scalarOperands);
}
//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMinOp
//===----------------------------------------------------------------------===//

template <>
Value getIdentityValue<ONNXReduceMinOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.positiveInf(type);
}
template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceMinOp>() {
  return VectorBuilder::CombiningKind::MIN;
}
template <>
GenOpMix getGenOpMix<ONNXReduceMinOp>(Type t, Operation *op) {
  return {{GenericOps::MinMaxGop, 1}};
}
template <>
Value emitScalarOpFor<ONNXReduceMinOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  MathBuilder createMath(rewriter, loc);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  return createMath.min(lhs, rhs);
}

template <>
Value getIdentityValue<ONNXReduceMinV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return getIdentityValue<ONNXReduceMinOp>(rewriter, loc, type);
}
template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceMinV13Op>() {
  return getCombiningKind<ONNXReduceMinOp>();
}
template <>
GenOpMix getGenOpMix<ONNXReduceMinV13Op>(Type t, Operation *op) {
  return getGenOpMix<ONNXReduceMinOp>(t, op);
}
template <>
Value emitScalarOpFor<ONNXReduceMinV13Op>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  return emitScalarOpFor<ONNXReduceMinOp>(
      rewriter, loc, op, elementType, scalarOperands);
}

//===----------------------------------------------------------------------===//

using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
    MathBuilder, MemRefBuilder, VectorBuilder, AffineBuilderKrnlMem, SCFBuilder,
    AffineBuilder>;
using PreProcessFn = std::function<mlir::Value(mlir::Value)>;

//===----------------------------------------------------------------------===//
// Helper function to perform reduction when an entire tensor is reduced to a
// single value. Support the reduction for up to 2 operations at once. If only
// one is needed, then pass ONNXNoneOp in the second slot.
// If PreProcessFn is given, each input value is preprocessed by the function
// before doing reduction.
// Return true if we can optimize the reduction, false otherwise.

template <typename BUILDER, typename ONNXReductionOp1,
    typename ONNXReductionOp2>
void emitOneStepOfFullSIMDReduction(ConversionPatternRewriter &rewriter,
    Operation *op, MDBuilder &create, Type elementType, IndexExpr lb,
    IndexExpr ub, int64_t VL, bool simdOnly, IndexExpr t, int64_t tNum,
    bool hasTwoRed, Value input1, Value input2, Value tmp1, Value tmp2,
    PreProcessFn preProc1, PreProcessFn preProc2, Value output1, Value output2,
    Value divisorForMean) {

  VectorType vecType = VectorType::get({VL}, elementType);

  SmallVector<Value, 2> inputs, tmps, outputs, initVals;
  SmallVector<DimsExpr, 2> inputAFs, tmpAFs, outputAFs;
  IndexExpr zero = LitIE(0);

  // Init data for 1st reduction
  inputs.emplace_back(input1);
  DimsExpr inputAF(1, zero); // Inputs starts at 0
  inputAFs.emplace_back(inputAF);
  tmps.emplace_back(tmp1);
  DimsExpr tmpAF(1, t * VL); // Each thread t starts a new section of VL values.
  tmpAFs.emplace_back(tmpAF);
  outputs.emplace_back(output1);
  DimsExpr outputAF; // By default, no value (scalar).
  if (tNum > 1) {
    outputAF.emplace_back(t); // If parallel, indexed by t.
  }
  outputAFs.emplace_back(outputAF);
  initVals.emplace_back(getIdentityValue<ONNXReductionOp1>(
      rewriter, create.getLoc(), elementType));
  // Init data for 2nd reduction.
  if (hasTwoRed) {
    inputs.emplace_back(input2);
    inputAFs.emplace_back(inputAF);
    tmps.emplace_back(tmp2);
    tmpAFs.emplace_back(tmpAF);
    outputs.emplace_back(output2);
    outputAFs.emplace_back(outputAF);
    initVals.emplace_back(getIdentityValue<ONNXReductionOp2>(
        rewriter, create.getLoc(), elementType));
  }

  // Create the reduction functions.
  llvm::SmallVector<impl::SimdReductionBodyFn<BUILDER>, 2> redBodyFnList;
  llvm::SmallVector<impl::SimdPostReductionBodyFn<BUILDER>, 2>
      postRedBodyFnList;
  // Push functions for the first reduction.
  redBodyFnList.emplace_back(
      [&](const BUILDER &b, Value inputVal, Value tmpVal, int64_t VL) {
        Type currType = (VL > 1) ? vecType : elementType;
        // Perform reduction of tmp and input.
        if (preProc1)
          inputVal = preProc1(inputVal);
        return emitScalarOpFor<ONNXReductionOp1>(
            rewriter, create.getLoc(), op, currType, {tmpVal, inputVal});
      });
  postRedBodyFnList.emplace_back(
      [&](const BUILDER &b, Value tmpVal, int64_t VL) {
        // Perform horizontal reductions.
        Value scalarVal =
            create.vec.reduction(getCombiningKind<ONNXReductionOp1>(), tmpVal);
        if (tNum == 1) { /* parallel: do it for the final iteration only */
          if (divideByMean<ONNXReductionOp1>())
            scalarVal = create.math.div(scalarVal, divisorForMean);
        }
        return scalarVal;
      });
  if (hasTwoRed) {
    // Push functions for the second reduction.
    redBodyFnList.emplace_back(
        [&](const BUILDER &b, Value inputVal, Value tmpVal, int64_t VL) {
          Type currType = (VL > 1) ? vecType : elementType;
          // Perform reduction of tmp and input.
          if (preProc2)
            inputVal = preProc2(inputVal);
          return emitScalarOpFor<ONNXReductionOp2>(
              rewriter, create.getLoc(), op, currType, {tmpVal, inputVal});
        });
    postRedBodyFnList.emplace_back(
        [&](const BUILDER &b, Value tmpVal, int64_t VL) {
          // Perform horizontal reductions.
          Value scalarVal = create.vec.reduction(
              getCombiningKind<ONNXReductionOp2>(), tmpVal);
          if (tNum == 1) { /* parallel: do it for the final iteration only */
            if (divideByMean<ONNXReductionOp2>())
              scalarVal = create.math.div(scalarVal, divisorForMean);
          }
          return scalarVal;
        });
  }
  // Call simd reduce.
  BUILDER builder(create.vec);
  builder.simdReduceIE(lb, ub, VL, simdOnly, inputs, inputAFs, tmps, tmpAFs,
      outputs, outputAFs, initVals, redBodyFnList, postRedBodyFnList);
}

template <typename ONNXReductionOp1, typename ONNXReductionOp2>
bool emitFullSIMDReductionFor(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value input, PreProcessFn preProc1, PreProcessFn preProc2,
    Value &alloc1, Value &alloc2, bool enableParallel) {
  // Create scope.
  IndexExprScope scope(&rewriter, loc);
  MDBuilder create(rewriter, loc);
  // Get info.
  MemRefType inputType = mlir::cast<MemRefType>(input.getType());
  Type elementType = inputType.getElementType();
  int64_t inputRank = inputType.getRank();
  DimsExpr inputDims, flatInputDims;
  create.krnlIE.getShapeAsSymbols(input, inputDims);
  // Flatten entirely the input memref.
  Value flatInput = create.mem.reshapeToFlatInnermost(
      input, inputDims, flatInputDims, inputRank);
  IndexExpr zero = LitIE(0);
  IndexExpr lb = zero;
  IndexExpr ub = flatInputDims[0];
  // Compute the divisor that is the number of elements participated in
  // reduction, i.e., 'divisor = size of input / size of output, where
  // output size == 1'.
  Value divisorForMean = create.math.cast(elementType, ub.getValue());

  // Has one or 2 reductions?
  bool hasTwoRed = true;
  if constexpr (std::is_same<ONNXReductionOp2, ONNXNoneOp>::value)
    hasTwoRed = false;

  // Study SIMD. Assume here that since SIMD is determined by the input type
  // (which is expected to be the same as the output scalar value), both
  // reduction will have the same archVL.
  GenOpMix mix = getGenOpMix<ONNXReductionOp1>(elementType, op);
  if (hasTwoRed) {
    GenOpMix mix2 = getGenOpMix<ONNXReductionOp2>(elementType, op);
    mix = computeGenOpMixUnion(mix, mix2);
  }
  int64_t collapsedInnermostLoops = inputRank;
  int64_t simdLoopStaticTripCount;
  bool simdOnly, canOverCompute = false;
  int64_t totVL =
      computeSuitableUnrollFactor(inputType, collapsedInnermostLoops, mix,
          canOverCompute, simdLoopStaticTripCount, simdOnly);
  // Test if loop trip count is long enough for a parallel execution.
  if (enableParallel) {
    int64_t parId;
    if (findSuitableParallelDimension({lb}, {ub}, 0, 1, parId, 32 * totVL)) {
      onnxToKrnlParallelReport(
          op, true, parId, lb, ub, "simd reduction to one element");
    } else {
      enableParallel = false;
      onnxToKrnlParallelReport(op, false, -1, -1,
          "not enough work in simd reduction to one element");
    }
  }
  if (!enableParallel) {
    // Allocate temp and output memory
    Value tmp1, tmp2;
    MemRefType redType = MemRefType::get({totVL}, elementType);
    MemRefType outputType = MemRefType::get({}, elementType);
    tmp1 = create.mem.alignedAlloc(redType);
    /*output*/ alloc1 = create.mem.alloc(outputType);

    alloc2 = nullptr;
    if (hasTwoRed) {
      tmp2 = create.mem.alignedAlloc(redType);
      /*output*/ alloc2 = create.mem.alloc(outputType);
    }
    int64_t tNum = 1; // No parallelism.
    IndexExpr t = zero;
    // OK to use Krnl builder here as we have a simple loop structure.
    emitOneStepOfFullSIMDReduction<KrnlBuilder, ONNXReductionOp1,
        ONNXReductionOp2>(rewriter, op, create, elementType, lb, ub, totVL,
        simdOnly, t, tNum, hasTwoRed, flatInput, flatInput, tmp1, tmp2,
        preProc1, preProc2, alloc1, alloc2, divisorForMean);
  } else {
    // Performs 2 rounds: first round compute a parallel partial reduction
    // where each (possibly virtual) thread is responsible for one chunk.
    // Second round computes the final reduction done by one thread.

    // TODO: this should not be hardwired but gotten from an option.
    int64_t tNum = 8;

    // Round 1.
    MemRefType redType = MemRefType::get({tNum * totVL}, elementType);
    MemRefType outputType = MemRefType::get({tNum}, elementType);
    Value tmp1, tmp2, output1, output2;

    tmp1 = create.mem.alignedAlloc(redType);
    output1 = create.mem.alloc(outputType);
    if (hasTwoRed) {
      tmp2 = create.mem.alignedAlloc(redType);
      output2 = create.mem.alloc(outputType);
    }

    IndexExpr tNumIE = LitIE(tNum);
    bool simdOnly = false; // Refine, but since we are chunking input, safer.
    create.krnl.forExplicitParallelLoopIE(
        lb, ub, tNumIE, [&](const KrnlBuilder &ck, mlir::ValueRange loopInd) {
          IndexExprScope scope(ck);
          MDBuilder create(ck);
          IndexExpr t = DimIE(loopInd[0]);
          IndexExpr currLB = SymIE(loopInd[1]);
          IndexExpr currUB = SymIE(loopInd[2]);
          // Use SCF builder because the partition of outer loop into block
          // makes the formulas non-affine.
          emitOneStepOfFullSIMDReduction<SCFBuilder, ONNXReductionOp1,
              ONNXReductionOp2>(rewriter, op, create, elementType, currLB,
              currUB, totVL, simdOnly, t, tNum, hasTwoRed, flatInput, flatInput,
              tmp1, tmp2, preProc1, preProc2, output1, output2, nullptr);
          // Result here, each iteration would have generate 1 value in
          // output1 &2,
        });
    // Now we need to reduce output's tNum values into one. Reuse tmps.
    MemRefType finalOutputType = MemRefType::get({}, elementType);
    /*output*/ alloc1 = create.mem.alloc(finalOutputType);
    alloc2 = nullptr;
    if (hasTwoRed)
      /*output*/ alloc2 = create.mem.alloc(finalOutputType);
    IndexExpr finalLB = zero;
    IndexExpr finalUB = tNumIE;
    IndexExpr t = zero;
    // Reduction here is straight forward, Krnl builder is fine.
    emitOneStepOfFullSIMDReduction<KrnlBuilder, ONNXReductionOp1,
        ONNXReductionOp2>(rewriter, op, create, elementType, finalLB, finalUB,
        /*VL*/ 1, /*simd only*/ false, t, /*thread num */ 1, hasTwoRed, output1,
        output2, tmp1, tmp2, preProc1, preProc2, alloc1, alloc2,
        divisorForMean);
  }

  if (hasTwoRed)
    onnxToKrnlSimdReport(op, /*successful*/ true, totVL,
        simdLoopStaticTripCount, "fused reduction to a scalar");
  else
    onnxToKrnlSimdReport(op, /*successful*/ true, totVL,
        simdLoopStaticTripCount, "reduction to a scalar");

  return true;
}

void emitMinMaxReductionToScalar(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Value input, Value &minAlloc, Value &maxAlloc,
    bool enableSIMD, bool enableParallel) {
  // Try optimized path first.
  if (enableSIMD &&
      emitFullSIMDReductionFor<ONNXReduceMinOp, ONNXReduceMaxOp>(rewriter, loc,
          op, input, nullptr, nullptr, minAlloc, maxAlloc, enableParallel)) {
    return;
  }
  // Could not optimize the pattern, generate default path.
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  Type elementType = mlir::cast<MemRefType>(input.getType()).getElementType();
  MemRefType outputType = MemRefType::get({}, elementType);
  Value none = create.onnx.none();
  // Generate reductions.
  minAlloc = create.onnx.toMemref(
      create.onnx.reduceMin(outputType, input, none, false));
  maxAlloc = create.onnx.toMemref(
      create.onnx.reduceMax(outputType, input, none, false));
}

void emitSymmetricQuantRecscaleToScalar(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Value input, uint64_t bitWidth,
    Value &recscale, bool enableSIMD, bool enableParallel) {
  Type elemType = getElementType(input.getType());
  assert(elemType.isF32() && "Only support f32");
  double range = static_cast<double>((1 << (bitWidth - 1)) - 1);

  // Try optimized path first.
  Value absmaxMemRef, noused;
  MultiDialectBuilder<OnnxBuilder, KrnlBuilder, MathBuilder> create(
      rewriter, loc);
  if (enableSIMD &&
      emitFullSIMDReductionFor<ONNXReduceMaxOp, ONNXNoneOp>(
          rewriter, loc, op, input, [&](Value v) { return create.math.abs(v); },
          nullptr, absmaxMemRef, noused, enableParallel)) {
    Value cst = create.math.constant(elemType, range);
    Value absmax = create.krnl.load(absmaxMemRef);
    recscale = create.math.div(cst, absmax);
    return;
  }

  // Could not optimize the pattern, generate default path.
  Value none = create.onnx.none();
  RankedTensorType scalarTy = RankedTensorType::get({}, elemType);
  Value cst = create.onnx.constant(
      DenseElementsAttr::get(scalarTy, static_cast<float>(range)));
  Value recscaleMemRef = create.onnx.toMemref(
      create.onnx.div(cst, create.onnx.reduceMax(scalarTy,
                               create.onnx.abs(input), none, false, false)));
  recscale = create.krnl.load(recscaleMemRef);
}
//===----------------------------------------------------------------------===//
// Generic reduction code (for current and legacy using "if constexpr".
// Function use SIMD if all reductions occur consecutively in the innermost
// loops.

template <typename ONNXReductionOp, RLegacy legacyOp>
struct ONNXReductionOpLowering : public OpConversionPattern<ONNXReductionOp> {
  using OpAdaptor = typename ONNXReductionOp::Adaptor;
  bool enableSIMD = false;
  bool enableParallel = false;

  ONNXReductionOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableSIMD, bool enableParallel)
      : OpConversionPattern<ONNXReductionOp>(typeConverter, ctx),
        enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXReductionOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ONNXReductionOp reduceOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    /*
     * Condition: reduction function must be associative and commutative.
     *
     * Example 1 (here, reduction function is `+`):
     * Induction variables: (i0, i1, i2)
     * axes = [0, 2]
     * keepdims = true
     * krnl.iterate() with (i0, i1, i2) {
     *   Y(0, i1, 0) += X(i0, i1, i2)
     * }
     *
     * Example 2 (here, reduction function is `+`):
     * Induction variables: (i0, i1, i2)
     * axes = [0, 2]
     * keepdims = false
     * krnl.iterate() with (i0, i1, i2) {
     *   Y(i1) += X(i0, i1, i2)
     * }
     *
     */

    Operation *op = reduceOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    Location loc = ONNXLoc<ONNXReductionOp>(op);
    Value input = operands[0];

    //////////////////////////////////////////////////////////////////////
    // Handle type conversion.
    MemRefType memRefInType = mlir::cast<MemRefType>(input.getType());
    Type convertedOutType =
        this->typeConverter->convertType(*op->result_type_begin());
    assert(convertedOutType && mlir::isa<MemRefType>(convertedOutType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefOutType = mlir::cast<MemRefType>(convertedOutType);
    int64_t inRank = memRefInType.getRank();
    int64_t outRank = memRefOutType.getRank();
    auto memRefOutShape = memRefOutType.getShape();
    auto elementOutType = memRefOutType.getElementType();

    //////////////////////////////////////////////////////////////////////
    // Read op attributes
    // Check KeepDims attribute
    int64_t keepdims = adaptor.getKeepdims();
    bool isKeepdims = (keepdims == 1);
    bool isNoop = false;
    if constexpr (legacyOp == RLegacy::Latest) {
      // Guarded by constexpr as legacy ops don't have this attribute, which
      // defaults to false in legacy ops.
      auto noop = llvm::dyn_cast<ONNXReductionOp>(op).getNoopWithEmptyAxes();
      isNoop = (noop == 1);
    }

    //////////////////////////////////////////////////////////////////////
    // Extract raw axes from operation.
    IndexExprScope scope(&rewriter, loc);
    MDBuilder create(rewriter, loc);

    DimsExpr rawAxesIE;
    Value axesVal = nullptr;
    bool hasNoAxes = false; // Axles is a None (i.e. optional value not given).
    bool dynamicAxes = false; // No dynamic axes unless proven otherwise.
    IndexExpr axisShape0;     // Shape to be used if dynamic.
    if constexpr (legacyOp == RLegacy::Latest) {
      // Deal with operations where we find axes in the inputs.
      axesVal = adaptor.getAxes();
      if (isNoneValue(axesVal)) {
        // Default value of having no axes.
        hasNoAxes = true;
      } else {
        // Check it has a rank of 0 or 1.
        int64_t axisRank = create.krnlIE.getShapedTypeRank(axesVal);
        assert((axisRank == 0 || axisRank == 1) && "expect rank 0 or 1");
        if (axisRank == 0)
          axisShape0 = LitIE(1);
        else
          axisShape0 = create.krnlIE.getShapeAsDim(axesVal, 0);

        if (!axisShape0.isLiteral())
          // Don't even know the shape of the axis... it is dynamic.
          dynamicAxes = true;
        else
          // We have a shape, try to get the array integers
          create.krnlIE.getIntFromArrayAsDims(axesVal, rawAxesIE);
      }
    } else if constexpr (legacyOp == RLegacy::UpTo13) {
      // Deal with operations where we find axes in the attributes.
      ArrayAttr axesAttrs = adaptor.getAxesAttr();
      if (!axesAttrs) {
        // Default value of having no axes.
        hasNoAxes = true;
      } else {
        // Has axes defined by the axis, get their index expressions
        create.krnlIE.getIntFromArrayAsLiterals(axesAttrs, rawAxesIE);
      }
    }

    //////////////////////////////////////////////////////////////////////
    // Reduction over all dimensions to a scalar value.
    bool fullReduction =
        hasNoAxes || (rawAxesIE.size() == static_cast<uint64_t>(inRank));
    if (fullReduction && !isKeepdims && enableSIMD) {
      Value alloc, none;
      if (emitFullSIMDReductionFor<ONNXReductionOp, ONNXNoneOp>(rewriter, loc,
              op, input, nullptr, nullptr, alloc, none, enableParallel)) {
        rewriter.replaceOp(op, alloc);
        return success();
      }
    }

    //////////////////////////////////////////////////////////////////////
    // Characterize literal axes: make unique and within [0, inRank).
    std::vector<int64_t> uniqueLitAxes;
    llvm::BitVector litAxes(inRank, false);
    if (hasNoAxes) {
      if (isNoop) {
        // No axes and is noop, should we not just return the input array?
      } else {
        // No axes, perform a full reduction.
        for (int64_t i = 0; i < inRank; ++i) {
          uniqueLitAxes.push_back(i);
          litAxes[i] = true;
        }
      }
    } else if (!dynamicAxes) {
      // Check raw axes.
      int64_t rawAxesRank = rawAxesIE.size();
      for (int64_t i = 0; i < rawAxesRank; ++i) {
        if (!rawAxesIE[i].isLiteral()) {
          dynamicAxes = true; // Unknown axes is being reduced.
          break;
        }
        // Has a literal, normalize it.
        int64_t axis = rawAxesIE[i].getLiteral();
        if (axis < -inRank || axis > inRank - 1) {
          return emitError(loc, "axes value out of range");
        }
        int64_t newAxis = axis >= 0 ? axis : (inRank + axis);
        // Record it if new.
        if (!litAxes[newAxis]) {
          uniqueLitAxes.push_back(newAxis);
          litAxes[newAxis] = true;
        }
      }
    }

    //////////////////////////////////////////////////////////////////////
    // Process axes.
    // With static axes, use this
    std::map<int64_t, int64_t> outInDimMap;
    // Info for SIMD (requires static).
    bool horizontalSimd = false;
    bool hasHorizontalSimdSupport = false;
    bool parallelSimd = false;
    int64_t innermostLoopCollapse = 0;
    int64_t totVL = 1;
    bool simdOnly = false;
    int64_t simdLoopStaticTripCount = 0;

    // With dynamic axes, use this
    Value maskVal = nullptr;
    Value falseVal = nullptr;
    Value trueVal = nullptr;
    Value valueOne = nullptr;
    if (!dynamicAxes) {
      // All axes are static, fill in the outInDimMap appropriately.
      outInDimMap =
          getReductionMapping(memRefInType, uniqueLitAxes, isKeepdims);
      // Analyze possibility of using SIMD execution.
      if (enableSIMD) {
        LLVM_DEBUG(llvm::dbgs() << "  SIMD: study if possible\n");
        // Look for horizontal reduction: innermost loops with reduction only.
        int64_t hNum = 0;
        for (int64_t i = inRank - 1; i >= 0; --i) {
          if (!litAxes[i])
            break; // Found first innermost dim without a reduction.
          hNum++;
        }
        // Currently for horizontal, requires 1) all of the reductions are
        // together in the innermost dims, and 2) not all dims are reduced.
        horizontalSimd =
            hNum > 0 && hNum == (int64_t)uniqueLitAxes.size() && hNum < inRank;
        LLVM_DEBUG(if (hNum > 0 && !horizontalSimd) llvm::dbgs()
                   << "  SIMD: unsupported horizontal simd mode\n");
        // Look for parallel reduction: innermost loops without reduction.
        int64_t pNum = 0;
        for (int64_t i = inRank - 1; i >= 0; --i) {
          if (litAxes[i])
            break; // Found first innermost dim with a reduction.
          pNum++;
        }
        parallelSimd = (pNum > 0);
        innermostLoopCollapse = hNum + pNum; // Only one nonzero.
        if (horizontalSimd || parallelSimd) {
          assert(!(horizontalSimd && parallelSimd) &&
                 "expected at most horizontal or parallel SIMD");
          DimsExpr inputDims;
          create.krnlIE.getShapeAsSymbols(input, inputDims);
          if (horizontalSimd) {
#if !DEBUG_FORCE_SHUFFLE_REDUCTION
            VectorBuilder::CombiningKind kind =
                getCombiningKind<ONNXReductionOp>();
            hasHorizontalSimdSupport =
                supportedHorizontalSIMDOp(kind, elementOutType);
#endif
          }
          // Currently only vectorize loops whose SIMD dimension is a multiple
          // of the natural SIMD width. Aka, we don't deal with SIMD of
          // partial vectors.
          GenOpMix mix = getGenOpMix<ONNXReductionOp>(elementOutType, op);
          bool canOverCompute = false;
          totVL =
              computeSuitableUnrollFactor(memRefInType, innermostLoopCollapse,
                  mix, canOverCompute, simdLoopStaticTripCount, simdOnly);
          if (!hasHorizontalSimdSupport) {
            // When we don't have horizontal SIMD support, we use a code gen
            // scheme that relies on unrolling. So we don't want any unrollVL
            // here. Some benchmarks have small trip counts (e.g. GPT2: 8).
            totVL = capVLForMaxUnroll(memRefInType, totVL, 1);
          }
#if REDUCTION_MULTIPLE_OF_VL_ONLY
          // Currently fails with krnl to affine without this. Should
          // consider an affine simd iterate/reduce. onnx-mlir
          // -shapeInformation=0:4x8 reducemean2.mlir -O3 --march=arm64
          if (!simdOnly) {
            totVL =
                capVLForSimdOnly(memRefInType, totVL, simdLoopStaticTripCount);
          }
#endif
          LLVM_DEBUG(llvm::dbgs() << "  SIMD: " << innermostLoopCollapse
                                  << " loops, totVL " << totVL << "\n");
          if (totVL <= 1) {
            horizontalSimd = parallelSimd = false;
            LLVM_DEBUG(llvm::dbgs() << "  SIMD: no good totVL\n");
          }
        }
      }
    } else {
      // Has one or more dynamic axes.
      if (!isKeepdims)
        return emitError(
            loc, "dynamic axes without getKeepdims() not implemented");

      // Define a mask memref with same size of input and bool type
      // maskVal[i] == true if ith dim will be reduced
      auto maskType =
          RankedTensorType::get({inRank}, rewriter.getIntegerType(1));
      // Convert the mask type to MemRefType.
      Type convertedMaskType = this->typeConverter->convertType(maskType);
      assert(convertedMaskType && mlir::isa<MemRefType>(convertedMaskType) &&
             "Failed to convert type to MemRefType");
      MemRefType maskTypeInMemRefType =
          mlir::cast<MemRefType>(convertedMaskType);
      maskVal = create.mem.alignedAlloc(maskTypeInMemRefType);
      falseVal = create.math.constant(rewriter.getIntegerType(1), 0);
      trueVal = create.math.constant(rewriter.getIntegerType(1), 1);
      valueOne = create.math.constantIndex(1);
      // Initialize mask to 0
      // Unless noop_with_empty_axesDim is false and axesDim is
      // ShapedType::kDynamic.
      Value initVal;
      if (!axisShape0.isLiteral() && !isNoop) {
        // initVal = axes.shape[0] == 0 ? true : false
        IndexExprScope axesLoopContext(&rewriter, loc);
        Value zeroIndex = create.math.constantIndex(0);
        Value cond = create.math.eq(axisShape0.getValue(), zeroIndex);
        initVal = create.math.select(cond, trueVal, falseVal);
      } else {
        // When axesDim is known, it can not be 0 due to !isNoneValue
        initVal = falseVal;
      }
      for (auto i = 0; i < inRank; i++) {
        Value indexVal = create.math.constantIndex(i);
        create.krnl.store(initVal, maskVal, indexVal);
      }

      // Consider the case when axes[i] is negative
      // maskVal[axes[i] < 0 ? axes[i]+inRank: axes[i]] = 1
      auto axesElementType =
          mlir::cast<MemRefType>(axesVal.getType()).getElementType();
      auto dataDimConst = create.math.constant(axesElementType, inRank);
      Value zeroValue = create.math.constant(axesElementType, 0);
      if (!axisShape0.isLiteral()) {
        // When axes is dynamic, generate a Krnl loop
        KrnlBuilder createKrnl(rewriter, loc);
        createKrnl.forLoopIE(LitIE(0), axisShape0, /*step*/ 1, /*par*/ false,
            [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
              Value axe = createKrnl.load(axesVal, loopInd[0]);
              Value cond = create.math.slt(axe, zeroValue);
              Value dim = create.math.select(
                  cond, create.math.add(axe, dataDimConst), axe);
              Value jVal = rewriter.create<arith::IndexCastOp>(
                  loc, rewriter.getIndexType(), dim);
              createKrnl.store(trueVal, maskVal, jVal);
            });
      } else {
        int64_t axesDim = axisShape0.getLiteral();
        for (int64_t i = 0; i < axesDim; ++i) {
          Value indexVal = create.math.constantIndex(i);
          Value axe = create.krnl.load(axesVal, indexVal);
          // Check negative
          Value cond = create.math.slt(axe, zeroValue);
          Value dim =
              create.math.select(cond, create.math.add(axe, dataDimConst), axe);
          create.math.select(cond, create.math.add(axe, dataDimConst), axe);
          Value jVal = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIndexType(), dim);
          create.krnl.store(trueVal, maskVal, jVal);
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "  SIMD " << (totVL > 1 ? "" : "im")
                            << "possible with totVL " << totVL << "\n");

    //////////////////////////////////////////////////////////////////////
    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    if (hasAllConstantDimensions(memRefOutType)) {
      alloc = create.mem.alignedAlloc(memRefOutType);
    } else {
      SmallVector<Value, 2> allocOperands;
      for (decltype(outRank) i = 0; i < outRank; ++i) {
        if (memRefOutShape[i] == ShapedType::kDynamic) {
          if (dynamicAxes) {
            // We appear to rely here on the fact that input and output rank
            // is identical. Dim size: maskVal[i] ? 1 : inputDim[i]
            Value inputDim = create.mem.dim(input, i);
            Value indexVal = create.math.constantIndex(i);
            Value mask = create.krnl.load(maskVal, indexVal);
            Value cond = create.math.eq(mask, trueVal);
            Value dim = create.math.select(cond, valueOne, inputDim);
            allocOperands.push_back(dim);
          } else {
            Value dim = create.mem.dim(input, outInDimMap[i]);
            allocOperands.push_back(dim);
          }
        }
      }
      alloc = create.mem.alignedAlloc(memRefOutType, allocOperands);
    }

    // Used if compute mean
    Value divisorForMean = nullptr;
    if (divideByMean<ONNXReductionOp>()) {
      // Compute the divisor that is the number of elements participated in
      // reduction, i.e., 'divisor = size of input / size of output'.
      IndexExprScope scope(create.krnl);
      IndexExpr inputSizeExpr = LitIE(1);
      for (unsigned i = 0; i < inRank; i++) {
        IndexExpr dimExpr = create.krnlIE.getShapeAsSymbol(input, i);
        inputSizeExpr = inputSizeExpr * dimExpr;
      }
      IndexExpr outputSizeExpr = LitIE(1);
      for (unsigned i = 0; i < outRank; i++) {
        IndexExpr dimExpr = create.krnlIE.getShapeAsSymbol(alloc, i);
        outputSizeExpr = outputSizeExpr * dimExpr;
      }
      IndexExpr divisorExpr = inputSizeExpr.floorDiv(outputSizeExpr);
      divisorForMean = create.math.cast(elementOutType, divisorExpr.getValue());
    }

    if (horizontalSimd) {
      if (hasHorizontalSimdSupport) {
        genHorizontalSimdReduction(rewriter, create, op, elementOutType, input,
            alloc, inRank, outRank, totVL, simdOnly, innermostLoopCollapse,
            isKeepdims, divisorForMean, enableParallel);
        onnxToKrnlSimdReport(op, /*successful*/ true, totVL,
            simdLoopStaticTripCount, "horizontal");
      } else {
        genShuffleHorizontalSimdReduction(rewriter, create, op, elementOutType,
            input, alloc, inRank, outRank, totVL, simdOnly,
            innermostLoopCollapse, isKeepdims, divisorForMean, enableParallel);
        onnxToKrnlSimdReport(op, /*successful*/ true, totVL,
            simdLoopStaticTripCount, "shuffle-horizontal");
      }
    } else {
      genScalarReduction(rewriter, create, op, elementOutType, input, alloc,
          inRank, outRank, dynamicAxes, maskVal, outInDimMap, divisorForMean,
          enableParallel);
      std::string msg;
      if (parallelSimd)
        msg = "no simd because no supported for parallel scheme";
      else
        msg = "unsupported";
      onnxToKrnlSimdReport(
          op, /*successful*/ false, /*vl*/ 0, simdLoopStaticTripCount, msg);
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }

  void genScalarReduction(ConversionPatternRewriter &rewriter,
      MDBuilder &create, Operation *op, Type elementType, Value input,
      Value alloc, int64_t inRank, int64_t outRank, bool dynamicAxes,
      Value maskVal, std::map<int64_t, int64_t> &outInDimMap,
      Value divisorForMean, bool enableParallel) const {
    LLVM_DEBUG(llvm::dbgs() << "gen scalar reduction\n");
    //////////////////////////////////////////////////////////////////////
    // There are two required and one optional Krnl loops:
    // - One to initialize the result memref,
    // - One to do reduction, and
    // - One to compute mean (optional).

    // Parallelism only if output is not a scalar.
    if (outRank == 0)
      enableParallel = false;

    // 1. Define loops to initialize the result.
    Value identity = getIdentityValue<ONNXReductionOp>(
        rewriter, create.getLoc(), elementType);
    create.krnl.memset(alloc, identity);

    ValueRange loop2Def = create.krnl.defineLoops(inRank);
    SmallVector<IndexExpr, 4> lbs2(inRank, LitIE(0));
    SmallVector<IndexExpr, 4> ubs2;
    create.krnlIE.getShapeAsSymbols(input, ubs2);
    Value trueVal = create.math.constant(rewriter.getIntegerType(1), 1);
    // TODO Temporary disable the 2nd loop parallelism, since its outermost
    // loop could be a reduction loop, where parallelism would not be safe.
    create.krnl.iterateIE(loop2Def, loop2Def, lbs2, ubs2,
        [&](const KrnlBuilder &kb, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(kb);
          Value zeroIndex = create.math.constantIndex(0);
          // Compute accumulator  access function.
          SmallVector<Value, 4> accumulatorAccessFct;
          for (decltype(inRank) i = 0; i < outRank; ++i) {
            if (dynamicAxes) {
              // For the reduced dim, the output index is always 0.
              Value indexVal = create.math.constantIndex(i);
              Value mask = create.krnl.load(maskVal, indexVal);
              Value cond = create.math.eq(mask, trueVal);
              Value dim = create.math.select(cond, zeroIndex, loopInd[i]);
              accumulatorAccessFct.push_back(dim);
            } else if (outInDimMap.find(i) != outInDimMap.end())
              accumulatorAccessFct.push_back(loopInd[outInDimMap[i]]);
            else
              accumulatorAccessFct.push_back(zeroIndex);
          }
          // Load accumulator value, accumulate, and store.
          Value next = create.krnl.load(input, loopInd);
          Value accumulated = create.krnl.load(alloc, accumulatorAccessFct);
          accumulated = emitScalarOpFor<ONNXReductionOp>(
              rewriter, create.getLoc(), op, elementType, {accumulated, next});
          create.krnl.store(accumulated, alloc, accumulatorAccessFct);
        });

    // 3. Define an Krnl loop to compute mean (optional).
    if (divideByMean<ONNXReductionOp>()) {
      // Compute mean
      ValueRange loop3Def = create.krnl.defineLoops(outRank);
      SmallVector<IndexExpr, 4> lbs3(outRank, LitIE(0));
      SmallVector<IndexExpr, 4> ubs3;
      create.krnlIE.getShapeAsSymbols(alloc, ubs3);
      if (enableParallel) {
        int64_t parId;
        if (findSuitableParallelDimension(lbs3, ubs3, 0, 1, parId,
                /*min iter for going parallel*/ 4)) {
          create.krnl.parallel(loop3Def[0]);
          onnxToKrnlParallelReport(
              op, true, 0, lbs3[0], ubs3[0], "reduction scalar mean");
        } else {
          onnxToKrnlParallelReport(op, false, 0, lbs3[0], ubs3[0],
              "not enough work in reduction scalar mean");
        }
      }
      create.krnl.iterateIE(loop3Def, loop3Def, lbs3, ubs3,
          [&](const KrnlBuilder &kb, ValueRange loopInd) {
            MultiDialectBuilder<KrnlBuilder, MathBuilder> create(kb);
            Value loadData = create.krnl.load(alloc, loopInd);
            Value meanVal = create.math.div(loadData, divisorForMean);
            create.krnl.store(meanVal, alloc, loopInd);
          });
    }
  }

  bool supportedHorizontalSIMDOp(
      VectorBuilder::CombiningKind getCombiningKind, Type elementType) const {
    int64_t len;
    switch (getCombiningKind) {
    case VectorBuilder::CombiningKind::ADD:
      len = VectorMachineSupport::getArchVectorLength(
          GenericOps::SumAcrossGop, elementType);
      break;
    case VectorBuilder::CombiningKind::MIN:
    case VectorBuilder::CombiningKind::MAX:
      len = VectorMachineSupport::getArchVectorLength(
          GenericOps::SumAcrossGop, elementType);
      break;
    default:
      len = 1;
    }
    return len != 1;
  }

  // Generate a single reduction, eventually using a horizontal reduction
  // (which, if the hardware supports it, will be one instruction; otherwise
  // it will be simulated by several operations).
  //
  // flatInput has been flattened from [N][M][R1][R2] to [N][M][R1*R2], where
  // the SIMD reduction is done along the last dim. By definition of what we
  // support here, R1*R2 mod VL = 0, namely the reduction dimension is a
  // multiple of VL (no partial SIMD).
  //
  // tmpAlloc has been flattened (if keepDim is true) to [N][M].
  //
  // outLoopInd defines which [n][m] is to be used to load the inputs to be
  // reduced (flatInput[n][m][*]) and where the reduction is to be saved
  // (flatAlloc[n][m]).

  void genOneHorizontalSimdReduction(ConversionPatternRewriter &rewriter,
      MDBuilder &create, Operation *op, Type elementType, VectorType vecType,
      Value tmpAlloc, Value flatInput, Value flatAlloc, Value initVec,
      Value divisorForMean, ValueRange outLoopInd, Value simdUB, int64_t VL,
      bool simdOnly) const {
    IndexExpr lb = LitIE(0);
    IndexExpr ub = SymIE(simdUB);
    SmallVector<IndexExpr, 4> outputAF = SymListIE(outLoopInd);
    SmallVector<IndexExpr, 4> inputAF = outputAF;
    inputAF.emplace_back(lb);
    SmallVector<IndexExpr, 4> tmpAF(2, lb); // tmpAlloc is 2D
    Value identity = getIdentityValue<ONNXReductionOp>(
        rewriter, create.getLoc(), elementType);
    create.krnl.simdReduceIE(lb, ub, VL, simdOnly,
        /* inputs*/ {flatInput}, {inputAF},
        /* temp */ {tmpAlloc}, {tmpAF},
        /* output */ {flatAlloc}, {outputAF},
        /* init */ {identity},
        /* reduction simd/scalar */
        {[&](const KrnlBuilder &kb, Value inputVal, Value tmpVal, int64_t VL) {
          Type type = VL > 1 ? vecType : elementType;
          return emitScalarOpFor<ONNXReductionOp>(
              rewriter, create.getLoc(), op, type, {tmpVal, inputVal});
        }},
        /* post processing */
        {[&](const KrnlBuilder &kb, Value tmpVal, int64_t VL) {
          // Horizontal reduction.
          Value accumulatedVal =
              create.vec.reduction(getCombiningKind<ONNXReductionOp>(), tmpVal);
          // Other post reduction operation...
          if (divideByMean<ONNXReductionOp>()) {
            accumulatedVal = create.math.div(accumulatedVal, divisorForMean);
          }
          return accumulatedVal;
        }});
  }

  // We assume here that the hardware has an efficient SIMD horizontal
  // operation, so we simply generate one horizontal SIMD reduction for each
  // reductions that needs to be performed.
  void genHorizontalSimdReduction(ConversionPatternRewriter &rewriter,
      MDBuilder &create, Operation *op, Type elementType, Value input,
      Value alloc, int64_t inRank, int64_t outRank, int64_t VL, bool simdOnly,
      int64_t collapsedInnermostLoops, bool isKeepDims, Value divisorForMean,
      bool enableParallel) const {
    LLVM_DEBUG(llvm::dbgs() << "gen horizontal simd reduction\n");
    assert(VL > 1 && "expected simd here");
    VectorType vecType = VectorType::get({VL}, elementType);
    // Flatten the input: in[N][M][Red1][Red2] -> in[N][M][Red1*Red2]
    DimsExpr inDims, flatInDims;
    create.krnlIE.getShapeAsSymbols(input, inDims);
    Value flatInput = create.mem.reshapeToFlatInnermost(
        input, inDims, flatInDims, collapsedInnermostLoops);
    int64_t flatInRank = flatInDims.size();
    Value simdUB = flatInDims[flatInRank - 1].getValue();
    // Flatten the output: only the non-reduced dims of in: -> [N][M]
    DimsExpr outDims, flatOutDims;
    create.krnlIE.getShapeAsSymbols(alloc, outDims);
    int64_t collapseOutInnermostLoop =
        isKeepDims ? collapsedInnermostLoops + 1 : 1;
    Value flatAlloc = create.mem.reshapeToFlatInnermost(
        alloc, outDims, flatOutDims, collapseOutInnermostLoop);
    int64_t flatOutRank = flatOutDims.size();
    // Flat output should have all but the flattened SIMD loop, so there
    // should only be a 1 rank difference between the two.
    assert(flatOutRank == flatInRank - 1 && "wrong assumptions about dims");

    // Parallelism only if output is not a scalar.
    if (flatOutRank == 0)
      enableParallel = false;

    // Compute type of alloca a small temp vector.
    MemRefType tmpType = MemRefType::get({1, VL}, elementType);
    // Define loops for input dimensions, blocking the inner dim by VL
    ValueRange outLoopDef = create.krnl.defineLoops(flatOutRank);
    SmallVector<IndexExpr, 4> lbs(flatOutRank, LitIE(0));
    if (enableParallel) {
      int64_t parId;
      if (findSuitableParallelDimension(lbs, flatOutDims, 0, 1, parId,
              /*min iter for going parallel*/ 128)) {
        create.krnl.parallel(outLoopDef[0]);
        onnxToKrnlParallelReport(
            op, true, 0, lbs[0], flatOutDims[0], "reduction h-simd");
      } else {
        enableParallel = false;
        onnxToKrnlParallelReport(op, false, 0, lbs[0], flatOutDims[0],
            "not enough work for reduction h-simd");
      }
    }
    create.krnl.iterateIE(outLoopDef, outLoopDef, lbs, flatOutDims,
        [&](const KrnlBuilder &ck, ValueRange outLoopInd) {
          MDBuilder create(ck);
          // When parallel, will stay inside; otherwise will migrate out.
          Value tmpAlloc = create.mem.alignedAlloc(tmpType);
          Value identity = getIdentityValue<ONNXReductionOp>(
              rewriter, create.getLoc(), elementType);
          Value initVec = create.vec.splat(vecType, identity);
          genOneHorizontalSimdReduction(rewriter, create, op, elementType,
              vecType, tmpAlloc, flatInput, flatAlloc, initVec, divisorForMean,
              outLoopInd, simdUB, VL, simdOnly);
        });
  }

  // We perform here VL Simd Reductions at once. We are guaranteed that there
  // are VL reductions to be performed. The algorithm works in 2 steps.
  //
  // In the first step, we perform the SIMD reductions of VL distinct
  // reductions using the "emitScalarOp" associated with that operation. At
  // the end of this step, we have VL distinct partial reductions, where each
  // of the VL vector register have a partial reduction in each of their own
  // VL SIMD slots.
  //
  // In the second step, we reduce each VL vectors of VL partial values into
  // one vector of VL fully-reduced values. We use shuffle patterns to
  // generate efficient code where each of the temporary vectors always
  // contain VL values. This is implemented by the create.vec.multiReduction
  // operation.
  //
  // Finally, the VL full reductions are stored as a vector operation in the
  // flatAlloc[m][n+0...+VL-1] output.

  void genVlHorizontalSimdReduction(ConversionPatternRewriter &rewriter,
      MDBuilder &create, Operation *op, Type elementType, VectorType vecType,
      Value tmpBlockedAlloc, Value flatInput, Value flatAlloc, Value initVec,
      Value divisorForMean, ValueRange blockedOutLoopInd,
      IndexExpr blockedCurrIndex, Value simdUB, int64_t VL,
      bool simdOnly) const {
    IndexExpr zero = LitIE(0);
    IndexExpr lb = zero;
    IndexExpr ub = SymIE(simdUB);
    int64_t rank = blockedOutLoopInd.size();
    DimsExpr inputAF = SymListIE(blockedOutLoopInd);
    inputAF[rank - 1] = blockedCurrIndex;
    inputAF.emplace_back(zero);
    DimsExpr tmpAF = {zero, zero};
    DimsExpr outputAF = SymListIE(blockedOutLoopInd);
    Value identity = getIdentityValue<ONNXReductionOp>(
        rewriter, create.getLoc(), elementType);
    if (simdOnly) {
      create.affine.simdReduce2DIE(
          lb, ub, VL, simdOnly, flatInput, inputAF, tmpBlockedAlloc, tmpAF,
          flatAlloc, outputAF, identity,
          [&](const AffineBuilder &b, Value inputVal, Value tmpVal,
              int64_t VL) {
            Type type = VL > 1 ? vecType : elementType;
            return emitScalarOpFor<ONNXReductionOp>(
                rewriter, b.getLoc(), op, type, {tmpVal, inputVal});
          },
          [&](const AffineBuilder &b, Value tmpVal, int VL) {
            if (divideByMean<ONNXReductionOp>())
              return create.math.div(tmpVal, divisorForMean);
            return tmpVal;
          });
    } else {
      create.scf.simdReduce2DIE( // Affine fails with dynamic shapes.
          lb, ub, VL, simdOnly, flatInput, inputAF, tmpBlockedAlloc, tmpAF,
          flatAlloc, outputAF, identity,
          [&](const SCFBuilder &b, Value inputVal, Value tmpVal, int64_t VL) {
            Type type = VL > 1 ? vecType : elementType;
            return emitScalarOpFor<ONNXReductionOp>(
                rewriter, b.getLoc(), op, type, {tmpVal, inputVal});
          },
          [&](const SCFBuilder &b, Value tmpVal, int VL) {
            if (divideByMean<ONNXReductionOp>())
              return create.math.div(tmpVal, divisorForMean);
            return tmpVal;
          });
    }
  }

  // Solution when there is no horizontal SIMD op support and that shuffle ops
  // are needed. Assuming a (flattened) output reduction tensor of [N][M],
  // this algorithm will block the inter dimension of the output tensor by VL.
  // For each block of VL values to be reduced, we use the efficient functions
  // that computes them using shuffles (genVlHorizontalSimdReduction). For the
  // last block (if any) that has fewer than VL remaining reductions to be
  // performed, we simply perform r<VL sequential reductions (which will use a
  // "simulated" horizontal operation to generate the final reduction, in
  // genOneHorizontalSimdReduction).

  void genShuffleHorizontalSimdReduction(ConversionPatternRewriter &rewriter,
      MDBuilder &create, Operation *op, Type elementType, Value input,
      Value alloc, int64_t inRank, int64_t outRank, int64_t VL, bool simdOnly,
      int64_t collapsedInnermostLoops, bool isKeepDims, Value divisorForMean,
      bool enableParallel) const {

    LLVM_DEBUG(llvm::dbgs() << "gen shuffle horizontal simd reduction\n");
    assert(VL > 1 && "expected simd here");
    VectorType vecType = VectorType::get({VL}, elementType);
    // Flatten the input: in[N][M][Red1][Red2] -> in[N][M][Red1*Red2]
    DimsExpr inDims, flatInDims;
    create.krnlIE.getShapeAsSymbols(input, inDims);
    Value flatInput = create.mem.reshapeToFlatInnermost(
        input, inDims, flatInDims, collapsedInnermostLoops);
    int64_t flatInRank = flatInDims.size();
    // Flatten input last dim is all of SIMD.
    Value simdUB = flatInDims[flatInRank - 1].getValue();
    assert(flatInRank > 1 &&
           "expected at least one dim to block after the simd dim");
    // Flatten the output: only the non-reduced dims of in: -> [N][M]
    DimsExpr outDims, flatOutDims;
    create.krnlIE.getShapeAsSymbols(alloc, outDims);
    int64_t collapseOutInnermostLoop =
        isKeepDims ? collapsedInnermostLoops + 1 : 1;
    Value flatAlloc = create.mem.reshapeToFlatInnermost(
        alloc, outDims, flatOutDims, collapseOutInnermostLoop);
    int64_t flatOutRank = flatOutDims.size();
    // Flat output should have all but the flattened SIMD loop, so there
    // should only be a 1 rank difference between the two.
    assert(flatOutRank == flatInRank - 1 && "wrong assumptions about dims");

    // Parallelism only if output is not a scalar.
    if (flatOutRank == 0 && enableParallel) {
      enableParallel = false;
      onnxToKrnlParallelReport(
          op, false, -1, 0, "zero flat out rank for reduction shuffle h-simd");
    }

    // Compute type of small temp vector.
    MemRefType tmpBlockedType = MemRefType::get({VL, VL}, elementType);
    // Define loops for input dimensions, blocking the inner out dim by VL
    ValueRange outLoopDef = create.krnl.defineLoops(flatOutRank);
    ValueRange blockedOutLoopDef =
        create.krnl.block(outLoopDef[flatOutRank - 1], VL);
    // All of the non-blocked loops, plus the blocked output loop
    SmallVector<Value, 4> optimizedOutLoopDef =
        firstFew<Value, 4>(outLoopDef, -2);
    optimizedOutLoopDef.emplace_back(blockedOutLoopDef[0]);
    // Iterate only over all but the inner loop of the flattened input.
    SmallVector<IndexExpr, 4> lbs(flatOutRank, LitIE(0));
    if (enableParallel) {
      int64_t parId;
      if (findSuitableParallelDimension(lbs, flatOutDims, 0, flatOutRank, parId,
              /*min iter for going parallel*/ 8 * VL)) {
        create.krnl.parallel(optimizedOutLoopDef[parId]);
        onnxToKrnlParallelReport(op, true, parId, lbs[parId],
            flatOutDims[parId], "reduction shuffle h-simd");
      } else {
        enableParallel = false;
        onnxToKrnlParallelReport(op, false, 0, lbs[0], flatOutDims[0],
            "not enough work for reduction shuffle h-simd");
      }
    }
    create.krnl.iterateIE(outLoopDef, optimizedOutLoopDef, lbs, flatOutDims,
        [&](const KrnlBuilder &ck, ValueRange blockedOutLoopInd) {
          MDBuilder create(ck);
          // When parallel, will stay inside; otherwise will migrate out.
          Value tmpBlockedAlloc = create.mem.alignedAlloc(tmpBlockedType);
          Value identity = getIdentityValue<ONNXReductionOp>(
              rewriter, create.getLoc(), elementType);
          Value initVec = create.vec.splat(vecType, identity);
          IndexExprScope innerScope(ck);
          IndexExpr blockedCurrIndex =
              DimIE(blockedOutLoopInd[flatOutRank - 1]);
          IndexExpr blockedUB = SymIE(flatOutDims[flatOutRank - 1].getValue());
          IndexExpr isFull =
              create.krnlIE.isTileFull(blockedCurrIndex, LitIE(VL), blockedUB);
          Value zero = create.math.constantIndex(0);
          Value isNotFullVal = create.math.slt(isFull.getValue(), zero);
          create.scf.ifThenElse(
              isNotFullVal,
              [&](const SCFBuilder &scf) {
                MDBuilder create(scf);
                // create.krnl.printf("partial tile\n");
                Value startOfLastBlockVal = blockedCurrIndex.getValue();
                Value blockedUBVal = blockedUB.getValue();
                create.scf.forLoop(startOfLastBlockVal, blockedUBVal, 1,
                    [&](const SCFBuilder &scf, ValueRange loopInd) {
                      MDBuilder create(scf);
                      Value blockLocalInd = loopInd[0];
                      // Output induction variables: same as the outer loop, but
                      // with the blocked index replaced by the inner index.
                      SmallVector<Value, 4> outLoopInd =
                          firstFew<Value, 4>(blockedOutLoopInd, -2);
                      outLoopInd.emplace_back(blockLocalInd);
                      // Perform reduction for one output value.
                      genOneHorizontalSimdReduction(rewriter, create, op,
                          elementType, vecType, tmpBlockedAlloc, flatInput,
                          flatAlloc, initVec, divisorForMean, outLoopInd,
                          simdUB, VL, simdOnly);
                    }); /* for inside blocked loop */
              },
              [&](const SCFBuilder &scf) {
                MDBuilder create(scf);
                // create.krnl.printf("full tile\n");
                genVlHorizontalSimdReduction(rewriter, create, op, elementType,
                    vecType, tmpBlockedAlloc, flatInput, flatAlloc, initVec,
                    divisorForMean, blockedOutLoopInd, blockedCurrIndex, simdUB,
                    VL, simdOnly);
              });
        }); /* blocked out loop */
  }

}; /* struct ONNXReductionOpLowering */

void populateLoweringONNXReductionOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD,
    bool enableParallel) {
  patterns.insert<
      ONNXReductionOpLowering<mlir::ONNXReduceMaxV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceMeanV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceMinV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceProdV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceSumV11Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceMaxOp, RLegacy::Latest>,
      ONNXReductionOpLowering<mlir::ONNXReduceMeanOp, RLegacy::Latest>,
      ONNXReductionOpLowering<mlir::ONNXReduceMinOp, RLegacy::Latest>,
      ONNXReductionOpLowering<mlir::ONNXReduceProdOp, RLegacy::Latest>,
      ONNXReductionOpLowering<mlir::ONNXReduceSumOp, RLegacy::Latest>>(
      typeConverter, ctx, enableSIMD, enableParallel);
}
} // namespace onnx_mlir
