/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Reduction.cpp - Lowering Reduction Ops ----------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reduction Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"
#define DEBUG_FORCE_SHUFFLE_REDUCTION 0

using namespace mlir;

namespace onnx_mlir {

// support

// Until num, inclusive. Negative numbers count from the back of the vector.
template <typename T, unsigned N>
SmallVector<T, N> firstFew(ValueRange vec, int64_t untilNum) {
  SmallVector<T, N> res;
  int64_t size = vec.size();
  if (untilNum < 0)
    untilNum += size;
  // If untilNum<0...  we get an empty vector, that is ok.
  assert(untilNum < size && "out of bound");
  for (int64_t i = 0; i <= untilNum; ++i)
    res.emplace_back(vec[i]);
  return res;
}

template <typename T, unsigned N>
SmallVector<T, N> firstFew(ArrayRef<T> vec, int64_t untilNum) {
  SmallVector<T, N> res;
  int64_t size = vec.size();
  if (untilNum < 0)
    untilNum += size;
  // If untilNum<0...  we get an empty vector, that is ok.
  assert(untilNum < size && "out of bound");
  for (int64_t i = 0; i <= untilNum; ++i)
    res.emplace_back(vec[i]);
  return res;
}

template <typename T, unsigned N>
SmallVector<T, N> firstFew(SmallVectorImpl<T> &vec, int64_t untilNum) {
  SmallVector<T, N> res;
  int64_t size = vec.size();
  if (untilNum < 0)
    untilNum += size;
  // If untilNum<0...  we get an empty vector, that is ok.
  assert(untilNum < size && "out of bound");
  for (int64_t i = 0; i <= untilNum; ++i)
    res.emplace_back(vec[i]);
  return res;
}

// From num, inclusive.  Negative numbers count from the back of the vector.
template <typename T, unsigned N>
SmallVector<T, N> lastFew(ValueRange vec, int64_t fromNum) {
  SmallVector<T, N> res;
  int64_t size = vec.size();
  if (fromNum < 0)
    fromNum += size;
  // If fromNum>= size...  we get an empty vector, that is ok.
  assert(fromNum >= 0 && "out of bound");
  for (int64_t i = fromNum; i < size; ++i)
    res.emplace_back(vec[i]);
  return res;
}

template <typename T, unsigned N>
SmallVector<T, N> lastFew(ArrayRef<T> vec, int64_t fromNum) {
  SmallVector<T, N> res;
  int64_t size = vec.size();
  if (fromNum < 0)
    fromNum += size;
  // If fromNum>= size...  we get an empty vector, that is ok.
  assert(fromNum >= 0 && "out of bound");
  for (int64_t i = fromNum; i < size; ++i)
    res.emplace_back(vec[i]);
  return res;
}

template <typename T, unsigned N>
SmallVector<T, N> lastFew(SmallVectorImpl<T> &vec, int64_t fromNum) {
  SmallVector<T, N> res;
  int64_t size = vec.size();
  if (fromNum < 0)
    fromNum += size;
  // If fromNum>= size...  we get an empty vector, that is ok.
  assert(fromNum >= 0 && "out of bound");
  for (int64_t i = fromNum; i < size; ++i)
    res.emplace_back(vec[i]);
  return res;
}
// end support

enum RLegacy { Latest, UpTo13 };

//
template <typename OP>
VectorBuilder::CombiningKind getCombiningKind() {
  llvm_unreachable("illegal combination kind");
}

// Identity values
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
Value getIdentityValue<ONNXReduceMaxV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.negativeInf(type);
}

template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceMaxV13Op>() {
  return VectorBuilder::CombiningKind::MAX;
}

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
Value getIdentityValue<ONNXReduceMinV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.positiveInf(type);
}

template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceMinV13Op>() {
  return VectorBuilder::CombiningKind::MIN;
}

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
Value getIdentityValue<ONNXReduceProdV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 1);
}

template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceProdV13Op>() {
  return VectorBuilder::CombiningKind::MUL;
}

template <>
Value getIdentityValue<ONNXReduceSumV11Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 0);
}

template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceSumV11Op>() {
  return VectorBuilder::CombiningKind::ADD;
}

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
Value getIdentityValue<ONNXReduceMeanV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 0);
}

template <>
VectorBuilder::CombiningKind getCombiningKind<ONNXReduceMeanV13Op>() {
  return VectorBuilder::CombiningKind::ADD;
}

// Scalar ops
template <>
struct ScalarOp<ONNXReduceProdV13Op> {
  using FOp = arith::MulFOp;
  using IOp = arith::MulIOp;
};

template <>
struct ScalarOp<ONNXReduceProdOp> {
  using FOp = arith::MulFOp;
  using IOp = arith::MulIOp;
};

template <>
struct ScalarOp<ONNXReduceSumV11Op> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

template <>
struct ScalarOp<ONNXReduceSumOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

template <>
struct ScalarOp<ONNXReduceMeanV13Op> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

template <>
struct ScalarOp<ONNXReduceMeanOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMaxOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXReduceMaxV13Op>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  MathBuilder createMath(rewriter, loc);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  Value max = createMath.sgt(lhs, rhs);
  return createMath.select(max, lhs, rhs);
}

template <>
Value emitScalarOpFor<ONNXReduceMaxOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  MathBuilder createMath(rewriter, loc);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  Value max = createMath.sgt(lhs, rhs);
  return createMath.select(max, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMinOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXReduceMinV13Op>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  MathBuilder createMath(rewriter, loc);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  Value min = createMath.slt(lhs, rhs);
  return createMath.select(min, lhs, rhs);
}

template <>
Value emitScalarOpFor<ONNXReduceMinOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  MathBuilder createMath(rewriter, loc);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  Value min = createMath.slt(lhs, rhs);
  return createMath.select(min, lhs, rhs);
}

// This duplicated code can be eliminated with if constexpr in c++ 17
// Or onnx uses input for axes for all ops
template <typename ONNXReductionOp, RLegacy legacyOp>
struct ONNXReductionOpLowering : public OpConversionPattern<ONNXReductionOp> {
  using OpAdaptor = typename ONNXReductionOp::Adaptor;
  bool enableSIMD = false;
  bool computeMean = false;
  bool enableParallel = false;

  using MDBuilder =
      MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
          MemRefBuilder, VectorBuilder, AffineBuilderKrnlMem, SCFBuilder>;

  ONNXReductionOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableSIMD, bool computeMean = false, bool enableParallel = false)
      : OpConversionPattern<ONNXReductionOp>(typeConverter, ctx),
        enableSIMD(enableSIMD), computeMean(computeMean),
        enableParallel(enableParallel) {}

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
    MemRefType memRefInType = input.getType().cast<MemRefType>();
    Type convertedOutType =
        this->typeConverter->convertType(*op->result_type_begin());
    assert(convertedOutType && convertedOutType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefOutType = convertedOutType.cast<MemRefType>();
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
        // Check it has a rank of 1.
        assert(
            create.krnlIE.getShapedTypeRank(axesVal) == 1 && "expect rank 1");
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
    int64_t VL = 0;
    int64_t estimatedSimdLoopTripCount;

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
          VectorMachineSupport *vms =
              VectorMachineSupport::getGlobalVectorMachineSupport();
          DimsExpr inputDims;
          create.krnlIE.getShapeAsSymbols(input, inputDims);
          int64_t unroll = 4;
          if (horizontalSimd) {
#if !DEBUG_FORCE_SHUFFLE_REDUCTION
            VectorBuilder::CombiningKind kind =
                getCombiningKind<ONNXReductionOp>();
            hasHorizontalSimdSupport =
                supportedHorizontalSIMDOp(vms, kind, elementOutType);
#endif
            if (!hasHorizontalSimdSupport) {
              // Does not have SIMD horizontal support, so use a scheme that
              // unroll the innermost non-simd loop by VL. Because trip counts
              // of such loops could be small (e.g. GPT2 = 8), we don't want a
              // large VL here.
              unroll = 1;
            }
          }
          LLVM_DEBUG(llvm::dbgs()
                     << "  SIMD: study with init unroll " << unroll << "\n");
          VL = create.vec.computeSuitableUnrollFactor(vms, memRefInType,
              inputDims, innermostLoopCollapse, unroll, /*canPad*/ false,
              estimatedSimdLoopTripCount);
          LLVM_DEBUG(llvm::dbgs() << "  SIMD: " << innermostLoopCollapse
                                  << " loops, VL " << VL << "\n");
          if (!VL) {
            horizontalSimd = parallelSimd = false;
            LLVM_DEBUG(llvm::dbgs() << "  SIMD: no good VL\n");
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
      assert(convertedMaskType && convertedMaskType.isa<MemRefType>() &&
             "Failed to convert type to MemRefType");
      MemRefType maskTypeInMemRefType = convertedMaskType.cast<MemRefType>();
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
          axesVal.getType().cast<MemRefType>().getElementType();
      auto dataDimConst = create.math.constant(axesElementType, inRank);
      Value zeroValue = create.math.constant(axesElementType, 0);
      if (!axisShape0.isLiteral()) {
        // When axes is dynamic, generate a Krnl loop
        KrnlBuilder createKrnl(rewriter, loc);
        ValueRange loopDef = createKrnl.defineLoops(1);
        if (enableParallel) {
          create.krnl.parallel(loopDef[0]);
          LLVM_DEBUG(
              llvm::dbgs() << "[Parallel Op]: " << op->getName() << "\n");
        }
        createKrnl.iterateIE(loopDef, loopDef, {LiteralIndexExpr(0)},
            {axisShape0}, [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
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
    LLVM_DEBUG(llvm::dbgs() << "  SIMD " << (VL ? "" : "im")
                            << "possible with vector length " << VL << "\n");

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
    if (computeMean) {
      // Compute the divisor that is the number of elements participated in
      // reduction, i.e., 'divisor = size of input / size of output'.
      IndexExprScope scope(create.krnl);
      IndexExpr inputSizeExpr = LiteralIndexExpr(1);
      for (unsigned i = 0; i < inRank; i++) {
        IndexExpr dimExpr = create.krnlIE.getShapeAsSymbol(input, i);
        inputSizeExpr = inputSizeExpr * dimExpr;
      }
      IndexExpr outputSizeExpr = LiteralIndexExpr(1);
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
            alloc, inRank, outRank, VL, innermostLoopCollapse, isKeepdims,
            divisorForMean, enableParallel);
        onnxToKrnlSimdReport(op, /*successful*/ true, VL,
            estimatedSimdLoopTripCount, "horizontal");
      } else {
        genShuffleHorizontalSimdReduction(rewriter, create, op, elementOutType,
            input, alloc, inRank, outRank, VL, innermostLoopCollapse,
            isKeepdims, divisorForMean, enableParallel);
        onnxToKrnlSimdReport(op, /*successful*/ true, VL,
            estimatedSimdLoopTripCount, "shuffle-horizontal");
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
          op, /*successful*/ false, /*vl*/ 0, estimatedSimdLoopTripCount, msg);
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }

  void genScalarReduction(ConversionPatternRewriter &rewriter,
      MDBuilder &create, Operation *op, Type elementType, Value input,
      Value alloc, int64_t inRank, int64_t outRank, bool dynamicAxes,
      Value maskVal, std::map<int64_t, int64_t> &outInDimMap,
      Value divisorForMean, bool enableParallel) const {
    //////////////////////////////////////////////////////////////////////
    // There are two required and one optional Krnl loops:
    // - One to initialize the result memref,
    // - One to do reduction, and
    // - One to compute mean (optional).

    // 1. Define loops to initialize the result.
    // Todo: consider using a krnl.memset
    ValueRange loop1Def = create.krnl.defineLoops(outRank);
    SmallVector<IndexExpr, 4> lbs1(outRank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> ubs1;
    create.krnlIE.getShapeAsSymbols(alloc, ubs1);
    if (enableParallel) {
      create.krnl.parallel(loop1Def[0]);
      LLVM_DEBUG(llvm::dbgs() << "[Parallel Op]: " << op->getName() << "\n");
    }
    create.krnl.iterateIE(loop1Def, loop1Def, lbs1, ubs1,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          Value identity = getIdentityValue<ONNXReductionOp>(
              rewriter, create.getLoc(), elementType);
          createKrnl.store(identity, alloc, loopInd);
        });

    ValueRange loop2Def = create.krnl.defineLoops(inRank);
    SmallVector<IndexExpr, 4> lbs2(inRank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> ubs2;
    create.krnlIE.getShapeAsSymbols(input, ubs2);
    Value trueVal = create.math.constant(rewriter.getIntegerType(1), 1);
    // TODO Temporary disable the 2nd loop parallelism, since its outermost
    // loop could be a reduction loop, where parallelism would not be safe.
    // if (enableParallel) {
    //   create.krnl.parallel(loop2Def[0]);
    //   LLVM_DEBUG(llvm::dbgs() << "[Parallel Op]: " << op->getName() << "\n");
    // }
    create.krnl.iterateIE(loop2Def, loop2Def, lbs2, ubs2,
        [&](KrnlBuilder &kb, ValueRange loopInd) {
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
    if (computeMean) {
      // Compute mean
      ValueRange loop3Def = create.krnl.defineLoops(outRank);
      SmallVector<IndexExpr, 4> lbs3(outRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs3;
      create.krnlIE.getShapeAsSymbols(alloc, ubs3);
      if (enableParallel) {
        create.krnl.parallel(loop3Def[0]);
        LLVM_DEBUG(llvm::dbgs() << "[Parallel Op]: " << op->getName() << "\n");
      }
      create.krnl.iterateIE(loop3Def, loop3Def, lbs3, ubs3,
          [&](KrnlBuilder &kb, ValueRange loopInd) {
            MultiDialectBuilder<KrnlBuilder, MathBuilder> create(kb);
            Value loadData = create.krnl.load(alloc, loopInd);
            Value meanVal = create.math.div(loadData, divisorForMean);
            create.krnl.store(meanVal, alloc, loopInd);
          });
    }
  }

  bool supportedHorizontalSIMDOp(VectorMachineSupport *vms,
      VectorBuilder::CombiningKind getCombiningKind, Type elementType) const {
    int64_t len;
    switch (getCombiningKind) {
    case VectorBuilder::CombiningKind::ADD:
      len = vms->getVectorLength(GenericOps::SumAcrossGop, elementType);
      break;
    case VectorBuilder::CombiningKind::MIN:
    case VectorBuilder::CombiningKind::MAX:
      len = vms->getVectorLength(GenericOps::SumAcrossGop, elementType);
      break;
    default:
      len = 0;
    }
    return len != 0;
  }

  void genOneHorizontalSimdReduction(ConversionPatternRewriter &rewriter,
      MDBuilder &create, Operation *op, Type elementType, VectorType vecType,
      Value tmpAlloca, Value flatInput, Value flatAlloc, Value initVec,
      Value divisorForMean, ValueRange outLoopInd, Value simdUB,
      int64_t VL) const {
    // Init temp memory to init values.
    Value zero = create.math.constantIndex(0);
    create.vec.store(initVec, tmpAlloca, {zero, zero});
    // Iterate over the SIMD blocks.
    ValueRange simdLoopDef = create.krnl.defineLoops(1);
    ValueRange blockedSimdLoopDef = create.krnl.block(simdLoopDef[0], VL);
    create.krnl.iterate(simdLoopDef, {blockedSimdLoopDef[0]}, {zero}, {simdUB},
        [&](KrnlBuilder &ck, ValueRange simdLoopInd) {
          MDBuilder create(ck);
          // Input values, loaded as a vector.
          SmallVector<Value, 4> inAccessVals(outLoopInd);
          inAccessVals.emplace_back(simdLoopInd[0]);
          Value inputVec = create.vec.load(vecType, flatInput, inAccessVals);
          Value tmpVec = create.vec.load(vecType, tmpAlloca, {zero, zero});
          // Sum into redVec
          Value accumulatedVec = emitScalarOpFor<ONNXReductionOp>(
              rewriter, create.getLoc(), op, vecType, {tmpVec, inputVec});
          create.vec.store(accumulatedVec, tmpAlloca, {zero, zero});
        });
    // Horizontal sum.
    Value reductionVec = create.vec.load(vecType, tmpAlloca, {zero, zero});
    Value accumulatedVal =
        create.vec.reduction(getCombiningKind<ONNXReductionOp>(), reductionVec);
    // other operation...
    if (computeMean) {
      accumulatedVal = create.math.div(accumulatedVal, divisorForMean);
    }
    // Store tmp into result.
    create.krnl.store(accumulatedVal, flatAlloc, outLoopInd);
  }

  void genHorizontalSimdReduction(ConversionPatternRewriter &rewriter,
      MDBuilder &create, Operation *op, Type elementType, Value input,
      Value alloc, int64_t inRank, int64_t outRank, int64_t VL,
      int64_t collapsedInnermostLoops, bool isKeepDims, Value divisorForMean,
      bool enableParallel) const {

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
    // Flat output should have all but the flattened SIMD loop, so there should
    // only be a 1 rank difference between the two.
    assert(flatOutRank == flatInRank - 1 && "wrong assumptions about dims");

    // Alloca a small temp vector.
    MemRefType tmpType = MemRefType::get({1, VL}, elementType);
    Value tmpAlloca = create.mem.alignedAlloca(tmpType);
    // Define loops for input dimensions, blocking the inner dim by VL
    ValueRange outLoopDef = create.krnl.defineLoops(flatOutRank);
    SmallVector<IndexExpr, 4> lbs(flatOutRank, LiteralIndexExpr(0));
    if (enableParallel) {
      create.krnl.parallel(outLoopDef[0]);
      LLVM_DEBUG(llvm::dbgs() << "[Parallel Op]: " << op->getName() << "\n");
    }
    create.krnl.iterateIE(outLoopDef, outLoopDef, lbs, flatOutDims,
        [&](KrnlBuilder &ck, ValueRange outLoopInd) {
          MDBuilder create(ck);
          Value identity = getIdentityValue<ONNXReductionOp>(
              rewriter, create.getLoc(), elementType);
          Value initVec = create.vec.splat(vecType, identity);
          genOneHorizontalSimdReduction(rewriter, create, op, elementType,
              vecType, tmpAlloca, flatInput, flatAlloc, initVec, divisorForMean,
              outLoopInd, simdUB, VL);
        });
  }

  void genVlHorizontalSimdReduction(ConversionPatternRewriter &rewriter,
      MDBuilder &create, Operation *op, Type elementType, VectorType vecType,
      Value tmpBlockedAlloca, Value flatInput, Value flatAlloc, Value initVec,
      Value divisorForMean, ValueRange blockedOutLoopInd,
      IndexExpr blockedCurrIndex, Value simdUB, int64_t VL) const {
    // Init temp memory to init values.
    Value zero = create.math.constantIndex(0);
    for (int64_t i = 0; i < VL; ++i) {
      create.vec.store(
          initVec, tmpBlockedAlloca, {create.math.constantIndex(i), zero});
    }
    // Blocked Simd loop.
    ValueRange simdLoopDef = create.krnl.defineLoops(1);
    ValueRange blockedSimdLoopDef = create.krnl.block(simdLoopDef[0], VL);
    create.krnl.iterate(simdLoopDef, {blockedSimdLoopDef[0]}, {zero}, {simdUB},
        [&](KrnlBuilder &ck, ValueRange simdLoopInd) {
          MDBuilder create(ck);
          // Loop over blocked output loop, block guaranteed to be full.
          for (int64_t i = 0; i < VL; ++i) {
            IndexExpr offset = LiteralIndexExpr(i);
            IndexExpr blockLocalIndIE = blockedCurrIndex + offset;
            Value blockLocalInd = blockLocalIndIE.getValue();
            // All of the non-blocked loop, plus the inter tile index of the
            // blocked loop, and the blocked simd loop.
            SmallVector<Value, 4> inAccessVals =
                firstFew<Value, 4>(blockedOutLoopInd, -2);
            inAccessVals.emplace_back(blockLocalInd);
            inAccessVals.emplace_back(simdLoopInd[0]);
            Value inputVec = create.vec.load(vecType, flatInput, inAccessVals);
            // The tmpInd value is between 0 and VL-1, and is local index -
            // blocked index.
            Value tmpInd = offset.getValue();
            Value tmpVec =
                create.vec.load(vecType, tmpBlockedAlloca, {tmpInd, zero});
            // Sum into redVec
            Value accumulatedVec = emitScalarOpFor<ONNXReductionOp>(
                rewriter, create.getLoc(), op, vecType, {tmpVec, inputVec});
            create.vec.store(accumulatedVec, tmpBlockedAlloca, {tmpInd, zero});
          } /* intra block output loop */
        }); /* blocked simd loop */
    // Load all temp vectors.
    SmallVector<Value, 4> redIn, redOut;
    for (int64_t i = 0; i < VL; ++i) {
      Value val = create.vec.load(
          vecType, tmpBlockedAlloca, {create.math.constantIndex(i), zero});
      redIn.emplace_back(val);
    }
    // Reduce all of the temp vectors at once.
    auto redFct = [&](Value a, Value b) -> Value {
      return emitScalarOpFor<ONNXReductionOp>(
          rewriter, create.getLoc(), op, vecType, {a, b});
    };
    create.vec.multiReduction(redIn, redFct, redOut);
    // The redOut list should have one value with SIMD of VL.
    assert(redOut.size() == 1 && "expected only one val");
    Value accumulatedVal = redOut[0];
    // Perform the mean computation if required.
    if (computeMean) {
      Value divisorForMeanVec = create.vec.splat(vecType, divisorForMean);
      accumulatedVal = create.math.div(accumulatedVal, divisorForMeanVec);
    }
    // Store final values.
    create.vec.store(accumulatedVal, flatAlloc, blockedOutLoopInd);
  }

  // Solution when there is no horizontal SIMD op support and that shuffle ops
  // are needed.
  void genShuffleHorizontalSimdReduction(ConversionPatternRewriter &rewriter,
      MDBuilder &create, Operation *op, Type elementType, Value input,
      Value alloc, int64_t inRank, int64_t outRank, int64_t VL,
      int64_t collapsedInnermostLoops, bool isKeepDims, Value divisorForMean,
      bool enableParallel) const {

    assert(VL > 1 && "expected simd here");
    IndexExpr VLIndexExpr = LiteralIndexExpr(VL);
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
    // Flat output should have all but the flattened SIMD loop, so there should
    // only be a 1 rank difference between the two.
    assert(flatOutRank == flatInRank - 1 && "wrong assumptions about dims");

    // Alloca a small temp vector.
    MemRefType tmpBlockedType = MemRefType::get({VL, VL}, elementType);
    Value tmpBlockedAlloca = create.mem.alignedAlloca(tmpBlockedType);
    // Define loops for input dimensions, blocking the inner out dim by VL
    ValueRange outLoopDef = create.krnl.defineLoops(flatOutRank);
    ValueRange blockedOutLoopDef =
        create.krnl.block(outLoopDef[flatOutRank - 1], VL);
    // All of the non-blocked loops, plus the blocked output loop
    SmallVector<Value, 4> optimizedOutLoopDef =
        firstFew<Value, 4>(outLoopDef, -2);
    optimizedOutLoopDef.emplace_back(blockedOutLoopDef[0]);
    // Iterate only over all but the inner loop of the flattened input.
    SmallVector<IndexExpr, 4> lbs(flatOutRank, LiteralIndexExpr(0));
    if (enableParallel) {
      create.krnl.parallel(optimizedOutLoopDef[0]);
      LLVM_DEBUG(llvm::dbgs() << "[Parallel Op]: " << op->getName() << "\n");
    }
    create.krnl.iterateIE(outLoopDef, optimizedOutLoopDef, lbs, flatOutDims,
        [&](KrnlBuilder &ck, ValueRange blockedOutLoopInd) {
          MDBuilder create(ck);
          Value identity = getIdentityValue<ONNXReductionOp>(
              rewriter, create.getLoc(), elementType);
          Value initVec = create.vec.splat(vecType, identity);
          IndexExprScope innerScope(ck);
          IndexExpr blockedCurrIndex =
              DimIndexExpr(blockedOutLoopInd[flatOutRank - 1]);
          IndexExpr blockedUB =
              SymbolIndexExpr(flatOutDims[flatOutRank - 1].getValue());
          IndexExpr isFull = create.krnlIE.isTileFull(
              blockedCurrIndex, LiteralIndexExpr(VL), blockedUB);
          Value zero = create.math.constantIndex(0);
          Value isNotFullVal = create.math.slt(isFull.getValue(), zero);
          create.scf.ifThenElse(
              isNotFullVal,
              [&](SCFBuilder &scf) {
                MDBuilder create(scf);
                // create.krnl.printf("partial tile\n");
                Value startOfLastBlockVal = blockedCurrIndex.getValue();
                Value blockedUBVal = blockedUB.getValue();
                create.scf.forLoop(startOfLastBlockVal, blockedUBVal, 1,
                    [&](SCFBuilder &scf, Value blockLocalInd) {
                      MDBuilder create(scf);
                      // Output induction variables: same as the outer loop, but
                      // with the blocked index replaced by the inner index.
                      SmallVector<Value, 4> outLoopInd =
                          firstFew<Value, 4>(blockedOutLoopInd, -2);
                      outLoopInd.emplace_back(blockLocalInd);
                      // Perform reduction for one output value.
                      genOneHorizontalSimdReduction(rewriter, create, op,
                          elementType, vecType, tmpBlockedAlloca, flatInput,
                          flatAlloc, initVec, divisorForMean, outLoopInd,
                          simdUB, VL);
                    }); /* for inside blocked loop */
              },
              [&](SCFBuilder &scf) {
                MDBuilder create(scf);
                // create.krnl.printf("full tile\n");
                genVlHorizontalSimdReduction(rewriter, create, op, elementType,
                    vecType, tmpBlockedAlloca, flatInput, flatAlloc, initVec,
                    divisorForMean, blockedOutLoopInd, blockedCurrIndex, simdUB,
                    VL);
              });
        }); /* blocked out loop */
  }

}; /* struct ONNXReductionOpLowering */

void populateLoweringONNXReductionOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD,
    bool enableParallel) {
  patterns.insert<
      ONNXReductionOpLowering<mlir::ONNXReduceMaxV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceMinV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceProdV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceSumV11Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceMaxOp, RLegacy::Latest>,
      ONNXReductionOpLowering<mlir::ONNXReduceMinOp, RLegacy::Latest>,
      ONNXReductionOpLowering<mlir::ONNXReduceProdOp, RLegacy::Latest>,
      ONNXReductionOpLowering<mlir::ONNXReduceSumOp, RLegacy::Latest>>(
      typeConverter, ctx, enableSIMD, enableParallel);
  patterns.insert<
      ONNXReductionOpLowering<mlir::ONNXReduceMeanV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceMeanOp, RLegacy::Latest>>(
      typeConverter, ctx, enableSIMD, /*computeMean=*/true, enableParallel);
}
} // namespace onnx_mlir
