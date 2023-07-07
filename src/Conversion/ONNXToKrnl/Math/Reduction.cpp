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

using namespace mlir;

namespace onnx_mlir {

enum RLegacy { Latest, UpTo13 };

// Identity values
template <>
Value getIdentityValue<ONNXReduceMaxOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.negativeInf(type);
}

template <>
Value getIdentityValue<ONNXReduceMaxV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.negativeInf(type);
}

template <>
Value getIdentityValue<ONNXReduceMinOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.positiveInf(type);
}

template <>
Value getIdentityValue<ONNXReduceMinV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.positiveInf(type);
}

template <>
Value getIdentityValue<ONNXReduceProdOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 1);
}

template <>
Value getIdentityValue<ONNXReduceProdV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 1);
}

template <>
Value getIdentityValue<ONNXReduceSumV11Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 0);
}

template <>
Value getIdentityValue<ONNXReduceSumOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 0);
}

template <>
Value getIdentityValue<ONNXReduceMeanOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 0);
}

template <>
Value getIdentityValue<ONNXReduceMeanV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  MathBuilder createMath(rewriter, loc);
  return createMath.constant(type, 0);
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

  ONNXReductionOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableSIMD, bool computeMean = false)
      : OpConversionPattern<ONNXReductionOp>(typeConverter, ctx),
        enableSIMD(enableSIMD), computeMean(computeMean) {}

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
    Type convertedType =
        this->typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefOutType = convertedType.cast<MemRefType>();
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
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder, VectorBuilder>
        create(rewriter, loc);

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
    // Find out if we have constant axes: make unique and within [0, inRank).
    std::vector<int64_t> uniqueLitAxes;
    // The noRedInnerSpan = x indicates that the x innermost dimensions have
    // no reduction in them.
    int64_t noRedInnerSpan = inRank; // Until proven otherwise, no reduction
    if (hasNoAxes) {
      if (isNoop) {
        // No axes and is noop, should we not just return the input array?
      } else {
        // No axes, perform a full reduction.
        for (int64_t i = 0; i < inRank; ++i)
          uniqueLitAxes.push_back(i);
        noRedInnerSpan = 0; // No dimensions without a reduction.
      }
    } else if (!dynamicAxes) {
      // Check raw axes.
      int64_t rawAxesRank = rawAxesIE.size();
      for (int64_t i = 0; i < rawAxesRank; ++i) {
        if (!rawAxesIE[i].isLiteral()) {
          dynamicAxes = true; // Unknown axes is being reduced.
          noRedInnerSpan = 0; // Possibly no dimension without a reduction.
          break;
        }
        // Has a literal, normalize it; make sure it is unique
        int64_t axis = rawAxesIE[i].getLiteral();
        if (axis < -inRank || axis > inRank - 1) {
          return emitError(loc, "axes value out of range");
        }
        int64_t newAxis = axis >= 0 ? axis : (inRank + axis);
        if (std::find(uniqueLitAxes.begin(), uniqueLitAxes.end(), newAxis) ==
            uniqueLitAxes.end()) {
          // Has a new unique literal axes, save it.
          uniqueLitAxes.push_back(newAxis);
          int64_t span = inRank - (newAxis + 1);
          noRedInnerSpan = span < noRedInnerSpan ? span : noRedInnerSpan;
        }
      }
    } else {
      // Already dynamic.
      noRedInnerSpan = 0; // Possibly no dimension without a reduction.
    }

    //////////////////////////////////////////////////////////////////////
    // Process axes.
    // With static axes, use this
    std::map<int64_t, int64_t> outInDimMap;
    // With dynamic axes, use this
    Value maskVal = nullptr;
    Value falseVal = nullptr;
    Value trueVal = nullptr;
    Value valueOne = nullptr;
    int64_t uVL = 0; // No SIMD.
    if (!dynamicAxes) {
      // All axes are static, fill in the outInDimMap appropriately.
      assert(noRedInnerSpan >= 0 && noRedInnerSpan <= inRank && "bad span");
      outInDimMap =
          getReductionMapping(memRefInType, uniqueLitAxes, isKeepdims);
      if (enableSIMD && noRedInnerSpan > 0) {
        LLVM_DEBUG(llvm::dbgs() << "  SIMD: study if possible along the "
                                << noRedInnerSpan << " innermost dim(s)\n";);
        VectorMachineSupport *vms =
            VectorMachineSupport::getGlobalVectorMachineSupport();
        DimsExpr inputDims;
        create.krnlIE.getShapeAsSymbols(input, inputDims);
        uVL = create.vec.SuitableUnrollFactor(
            vms, memRefInType, inputDims, noRedInnerSpan, 4, /*canPad*/ false);
        LLVM_DEBUG(llvm::dbgs()
                   << "  SIMD " << (uVL ? "" : "im")
                   << "possible with vector length " << uVL << "\n");
      }
    } else {
      // Has one or more dynamic axes.
      assert(noRedInnerSpan == 0 && "expected no span");
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

    //////////////////////////////////////////////////////////////////////
    // There are two required and one optional Krnl loops:
    // - One to initialize the result memref,
    // - One to do reduction, and
    // - One to compute mean (optional).

    // 1. Define loops to initialize the result.
    ValueRange loop1Def = create.krnl.defineLoops(outRank);
    SmallVector<IndexExpr, 4> lbs1(outRank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> ubs1;
    create.krnlIE.getShapeAsSymbols(alloc, ubs1);
    create.krnl.iterateIE(loop1Def, loop1Def, lbs1, ubs1,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          Value identity =
              getIdentityValue<ONNXReductionOp>(rewriter, loc, elementOutType);
          createKrnl.store(identity, alloc, loopInd);
        });

    // 2. Define an Krnl loop to do reduction.
    if (false && uVL > 0) {
      // Create flattened rank with unrolled inner loop by VL
      int64_t flatRank = inRank - noRedInnerSpan + 1;
      ValueRange loop2Def = create.krnl.defineLoops(inRank);
      SmallVector<IndexExpr, 4> inputDims;
      create.krnlIE.getShapeAsSymbols(input, inputDims);
      DimsExpr flatInputDims;
      Value flatInput = create.mem.reshapeToFlat(
          input, inputDims, flatInputDims, noRedInnerSpan);

      SmallVector<IndexExpr, 4> lbs2(flatRank, LiteralIndexExpr(0));

    } else {
      ValueRange loop2Def = create.krnl.defineLoops(inRank);
      SmallVector<IndexExpr, 4> lbs2(inRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs2;
      create.krnlIE.getShapeAsSymbols(input, ubs2);
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
            accumulated = emitScalarOpFor<ONNXReductionOp>(rewriter, loc, op,
                memRefOutType.getElementType(), {accumulated, next});
            create.krnl.store(accumulated, alloc, accumulatorAccessFct);
          });
    }
    // 3. Define an Krnl loop to compute mean (optional).
    if (computeMean) {
      Type elementType = memRefOutType.getElementType();
      // Compute the divisor that is the number of elements participated in
      // reduction, i.e., 'divisor = size of input / size of output'.
      IndexExprScope scope(&rewriter, loc);
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
      Value divisor = divisorExpr.getValue();
      if (elementType.isa<FloatType>()) {
        divisor = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIntegerType(64), divisor);
        divisor = rewriter.create<arith::UIToFPOp>(loc, elementType, divisor);
      } else if (elementType.isa<IntegerType>())
        divisor = create.math.cast(elementType, divisor);
      else
        llvm_unreachable("unsupported element type");

      // Compute mean
      ValueRange loop3Def = create.krnl.defineLoops(outRank);
      SmallVector<IndexExpr, 4> lbs3(outRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs3;
      create.krnlIE.getShapeAsSymbols(alloc, ubs3);
      create.krnl.iterateIE(loop3Def, loop3Def, lbs3, ubs3,
          [&](KrnlBuilder &kb, ValueRange loopInd) {
            MultiDialectBuilder<KrnlBuilder, MathBuilder> create(kb);
            Value loadData = create.krnl.load(alloc, loopInd);
            Value meanVal = create.math.div(loadData, divisor);
            create.krnl.store(meanVal, alloc, loopInd);
          });
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXReductionOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD) {
  patterns.insert<
      ONNXReductionOpLowering<mlir::ONNXReduceMaxV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceMinV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceProdV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceSumV11Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceMaxOp, RLegacy::Latest>,
      ONNXReductionOpLowering<mlir::ONNXReduceMinOp, RLegacy::Latest>,
      ONNXReductionOpLowering<mlir::ONNXReduceProdOp, RLegacy::Latest>,
      ONNXReductionOpLowering<mlir::ONNXReduceSumOp, RLegacy::Latest>>(
      typeConverter, ctx, enableSIMD);
  patterns.insert<
      ONNXReductionOpLowering<mlir::ONNXReduceMeanV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLowering<mlir::ONNXReduceMeanOp, RLegacy::Latest>>(
      typeConverter, ctx, enableSIMD, /*computeMean=*/true);
}

} // namespace onnx_mlir
