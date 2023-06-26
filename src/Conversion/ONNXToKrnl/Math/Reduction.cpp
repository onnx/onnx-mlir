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

#if 1 // hi alex
template <typename ONNXReductionOp>
struct ONNXOldReductionOpLowering
    : public OpConversionPattern<ONNXReductionOp> {
  using OpAdaptor = typename ONNXReductionOp::Adaptor;
  bool computeMean = false;

  ONNXOldReductionOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool computeMean = false)
      : OpConversionPattern<ONNXReductionOp>(typeConverter, ctx) {
    this->computeMean = computeMean;
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
    MemRefType memRefInType = input.getType().cast<MemRefType>();
    // Convert the output type to MemRefType.
    Type convertedType =
        this->typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefOutType = convertedType.cast<MemRefType>();
    int64_t inRank = memRefInType.getRank();
    int64_t outRank = memRefOutType.getRank();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Get axes value defined by op
    // Leave empty is not defined
    std::vector<int64_t> definedAxes;
    ArrayAttr axisAttrs = adaptor.getAxesAttr();
    if (axisAttrs) {
      for (auto axisAttr : axisAttrs.getValue()) {
        int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
        definedAxes.push_back(axis);
      }
    }

    std::vector<int64_t> axes;
    if (definedAxes.size()) {
      for (auto axis : definedAxes) {
        if (axis < -inRank || axis > inRank - 1)
          return emitError(loc, "axes value out of range");
        int64_t newAxis = axis >= 0 ? axis : (inRank + axis);
        if (std::find(axes.begin(), axes.end(), newAxis) == axes.end())
          axes.push_back(newAxis);
      }
    } else
      for (decltype(inRank) i = 0; i < inRank; ++i)
        axes.push_back(i);

    // KeepDims
    auto keepdims = adaptor.getKeepdims();
    bool isKeepdims = (keepdims == 1) ? true : false;

    // Get type information
    auto elementOutType = memRefOutType.getElementType();
    std::map<int64_t, int64_t> outInDimMap =
        getReductionMapping(memRefInType, axes, isKeepdims);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(input, memRefOutType);

    // There are two required and one optional Krnl loops:
    // - One to initialize the result memref,
    // - One to do reduction, and
    // - One to compute mean (optional).

    // 1. Define loops to initialize the result.
    std::vector<Value> originalLoopsInit;
    defineLoops(rewriter, loc, originalLoopsInit, outRank);

    // Iteration information
    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack packInit(rewriter, originalLoopsInit);
    for (decltype(outRank) i = 0; i < outRank; ++i)
      addDimensionToPack(rewriter, loc, packInit, alloc, i);

    KrnlIterateOp iterateOpInit = create.krnl.iterate(packInit);
    Block &iterationBlockInit = iterateOpInit.getBodyRegion().front();

    // Perform the insertions into the body of the initialization loop.

    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlockInit);

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : iterationBlockInit.getArguments())
      loopIVs.push_back(arg);

    Value identity =
        getIdentityValue<ONNXReductionOp>(rewriter, loc, elementOutType);
    create.krnl.store(identity, alloc, loopIVs);

    // 2. Define an Krnl loop to do reduction.
    rewriter.setInsertionPointAfter(iterateOpInit);
    auto ipMainRegion = rewriter.saveInsertionPoint();
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, inRank);
    // Iteration information
    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack pack(rewriter, originalLoops);
    for (decltype(inRank) i = 0; i < inRank; ++i)
      addDimensionToPack(rewriter, loc, pack, input, i);

    KrnlIterateOp iterateOp = create.krnl.iterate(pack);
    Block &iterationBlock = iterateOp.getBodyRegion().front();

    // Perform the insertions into the body of the reduction loop.
    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation:
    SmallVector<Value, 4> inLoopIVs, outLoopIVs;
    auto args = iterationBlock.getArguments();
    for (unsigned int i = 0; i < args.size(); ++i) {
      inLoopIVs.push_back(args[i]);
    }
    Value zeroIndex = nullptr;
    for (decltype(inRank) i = 0; i < outRank; ++i) {
      if (outInDimMap.find(i) != outInDimMap.end()) {
        outLoopIVs.push_back(inLoopIVs[outInDimMap[i]]);
      } else {
        if (zeroIndex) {
          outLoopIVs.push_back(zeroIndex);
        } else {
          zeroIndex = create.math.constantIndex(0);
          outLoopIVs.push_back(zeroIndex);
        }
      }
    }

    Value next = create.krnl.load(input, inLoopIVs);
    Value accumulated = create.krnl.load(alloc, outLoopIVs);
    accumulated = emitScalarOpFor<ONNXReductionOp>(
        rewriter, loc, op, memRefOutType.getElementType(), {accumulated, next});
    create.krnl.store(accumulated, alloc, outLoopIVs);

    // 3. Define an Krnl loop to compute mean (optional).
    rewriter.restoreInsertionPoint(ipMainRegion);
    if (computeMean) {
      Type elementType = memRefOutType.getElementType();
      // Compute the divisor that is the number of elements participated in
      // reduction, i.e., 'divisor = size of input / size of output'.
      IndexExprScope scope(&rewriter, loc);
      IndexExpr inputSizeExpr = LiteralIndexExpr(1);
      for (unsigned i = 0; i < inRank; i++) {
        IndexExpr dimExpr = create.krnlIE.getShapeAsDim(input, i);
        inputSizeExpr = inputSizeExpr * dimExpr;
      }
      IndexExpr outputSizeExpr = LiteralIndexExpr(1);
      for (unsigned i = 0; i < outRank; i++) {
        IndexExpr dimExpr = create.krnlIE.getShapeAsDim(alloc, i);
        outputSizeExpr = outputSizeExpr * dimExpr;
      }
      IndexExpr divisorExpr = inputSizeExpr.floorDiv(outputSizeExpr);
      Value divisor = divisorExpr.getValue();
      if (elementType.isa<FloatType>()) {
        divisor = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIntegerType(64), divisor);
        divisor = rewriter.create<arith::UIToFPOp>(loc, elementType, divisor);
      } else if (elementType.isa<IntegerType>()) {
        divisor =
            rewriter.create<arith::IndexCastOp>(loc, elementType, divisor);
      } else
        llvm_unreachable("unsupported element type");

      // Compute mean
      ValueRange loopDef = create.krnl.defineLoops(outRank);
      SmallVector<IndexExpr, 4> lbs(outRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            Value loadData = createKrnl.load(alloc, loopInd);
            Value meanVal = create.math.div(loadData, divisor);
            createKrnl.store(meanVal, alloc, loopInd);
          });
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

// This duplicated code can be eliminated with if constexpr in c++ 17
// Or onnx uses input for axes for all ops
// legacyOp == 0 => current
// legacyOp == 1 => axis are read as attributes.
template <typename ONNXReductionOp, RLegacy legacyOp>
struct ONNXReductionOpLowering : public OpConversionPattern<ONNXReductionOp> {
  using OpAdaptor = typename ONNXReductionOp::Adaptor;
  bool computeMean = false;

  ONNXReductionOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool computeMean = false)
      : OpConversionPattern<ONNXReductionOp>(typeConverter, ctx),
        computeMean(computeMean) {}

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
    auto input = operands[0];
    auto axesVal = operands[1];

    auto memRefInType = input.getType().cast<MemRefType>();
    // Convert the output type to MemRefType.
    Type convertedType =
        this->typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefOutType = convertedType.cast<MemRefType>();
    int64_t inRank = memRefInType.getRank();
    int64_t outRank = memRefOutType.getRank();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // KeepDims
    int64_t keepdims = adaptor.getKeepdims();
    bool isKeepdims = (keepdims == 1);

    // Noop
    auto noop = llvm::dyn_cast<ONNXReductionOp>(op).getNoopWithEmptyAxes();
    bool isNoop = (noop == 1);

    // Get axes dims
    IndexExprScope mainScope(&rewriter, loc);
    DimsExpr axesDims;
    if (isNoneValue(axesVal) && !isNoop) {
      // The default is to reduce over all the dimensions of the input tensor if
      // 'noop_with_empty_axes' is false
      for (int64_t i = 0; i < inRank; ++i)
        axesDims.emplace_back(LiteralIndexExpr(i));
    } else
      create.krnlIE.getShapeAsDims(axesVal, axesDims);

    // Get type information
    auto memRefOutShape = memRefOutType.getShape();
    auto elementOutType = memRefOutType.getElementType();

    bool dynamicAxes = false;
    Value maskVal = nullptr;
    Value falseVal = nullptr;
    Value trueVal = nullptr;
    Value valueOne = nullptr;
    std::map<int64_t, int64_t> outInDimMap;

    Value axesValue = llvm::dyn_cast<ONNXReductionOp>(op).getAxes();
    // Dynamic axes
    if (!isNoneValue(axesValue) && !getONNXConstantOp(axesValue)) {
      dynamicAxes = true;
      // Handle only when keepdims == true
      if (!isKeepdims)
        return emitError(loc, "not getKeepdims() not implemented");

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
      auto axesDim = axesVal.getType().cast<MemRefType>().getShape()[0];

      // Initialize mask to 0
      // Unless noop_with_empty_axesDim is false and axesDim is
      // ShapedType::kDynamic.
      Value initVal;
      if (axesDim == ShapedType::kDynamic && !isNoop) {
        IndexExprScope axesLoopContext(&rewriter, loc);
        Value zeroIndex = create.math.constantIndex(0);
        IndexExpr axesBound0 = create.krnlIE.getShapeAsDim(axesVal, 0);
        Value cond = create.math.eq(axesBound0.getValue(), zeroIndex);
        // initVal = axes.shape[0] == 0 ? true : false
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
      if (axesDim == ShapedType::kDynamic) {
        // When axes is dynamic, generate a Krnl loop
        KrnlBuilder createKrnl(rewriter, loc);
        ValueRange loopDef = createKrnl.defineLoops(1);
        SmallVector<IndexExpr, 4> lbs(1, LiteralIndexExpr(0));
        createKrnl.iterateIE(loopDef, loopDef, lbs, axesDims,
            [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
              Value axe = createKrnl.load(axesVal, loopInd[0]);
              Value cond = create.math.slt(axe, zeroValue);
              Value dim = create.math.select(
                  cond, create.math.add(axe, dataDimConst), axe);
              Value jVal = rewriter.create<arith::IndexCastOp>(
                  loc, rewriter.getIndexType(), dim);
              createKrnl.store(trueVal, maskVal, jVal);
            });
      } else {
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
    } else {
      // Get axes value defined by op. Leave empty is not defined.
      std::vector<int64_t> definedAxes;

      // Assume it is verified that axes are known. Convert DenseElementsAttr to
      // ArrayAttr.
      if (!isNoneValue(axesValue) && getONNXConstantOp(axesValue)) {
        auto constAxes = getONNXConstantOp(axesValue)
                             .getValueAttr()
                             .dyn_cast_or_null<mlir::DenseElementsAttr>();
        for (auto element : constAxes.getValues<IntegerAttr>())
          definedAxes.push_back(element.getInt());
      }

      std::vector<int64_t> axes;
      if (definedAxes.size()) {
        for (auto axis : definedAxes) {
          if (axis < -inRank || axis > inRank - 1) {
            return emitError(loc, "axes value out of range");
          }
          int64_t newAxis = axis >= 0 ? axis : (inRank + axis);
          if (std::find(axes.begin(), axes.end(), newAxis) == axes.end())
            axes.push_back(newAxis);
        }
      } else if (!isNoop) {
        for (decltype(inRank) i = 0; i < inRank; ++i) {
          axes.push_back(i);
        }
      }
      outInDimMap = getReductionMapping(memRefInType, axes, isKeepdims);
    }

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    if (hasAllConstantDimensions(memRefOutType)) {
      alloc = create.mem.alignedAlloc(memRefOutType);
    } else {
      SmallVector<Value, 2> allocOperands;
      for (decltype(outRank) i = 0; i < outRank; ++i) {
        if (memRefOutShape[i] == ShapedType::kDynamic) {
          if (dynamicAxes) {
            // Dim size: maskVal[i] ? 1 : inputDim[i]
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

    // There are two required and one optional Krnl loops:
    // - One to initialize the result memref,
    // - One to do reduction, and
    // - One to compute mean (optional).

    // 1. Define loops to initialize the result.
    std::vector<Value> originalLoopsInit;
    defineLoops(rewriter, loc, originalLoopsInit, outRank);

    // Iteration information
    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack packInit(rewriter, originalLoopsInit);
    for (decltype(outRank) i = 0; i < outRank; ++i)
      addDimensionToPack(rewriter, loc, packInit, alloc, i);

    KrnlIterateOp iterateOpInit = create.krnl.iterate(packInit);
    Block &iterationBlockInit = iterateOpInit.getBodyRegion().front();

    // Perform the insertions into the body of the initialization loop.

    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlockInit);

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : iterationBlockInit.getArguments()) {
      loopIVs.push_back(arg);
    }

    Value identity =
        getIdentityValue<ONNXReductionOp>(rewriter, loc, elementOutType);
    create.krnl.store(identity, alloc, loopIVs);

    // 2. Define an Krnl loop to do reduction.
    rewriter.setInsertionPointAfter(iterateOpInit);
    auto ipMainRegion = rewriter.saveInsertionPoint();
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, inRank);
    // Iteration information
    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack pack(rewriter, originalLoops);
    for (decltype(inRank) i = 0; i < inRank; ++i)
      addDimensionToPack(rewriter, loc, pack, input, i);

    KrnlIterateOp iterateOp = create.krnl.iterate(pack);
    Block &iterationBlock = iterateOp.getBodyRegion().front();

    // Perform the insertions into the body of the reduction loop.
    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation:
    SmallVector<Value, 4> inLoopIVs, outLoopIVs;
    auto args = iterationBlock.getArguments();
    for (unsigned int i = 0; i < args.size(); ++i) {
      inLoopIVs.push_back(args[i]);
    }
    // Value zeroIndex = nullptr;
    Value zeroIndex = create.math.constantIndex(0);
    for (decltype(inRank) i = 0; i < outRank; ++i) {
      if (dynamicAxes) {
        // For the reduced dim, the output index is always 0
        Value indexVal = create.math.constantIndex(i);
        Value mask = create.krnl.load(maskVal, indexVal);
        Value cond = create.math.eq(mask, trueVal);
        Value dim = create.math.select(cond, zeroIndex, inLoopIVs[i]);
        outLoopIVs.push_back(dim);
      } else if (outInDimMap.find(i) != outInDimMap.end())
        outLoopIVs.push_back(inLoopIVs[outInDimMap[i]]);
      else
        outLoopIVs.push_back(zeroIndex);
    }

    Value next = create.krnl.load(input, inLoopIVs);
    Value accumulated = create.krnl.load(alloc, outLoopIVs);
    accumulated = emitScalarOpFor<ONNXReductionOp>(
        rewriter, loc, op, memRefOutType.getElementType(), {accumulated, next});
    create.krnl.store(accumulated, alloc, outLoopIVs);

    // 3. Define an Krnl loop to compute mean (optional).
    rewriter.restoreInsertionPoint(ipMainRegion);
    if (computeMean) {
      Type elementType = memRefOutType.getElementType();
      // Compute the divisor that is the number of elements participated in
      // reduction, i.e., 'divisor = size of input / size of output'.
      IndexExprScope scope(&rewriter, loc);
      IndexExpr inputSizeExpr = LiteralIndexExpr(1);
      for (unsigned i = 0; i < inRank; i++) {
        IndexExpr dimExpr = create.krnlIE.getShapeAsDim(input, i);
        inputSizeExpr = inputSizeExpr * dimExpr;
      }
      IndexExpr outputSizeExpr = LiteralIndexExpr(1);
      for (unsigned i = 0; i < outRank; i++) {
        IndexExpr dimExpr = create.krnlIE.getShapeAsDim(alloc, i);
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
      ValueRange loopDef = create.krnl.defineLoops(outRank);
      SmallVector<IndexExpr, 4> lbs(outRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            Value loadData = createKrnl.load(alloc, loopInd);
            Value meanVal = create.math.div(loadData, divisor);
            createKrnl.store(meanVal, alloc, loopInd);
          });
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

// hi alex
#endif
#if 1

// This duplicated code can be eliminated with if constexpr in c++ 17
// Or onnx uses input for axes for all ops
template <typename ONNXReductionOp, RLegacy legacyOp>
struct ONNXReductionOpLoweringNEW
    : public OpConversionPattern<ONNXReductionOp> {
  using OpAdaptor = typename ONNXReductionOp::Adaptor;
  bool computeMean = false;

  ONNXReductionOpLoweringNEW(
      TypeConverter &typeConverter, MLIRContext *ctx, bool computeMean = false)
      : OpConversionPattern<ONNXReductionOp>(typeConverter, ctx),
        computeMean(computeMean) {}

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
        MemRefBuilder>
        create(rewriter, loc);

    DimsExpr rawAxesIE;
    Value axesVal = nullptr;
    bool hasNoAxes = false; // Axles is a None (i.e. optional value not given).
    bool dynamicAxes = false; // No dynamic axes unless proven otherwise.
    if constexpr (legacyOp == RLegacy::Latest) {
      // Deal with operations where we find axes in the inputs.
      axesVal = adaptor.getAxes();
      //axesVal = operands[1];
      axesVal.dump();
      if (isNoneValue(axesVal)) {
        // Default value of having no axes.
        hasNoAxes = true;
      } else {
        // Check it has a rank of 1.
        assert(
            create.krnlIE.getShapedTypeRank(axesVal) == 1 && "expect rank 1");
        IndexExpr axisShape0 = create.krnlIE.getShapeAsDim(axesVal, 0);

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
    // Find out if we have constant axes (make unique and within [0, inRank).
    std::vector<int64_t> uniqueLitAxes;
    if (hasNoAxes) {
      if (isNoop) {
        // No axes and is noop, should we not just return the input array?
      } else {
        // No axes, perform a full reduction.
        for (int64_t i = 0; i < inRank; ++i)
          uniqueLitAxes.push_back(i);
      }
    } else if (!dynamicAxes) {
      // Check raw axes.
      int64_t rawAxesRank = rawAxesIE.size();
      for (int64_t i = 0; i < rawAxesRank; ++i) {
        if (!rawAxesIE[i].isLiteral()) {
          dynamicAxes = true;
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
        }
      }
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

    if (!dynamicAxes) {
      // All axes are static, fill in the outInDimMap appropriately.
      outInDimMap =
          getReductionMapping(memRefInType, uniqueLitAxes, isKeepdims);
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
      assert(axesVal && "expected axesVal");
      IndexExpr axesBound0 = create.krnlIE.getShapeAsDim(axesVal, 0);

      // Initialize mask to 0
      // Unless noop_with_empty_axesDim is false and axesDim is
      // ShapedType::kDynamic.
      Value initVal;
      if (!axesBound0.isLiteral() && !isNoop) {
        // initVal = axes.shape[0] == 0 ? true : false
        IndexExprScope axesLoopContext(&rewriter, loc);
        Value zeroIndex = create.math.constantIndex(0);
        Value cond = create.math.eq(axesBound0.getValue(), zeroIndex);
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
      if (!axesBound0.isLiteral()) {
        // When axes is dynamic, generate a Krnl loop
        KrnlBuilder createKrnl(rewriter, loc);
        ValueRange loopDef = createKrnl.defineLoops(1);
        createKrnl.iterateIE(loopDef, loopDef, {LiteralIndexExpr(0)},
            {axesBound0}, [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
              Value axe = createKrnl.load(axesVal, loopInd[0]);
              Value cond = create.math.slt(axe, zeroValue);
              Value dim = create.math.select(
                  cond, create.math.add(axe, dataDimConst), axe);
              Value jVal = rewriter.create<arith::IndexCastOp>(
                  loc, rewriter.getIndexType(), dim);
              createKrnl.store(trueVal, maskVal, jVal);
            });
      } else {
        int64_t axesDim = axesBound0.getLiteral();
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
            // hi alex: we rely here on the fact that input and output rank is
            // identical. Dim size: maskVal[i] ? 1 : inputDim[i]
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

    // There are two required and one optional Krnl loops:
    // - One to initialize the result memref,
    // - One to do reduction, and
    // - One to compute mean (optional).

    // 1. Define loops to initialize the result.
    std::vector<Value> originalLoopsInit;
    defineLoops(rewriter, loc, originalLoopsInit, outRank);

    // Iteration information
    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack packInit(rewriter, originalLoopsInit);
    for (decltype(outRank) i = 0; i < outRank; ++i)
      addDimensionToPack(rewriter, loc, packInit, alloc, i);

    KrnlIterateOp iterateOpInit = create.krnl.iterate(packInit);
    Block &iterationBlockInit = iterateOpInit.getBodyRegion().front();

    // Perform the insertions into the body of the initialization loop.

    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlockInit);

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : iterationBlockInit.getArguments()) {
      loopIVs.push_back(arg);
    }

    Value identity =
        getIdentityValue<ONNXReductionOp>(rewriter, loc, elementOutType);
    create.krnl.store(identity, alloc, loopIVs);

    // 2. Define an Krnl loop to do reduction.
    rewriter.setInsertionPointAfter(iterateOpInit);
    auto ipMainRegion = rewriter.saveInsertionPoint();
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, inRank);
    // Iteration information
    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack pack(rewriter, originalLoops);
    for (decltype(inRank) i = 0; i < inRank; ++i)
      addDimensionToPack(rewriter, loc, pack, input, i);

    KrnlIterateOp iterateOp = create.krnl.iterate(pack);
    Block &iterationBlock = iterateOp.getBodyRegion().front();

    // Perform the insertions into the body of the reduction loop.
    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation:
    SmallVector<Value, 4> inLoopIVs, outLoopIVs;
    auto args = iterationBlock.getArguments();
    for (unsigned int i = 0; i < args.size(); ++i) {
      inLoopIVs.push_back(args[i]);
    }
    // Value zeroIndex = nullptr;
    Value zeroIndex = create.math.constantIndex(0);
    for (decltype(inRank) i = 0; i < outRank; ++i) {
      if (dynamicAxes) {
        // For the reduced dim, the output index is always 0
        Value indexVal = create.math.constantIndex(i);
        Value mask = create.krnl.load(maskVal, indexVal);
        Value cond = create.math.eq(mask, trueVal);
        Value dim = create.math.select(cond, zeroIndex, inLoopIVs[i]);
        outLoopIVs.push_back(dim);
      } else if (outInDimMap.find(i) != outInDimMap.end())
        outLoopIVs.push_back(inLoopIVs[outInDimMap[i]]);
      else
        outLoopIVs.push_back(zeroIndex);
    }

    Value next = create.krnl.load(input, inLoopIVs);
    Value accumulated = create.krnl.load(alloc, outLoopIVs);
    accumulated = emitScalarOpFor<ONNXReductionOp>(
        rewriter, loc, op, memRefOutType.getElementType(), {accumulated, next});
    create.krnl.store(accumulated, alloc, outLoopIVs);

    // 3. Define an Krnl loop to compute mean (optional).
    rewriter.restoreInsertionPoint(ipMainRegion);
    if (computeMean) {
      Type elementType = memRefOutType.getElementType();
      // Compute the divisor that is the number of elements participated in
      // reduction, i.e., 'divisor = size of input / size of output'.
      IndexExprScope scope(&rewriter, loc);
      IndexExpr inputSizeExpr = LiteralIndexExpr(1);
      for (unsigned i = 0; i < inRank; i++) {
        IndexExpr dimExpr = create.krnlIE.getShapeAsDim(input, i);
        inputSizeExpr = inputSizeExpr * dimExpr;
      }
      IndexExpr outputSizeExpr = LiteralIndexExpr(1);
      for (unsigned i = 0; i < outRank; i++) {
        IndexExpr dimExpr = create.krnlIE.getShapeAsDim(alloc, i);
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
      ValueRange loopDef = create.krnl.defineLoops(outRank);
      SmallVector<IndexExpr, 4> lbs(outRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            Value loadData = createKrnl.load(alloc, loopInd);
            Value meanVal = create.math.div(loadData, divisor);
            createKrnl.store(meanVal, alloc, loopInd);
          });
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

#endif

void populateLoweringONNXReductionOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<
      ONNXReductionOpLoweringNEW<mlir::ONNXReduceMaxV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLoweringNEW<mlir::ONNXReduceMinV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLoweringNEW<mlir::ONNXReduceProdV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLoweringNEW<mlir::ONNXReduceSumV11Op, RLegacy::UpTo13>,
      ONNXReductionOpLoweringNEW<mlir::ONNXReduceMaxOp, RLegacy::Latest>,
      ONNXReductionOpLoweringNEW<mlir::ONNXReduceMinOp, RLegacy::Latest>,
      ONNXReductionOpLoweringNEW<mlir::ONNXReduceProdOp, RLegacy::Latest>,
      ONNXReductionOpLoweringNEW<mlir::ONNXReduceSumOp, RLegacy::Latest>>(
      typeConverter, ctx);
  patterns.insert<
      ONNXReductionOpLoweringNEW<mlir::ONNXReduceMeanV13Op, RLegacy::UpTo13>,
      ONNXReductionOpLoweringNEW<mlir::ONNXReduceMeanOp, RLegacy::Latest>>(
      typeConverter, ctx, /*computeMean=*/true);
}

} // namespace onnx_mlir
