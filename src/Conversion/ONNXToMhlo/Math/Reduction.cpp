/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Reduction.cpp - Lowering Reduction Ops ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reduction Operators to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// Identity values
template <>
Value getIdentityValue<ONNXReduceMaxOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
	Type elementType = type.getElementType().cast<FloatType>();
  return rewriter.create<mhlo::ConstOp>(loc, rewriter.getFloatAttr(
                 elementType, APFloat::getInf(elementType.getFloatSemantics(),/*isNegative=*/true)));
}

template <>
Value getIdentityValue<ONNXReduceMinOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
	Type elementType = type.getElementType().cast<FloatType>();
  return rewriter.create<mhlo::ConstOp>(loc, rewriter.getFloatAttr(
                 elementType, APFloat::getInf(elementType.getFloatSemantics(),/*isNegative=*/false)));
}

template <>
Value getIdentityValue<ONNXReduceSumOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
	return rewriter.create<mhlo::ConstOp>(loc, rewriter.getZeroAttr(type));
}

template <>
Value getIdentityValue<ONNXReduceMeanOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return rewriter.create<mhlo::ConstOp>(loc, rewriter.getZeroAttr(type));
}

void getReductionShape(Type inputType, axes, bool isKeepdims, SmallVector<int64_t>& reduceShape);
SmallVector<int64_t> reduceShape;

// Create "mhlo.reduce", "operand" is reduce input and "zero" is init value,
// reduce from operand to operand[reduceIdx].
template <typename ReduceOp>
Value createReduce(Location loc, Value operand, Value zero,
    SmallVector<int64_t> &reduceShape, int64_t reduceIdx,
    PatternRewriter &rewriter) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  Type reduceResultType =
      RankedTensorType::get(reduceShape, operandType.getElementType());
  mhlo::ReduceOp reduce = rewriter.create<mhlo::ReduceOp>(loc, reduceResultType,
      operand, zero, rewriter.getI64TensorAttr({reduceIdx}));

  // setup "mhlo.reduce"'s body
  Region &region = reduce.body();
  Block &block = region.emplaceBlock();
  RankedTensorType blockArgumentType =
      RankedTensorType::get({}, operandType.getElementType());
  block.addArgument(blockArgumentType, loc);
  block.addArgument(blockArgumentType, loc);
  auto firstArgument = *block.args_begin();
  auto secondArgument = *block.args_rbegin();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value reduceResult =
        rewriter.create<ReduceOp>(loc, firstArgument, secondArgument);
    rewriter.create<mhlo::ReturnOp>(loc, reduceResult);
  }
  return reduce.getResult(0);
}



template <typename ONNXReductionOp>
struct ONNXReductionOpLowering : public ConversionPattern {
  bool computeMean = false;

  ONNXReductionOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool computeMean = false)
      : ConversionPattern(
            typeConverter, ONNXReductionOp::getOperationName(), 1, ctx) {
    this->computeMean = computeMean;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = operands[0];
		// Type
		Type inputType = input.getType().cast<RankedTensorType>();
    if (inputType == nullptr) {
      return failure();
    }
		Type outputType = *op->result_type_begin().cast<RankedTensorType>();
		if (outputType == nullptr) {
			return failure();
		}
		Value zero = getIdentityValue<ONNXReductionOp>(rewriter, loc, outputType);
    int64_t inRank = inputType.getRank();
    // int64_t outRank = outputType.getRank();



    // Get axes value defined by op
    // Leave empty is not defined
    std::vector<int64_t> definedAxes;
    ArrayAttr axisAttrs = llvm::dyn_cast<ONNXReductionOp>(op).axesAttr();
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
        int64_t newaxis = axis >= 0 ? axis : (inRank + axis);
        if (std::find(axes.begin(), axes.end(), newaxis) == axes.end())
          axes.push_back(newaxis);
      }
    } else
      for (decltype(inRank) i = 0; i < inRank; ++i)
        axes.push_back(i);

    // KeepDims
    auto keepdims = llvm::dyn_cast<ONNXReductionOp>(op).keepdims();
    bool isKeepdims = (keepdims == 1) ? true : false;

    // Get type information
    auto outShape = outputType.getShape();
    auto elementOutType = outputType.getElementType();
    SmallVector<int64_t> reducedShape =
        getReductionShape(inputType, axes, isKeepdims);

		
  }
};

struct ONNXReduceSumOpLowering : public ConversionPattern {
  bool computeMean = false;

  ONNXReduceSumOpLowering(MLIRContext *ctx, bool computeMean = false)
      : ConversionPattern(ONNXReduceSumOp::getOperationName(), 1, ctx),
        computeMean(computeMean) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    
    Location loc = op->getLoc();
    ONNXReduceSumOpAdaptor operandAdaptor(operands);
    ONNXReduceSumOp reduceSumOp = cast<ONNXReduceSumOp>(op);
    auto input = operands[0];
    auto axesVal = operands[1];
		Type inputType = input.getType().dyn_cast<RankedTensorType>();
		if (inputType == nullptr) {
			return failure();
		}
		Type outputType = *op->result_type_begin();
		int64_t inRank = inputType.getRank();
		int64_t outRank = outputType.getRank();

    // KeepDims
    int64_t keepdims = reduceSumOp.keepdims();
    bool isKeepdims = (keepdims == 1);

    ONNXReduceSumOpShapeHelper shapeHelper(&reduceSumOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // Get type information
    auto memRefOutShape = memRefOutType.getShape();
    auto elementOutType = memRefOutType.getElementType();

    bool dynamicAxes = false;
    Value maskVal = nullptr;
    Value falseVal = nullptr;
    Value trueVal = nullptr;
    Value valueOne = nullptr;
    std::map<int64_t, int64_t> outInDimMap;

    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);

    Value axesValue = reduceSumOp.axes();
    // Dynamic axes
    if (!isFromNone(axesValue) && !getONNXConstantOp(axesValue)) {
      dynamicAxes = true;
      // Handle only when keepdims == true
      if (!isKeepdims)
        return emitError(loc, "not keepdims() not implemented");

      // Define a mask memref with same size of input and bool type
      // maskVal[i] == true if ith dim will be reduced
      bool insertDealloc = checkInsertDealloc(op);
      auto maskType =
          RankedTensorType::get({inRank}, rewriter.getIntegerType(1));
      // Convert the mask type to MemRefType.
      Type convertedMaskType = typeConverter->convertType(maskType);
      assert(convertedMaskType && convertedMaskType.isa<MemRefType>() &&
             "Failed to convert type to MemRefType");
      MemRefType maskTypeInMemRefType = convertedMaskType.cast<MemRefType>();
      maskVal = insertAllocAndDealloc(
          maskTypeInMemRefType, loc, rewriter, insertDealloc);
      falseVal = create.math.constant(rewriter.getIntegerType(1), 0);
      trueVal = create.math.constant(rewriter.getIntegerType(1), 1);
      valueOne = create.math.constantIndex(1);
      auto axesDim = axesVal.getType().cast<MemRefType>().getShape()[0];

      // Initialize mask to 0
      // Unless noop_with_empty_axesDim is false and axesDim is -1
      Value initVal;
      if (axesDim == -1 && !reduceSumOp.noop_with_empty_axes()) {
        IndexExprScope axesloopContex(&rewriter, loc);
        MemRefBoundsIndexCapture axesBounds(axesVal);
        Value zeroIndex = create.math.constantIndex(0);
        Value cond = create.math.eq(axesBounds.getDim(0).getValue(), zeroIndex);
        initVal = create.math.select(cond, trueVal, falseVal);
      } else {
        // When axesDim is known, it can not be 0 due to !isFromNone
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
      if (axesDim == -1) {
        // When axes is dynamic, generate a Krnl loop
        KrnlBuilder createKrnl(rewriter, loc);
        ValueRange loopDef = createKrnl.defineLoops(1);
        SmallVector<IndexExpr, 4> lbs(1, LiteralIndexExpr(0));
        createKrnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.dimsForOutput(),
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
      if (!isFromNone(axesValue) && getONNXConstantOp(axesValue)) {
        auto constAxes = getONNXConstantOp(axesValue)
                             .valueAttr()
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
          int64_t newaxis = axis >= 0 ? axis : (inRank + axis);
          if (std::find(axes.begin(), axes.end(), newaxis) == axes.end())
            axes.push_back(newaxis);
        }
      } else if (!reduceSumOp.noop_with_empty_axes()) {
        for (decltype(inRank) i = 0; i < inRank; ++i) {
          axes.push_back(i);
        }
      }
      outInDimMap = getReductionMapping(memRefInType, axes, isKeepdims);
    }

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefOutType)) {
      alloc =
          insertAllocAndDealloc(memRefOutType, loc, rewriter, insertDealloc);
    } else {
      SmallVector<Value, 2> allocOperands;
      for (decltype(outRank) i = 0; i < outRank; ++i) {
        if (memRefOutShape[i] < 0) {
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
      if (insertDealloc) {
        Block *parentBlock = alloc.getDefiningOp()->getBlock();
        auto dealloc = create.mem.dealloc(alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
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
    Block &iterationBlockInit = iterateOpInit.bodyRegion().front();

    // Perform the insertions into the body of the initialization loop.

    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlockInit);

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : iterationBlockInit.getArguments()) {
      loopIVs.push_back(arg);
    }

    Value identity =
        getIdentityValue<ONNXReduceSumOp>(rewriter, loc, elementOutType);
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
    Block &iterationBlock = iterateOp.bodyRegion().front();

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
    accumulated = emitScalarOpFor<ONNXReduceSumOp>(
        rewriter, loc, op, memRefOutType.getElementType(), {accumulated, next});
    create.krnl.store(accumulated, alloc, outLoopIVs);

    // 3. Define an Krnl loop to compute mean (optional).
    rewriter.restoreInsertionPoint(ipMainRegion);
    MemRefBoundsIndexCapture inputBounds(input);
    MemRefBoundsIndexCapture allocBounds(alloc);
    if (computeMean) {
      Type elementType = memRefOutType.getElementType();
      // Compute the divisor that is the number of elements participated in
      // reduction, i.e., 'divisor = size of input / size of output'.
      IndexExprScope scope(&rewriter, loc);
      IndexExpr inputSizeExpr = LiteralIndexExpr(1);
      for (unsigned i = 0; i < inRank; i++) {
        DimIndexExpr dimExpr(inputBounds.getDim(i));
        inputSizeExpr = inputSizeExpr * dimExpr;
      }
      IndexExpr outputSizeExpr = LiteralIndexExpr(1);
      for (unsigned i = 0; i < outRank; i++) {
        DimIndexExpr dimExpr(allocBounds.getDim(i));
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
      KrnlBuilder createKrnl(rewriter, loc);
      ValueRange loopDef = createKrnl.defineLoops(outRank);
      SmallVector<IndexExpr, 4> lbs(outRank, LiteralIndexExpr(0));
      MemRefBoundsIndexCapture allocBounds(alloc);
      SmallVector<IndexExpr, 4> ubs;
      allocBounds.getDimList(ubs);
      createKrnl.iterateIE(loopDef, loopDef, lbs, ubs,
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


void populateLoweringONNXReductionOpPattern(RewritePatternSet &patterns,
    MLIRContext *ctx) {
  patterns.insert<ONNXReductionOpLowering<mlir::ONNXReduceMaxOp>,
      ONNXReductionOpLowering<mlir::ONNXReduceMinOp>,
      ONNXReductionOpLowering<mlir::ONNXReduceSumV11Op>,
			ONNXReduceSumOpLowering>(ctx);
  patterns.insert<ONNXReductionOpLowering<mlir::ONNXReduceMeanOp>>(
      ctx, /*computeMean=*/true);
}

} // namespace onnx_mlir
