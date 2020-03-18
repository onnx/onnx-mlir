//===----- reduction.cpp - Lowering Reduction Ops -------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reduction Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/onnx_to_krnl/onnx_to_krnl_common.hpp"

using namespace mlir;

// Identity values
template <>
Value getIdentityValue<ONNXReduceMaxOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return emitNegativeInfinityConstantOp(rewriter, loc, type);
}

template <>
Value getIdentityValue<ONNXReduceMinOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return emitPositiveInfinityConstantOp(rewriter, loc, type);
}

template <>
Value getIdentityValue<ONNXReduceProdOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return emitConstantOp(rewriter, loc, type, 1);
}

template <>
Value getIdentityValue<ONNXReduceSumOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return emitConstantOp(rewriter, loc, type, 0);
}

// Scalar ops
template <>
struct ScalarOp<ONNXReduceProdOp> {
  using FOp = MulFOp;
  using IOp = MulIOp;
};

template <>
struct ScalarOp<ONNXReduceSumOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMaxOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXReduceMaxOp>(Operation *op,
                                          ArrayRef<Type> result_types,
                                          ArrayRef<Value> operands,
                                          ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  Value lhs = operands[0];
  Value rhs = operands[1];
  Type element_type = lhs.getType();
  if (element_type.isa<IntegerType>()) {
    auto max = rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, lhs, rhs);
    auto result = rewriter.create<SelectOp>(loc, max, lhs, rhs);
    return result;
  } else if (element_type.isa<FloatType>()) {
    auto max = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
    auto result = rewriter.create<SelectOp>(loc, max, lhs, rhs);
    return result;
  } else {
    emitError(loc, "unsupported element type");
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMinOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXReduceMinOp>(Operation *op,
                                          ArrayRef<Type> result_types,
                                          ArrayRef<Value> operands,
                                          ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  Value lhs = operands[0];
  Value rhs = operands[1];
  Type element_type = lhs.getType();
  if (element_type.isa<IntegerType>()) {
    auto min = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, lhs, rhs);
    auto result = rewriter.create<SelectOp>(loc, min, lhs, rhs);
    return result;
  } else if (element_type.isa<FloatType>()) {
    auto min = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
    auto result = rewriter.create<SelectOp>(loc, min, lhs, rhs);
    return result;
  } else {
    emitError(loc, "unsupported element type");
    return nullptr;
  }
}

template <typename ONNXReductionOp>
struct ONNXReductionOpLowering : public ConversionPattern {
  ONNXReductionOpLowering(MLIRContext *ctx)
      : ConversionPattern(ONNXReductionOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
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
    auto loc = op->getLoc();
    auto memRefInType = operands[0].getType().cast<MemRefType>();
    auto memRefInShape = memRefInType.getShape();
    auto memRefOutType = convertToMemRefType(*op->result_type_begin());
    int64_t inRank = memRefInType.getRank();
    int64_t outRank = memRefOutType.getRank();

    // Get attributes
    ArrayAttr axisAttrs = llvm::dyn_cast<ONNXReductionOp>(op).axesAttr();
    std::vector<int64_t> axes;
    if (axisAttrs) {
      for (auto axisAttr : axisAttrs.getValue()) {
        int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
        axis = axis >= 0 ? axis : (inRank + axis);
        assert(axis >= -inRank && axis <= inRank - 1);
        if (std::find(axes.begin(), axes.end(), axis) == axes.end())
          axes.push_back(axis);
      }
    } else {
      for (decltype(inRank) i = 0; i < inRank; ++i) {
        axes.push_back(i);
      }
    }
    // KeepDims
    auto keepdims =
        llvm::dyn_cast<ONNXReductionOp>(op).keepdims();
    bool isKeepdims = (keepdims == 1) ? true : false;

    // Get type information
    auto memRefOutShape = memRefOutType.getShape();
    auto elementOutType = memRefOutType.getElementType();
    std::map<int64_t, int64_t> outInDimMap =
        getReductionMapping(memRefInType, axes, isKeepdims);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefOutType)) {
      alloc = insertAllocAndDealloc(memRefOutType, loc, rewriter, insertDealloc);
    } else {
      SmallVector<Value, 2> allocOperands;
      for (decltype(outRank) i = 0; i < outRank; ++i) {
        if (memRefOutShape[i] < 0) {
          auto dim = rewriter.create<DimOp>(loc, operands[0], outInDimMap[i]);
          allocOperands.push_back(dim);
        }
      }
      alloc = rewriter.create<AllocOp>(loc, memRefOutType, allocOperands);
      if (insertDealloc) {
        auto *parentBlock = alloc.getDefiningOp()->getBlock();
        auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }

    // There are two Krnl loops:
    // - One to initialize the result memref, and
    // - One to do reduction

    // Define loops to initialize the result.
    std::vector<Value> originalLoopsInit;
    std::vector<Value> optimizedLoopsInit;
    Block *optimizationBlockInit = defineLoops(rewriter, loc, originalLoopsInit,
            optimizedLoopsInit, outRank);

    // Iteration information
    KrnlIterateOperandPack packInit(rewriter, originalLoopsInit,
        optimizedLoopsInit);
    for (decltype(outRank) i = 0; i < outRank; ++i) {
      addDimensionToPack(rewriter, loc, packInit, alloc, i);
    }
    auto iterateOpInit = rewriter.create<KrnlIterateOp>(loc, packInit);
    Block &iterationBlockInit = iterateOpInit.bodyRegion().front();

    // Perform the insertions into the body of the initialization loop.
    // No optimization
    rewriter.setInsertionPointToEnd(optimizationBlockInit);
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoopsInit);

    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlockInit);

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : iterationBlockInit.getArguments()) {
      loopIVs.push_back(arg);
    }

    Value identity =
        getIdentityValue<ONNXReductionOp>(rewriter, loc, elementOutType);
    rewriter.create<StoreOp>(loc, identity, alloc, loopIVs);

    // Define an Krnl loop to do reduction.
    rewriter.setInsertionPointAfter(iterateOpInit);
    std::vector<Value> originalLoops, optimizedLoops;
    Block *optimizationBlock = defineLoops(rewriter, loc, originalLoops,
            optimizedLoops, inRank);
    // Iteration information
    KrnlIterateOperandPack pack(rewriter, originalLoops, optimizedLoops);
    for (decltype(inRank) i = 0; i < inRank; ++i) {
      addDimensionToPack(rewriter, loc, pack, operands[0], i);
    }
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Perform the insertions into the body of the reduction loop.
    // No optimization
    rewriter.setInsertionPointToEnd(optimizationBlock);
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation:
    SmallVector<Value, 4> inLoopIVs, outLoopIVs;
    auto args = iterationBlock.getArguments();
    for (int i = 0; i < args.size(); ++i) {
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
          zeroIndex = rewriter.create<ConstantIndexOp>(loc, 0);
          outLoopIVs.push_back(zeroIndex);
        }
      }
    }

    Value next, accumulated;
    next = rewriter.create<LoadOp>(loc, operands[0], inLoopIVs);
    accumulated = rewriter.create<LoadOp>(loc, alloc, outLoopIVs);
    accumulated = mapToLowerScalarOp<ONNXReductionOp>(
        op, memRefOutType.getElementType(), {accumulated, next}, rewriter);
    rewriter.create<StoreOp>(loc, accumulated, alloc, outLoopIVs);

    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};

void populateLoweringONNXReductionOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReductionOpLowering<mlir::ONNXReduceMaxOp>,
                  ONNXReductionOpLowering<mlir::ONNXReduceMinOp>,
                  ONNXReductionOpLowering<mlir::ONNXReduceProdOp>,
                  ONNXReductionOpLowering<mlir::ONNXReduceSumOp>>(ctx);
}
