/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Reduction.cpp - Lowering Reduction Ops ----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reduction Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

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

template <>
Value getIdentityValue<ONNXReduceMeanOp>(
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

template <>
struct ScalarOp<ONNXReduceMeanOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMaxOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXReduceMaxOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
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
    llvm_unreachable("unsupported element type");
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMinOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXReduceMinOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  if (elementType.isa<IntegerType>()) {
    auto min = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, lhs, rhs);
    auto result = rewriter.create<SelectOp>(loc, min, lhs, rhs);
    return result;
  } else if (elementType.isa<FloatType>()) {
    auto min = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
    auto result = rewriter.create<SelectOp>(loc, min, lhs, rhs);
    return result;
  } else {
    llvm_unreachable("unsupported element type");
  }
}

template <typename ONNXReductionOp>
struct ONNXReductionOpLowering : public ConversionPattern {
  bool computeMean = false;

  ONNXReductionOpLowering(MLIRContext *ctx, bool computeMean = false)
      : ConversionPattern(ONNXReductionOp::getOperationName(), 1, ctx) {
    this->computeMean = computeMean;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
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
    auto input = operands[0];
    auto memRefInType = input.getType().cast<MemRefType>();
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
    auto keepdims = llvm::dyn_cast<ONNXReductionOp>(op).keepdims();
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
      alloc =
          insertAllocAndDealloc(memRefOutType, loc, rewriter, insertDealloc);
    } else {
      SmallVector<Value, 2> allocOperands;
      for (decltype(outRank) i = 0; i < outRank; ++i) {
        if (memRefOutShape[i] < 0) {
          auto dim = rewriter.create<memref::DimOp>(loc, input, outInDimMap[i]);
          allocOperands.push_back(dim);
        }
      }
      alloc =
          rewriter.create<memref::AllocOp>(loc, memRefOutType, allocOperands);
      if (insertDealloc) {
        auto *parentBlock = alloc.getDefiningOp()->getBlock();
        auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
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
    KrnlIterateOperandPack packInit(rewriter, originalLoopsInit);
    for (decltype(outRank) i = 0; i < outRank; ++i) {
      addDimensionToPack(rewriter, loc, packInit, alloc, i);
    }
    auto iterateOpInit = rewriter.create<KrnlIterateOp>(loc, packInit);
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
        getIdentityValue<ONNXReductionOp>(rewriter, loc, elementOutType);
    rewriter.create<KrnlStoreOp>(loc, identity, alloc, loopIVs);

    // 2. Define an Krnl loop to do reduction.
    rewriter.setInsertionPointAfter(iterateOpInit);
    auto ipMainRegion = rewriter.saveInsertionPoint();
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, inRank);
    // Iteration information
    KrnlIterateOperandPack pack(rewriter, originalLoops);
    for (decltype(inRank) i = 0; i < inRank; ++i) {
      addDimensionToPack(rewriter, loc, pack, input, i);
    }
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Perform the insertions into the body of the reduction loop.
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
    next = rewriter.create<KrnlLoadOp>(loc, input, inLoopIVs);
    accumulated = rewriter.create<KrnlLoadOp>(loc, alloc, outLoopIVs);
    accumulated = emitScalarOpFor<ONNXReductionOp>(
        rewriter, loc, op, memRefOutType.getElementType(), {accumulated, next});
    rewriter.create<KrnlStoreOp>(loc, accumulated, alloc, outLoopIVs);

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
        divisor = rewriter.create<IndexCastOp>(
            loc, divisor, rewriter.getIntegerType(64));
        divisor = rewriter.create<UIToFPOp>(loc, elementType, divisor);
      } else if (elementType.isa<IntegerType>()) {
        divisor = rewriter.create<IndexCastOp>(loc, divisor, elementType);
      } else
        llvm_unreachable("unsupported element type");

      // Compute mean
      BuildKrnlLoop meanLoops(rewriter, loc, outRank);
      meanLoops.createDefineAndIterateOp(alloc);
      rewriter.setInsertionPointToStart(meanLoops.getIterateBlock());
      auto meanIVs = meanLoops.getAllInductionVar();
      auto loadData = rewriter.create<KrnlLoadOp>(loc, alloc, meanIVs);
      Value meanVal;
      if (elementType.isa<FloatType>())
        meanVal = rewriter.create<DivFOp>(loc, loadData, divisor);
      else if (elementType.isa<IntegerType>())
        meanVal = rewriter.create<SignedDivIOp>(loc, loadData, divisor);
      else
        llvm_unreachable("unsupported element type");
      rewriter.create<KrnlStoreOp>(loc, meanVal, alloc, meanIVs);
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXReductionOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReductionOpLowering<mlir::ONNXReduceMaxOp>,
      ONNXReductionOpLowering<mlir::ONNXReduceMinOp>,
      ONNXReductionOpLowering<mlir::ONNXReduceProdOp>,
      ONNXReductionOpLowering<mlir::ONNXReduceSumOp>>(ctx);
  patterns.insert<ONNXReductionOpLowering<mlir::ONNXReduceMeanOp>>(
      ctx, /*computeMean=*/true);
}
