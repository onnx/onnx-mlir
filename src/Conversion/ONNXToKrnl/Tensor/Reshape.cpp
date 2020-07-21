//===---------------- Reshape.cpp - Lowering Reshape Op -------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reshape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXReshapeOpLowering : public ConversionPattern {
  ONNXReshapeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXReshapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXReshapeOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    Value data = operandAdaptor.data();
    auto inputShape = data.getType().cast<MemRefType>().getShape();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto memRefShape = memRefType.getShape();
    Value alloc;

    // Compute size in bytes using the input tensor.
    Value tensorSize = emitConstantOp(rewriter, loc,
        rewriter.getIntegerType(64), getMemRefEltSizeInBytes(memRefType));
    for (int i = 0; i < inputShape.size(); ++i) {
      Value dimVal;
      if (inputShape[i] < 0) {
        Value dim = rewriter.create<DimOp>(loc, data, i);
        dimVal =
            rewriter.create<IndexCastOp>(loc, dim, rewriter.getIntegerType(64));
      } else {
        dimVal = emitConstantOp(
            rewriter, loc, rewriter.getIntegerType(64), inputShape[i]);
      }
      tensorSize = rewriter.create<MulIOp>(loc, tensorSize, dimVal);
    }

    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    } else {
      // If a dimension is zero, the actual dimension value is taken from the
      // input tensor.
      //
      // If the shape array has a negative dimension (-1), we compute its actual
      // dimension value from the other dimensions. But we don't have enough
      // information about the other dimensions at this point. So, we need to
      // scan the shape first to calculate reduction of all of the dimensions.
      // If the reduction is negative, then the shape array contains a negative
      // dimension. Otherwise, the reduction is the same as the one computed
      // from the input tensor.
      Value tensorSizeFromShape = emitConstantOp(rewriter, loc,
          rewriter.getIntegerType(64), getMemRefEltSizeInBytes(memRefType));
      SmallVector<Value, 4> DimInfo;
      for (int i = 0; i < memRefShape.size(); ++i) {
        Value index = emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
        // Load index from array of indices.
        Value loadedVal =
            rewriter.create<AffineLoadOp>(loc, operands[1], index);
        // If a dimension is zero, the actual dimension value is taken from the
        // input tensor.
        //
        // If a dimension is negative, it is computed from the other dimensions.
        // But we don't have enough information about the other dimensions at
        // this point. So, we let it as it is (-1), and compute it later.
        if (i < inputShape.size()) {
          Value dimVal;
          auto loadedValType = loadedVal.getType().cast<IntegerType>();
          if (inputShape[i] < 0) {
            Value dim = rewriter.create<DimOp>(loc, data, i);
            dimVal = rewriter.create<IndexCastOp>(loc, dim, loadedValType);
          } else {
            dimVal =
                emitConstantOp(rewriter, loc, loadedValType, inputShape[i]);
          }
          auto zero = emitConstantOp(rewriter, loc, loadedValType, 0);
          auto isZero =
              rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, loadedVal, zero);
          loadedVal = rewriter.create<SelectOp>(loc, isZero, dimVal, loadedVal);
        }
        // Check if the loaded index is already the correct width of 64 bits.
        // Convert the value to a 64 bit integer if needed.
        Value int64LoadedVal = loadedVal;
        if (loadedVal.getType().cast<IntegerType>().getWidth() < 64)
          int64LoadedVal = rewriter.create<ZeroExtendIOp>(
              loc, loadedVal, rewriter.getIntegerType(64));
        tensorSizeFromShape =
            rewriter.create<MulIOp>(loc, tensorSizeFromShape, int64LoadedVal);
        // Store intermediate results to use later.
        DimInfo.emplace_back(int64LoadedVal);
      }
      // Reverse tensorSizeFromShape since it is negative if the shape array has
      // a negative dimension. This is safe since we only use it to compute the
      // actual value for the negative dimension.
      auto zero = emitConstantOp(rewriter, loc, rewriter.getIntegerType(64), 0);
      tensorSizeFromShape =
          rewriter.create<SubIOp>(loc, zero, tensorSizeFromShape);

      // Obtain operands for AllocOp.
      SmallVector<Value, 4> allocOperands;
      auto negOne =
          emitConstantOp(rewriter, loc, rewriter.getIntegerType(64), -1);

      for (int i = 0; i < memRefShape.size(); ++i) {
        auto dimVal = DimInfo[i];
        auto isNegOne =
            rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, dimVal, negOne);
        // If dimension is negative, compute its value from the other
        // dimensions.
        auto actualDimVal =
            rewriter.create<SignedDivIOp>(loc, tensorSize, tensorSizeFromShape);
        auto loadedVal =
            rewriter.create<SelectOp>(loc, isNegOne, actualDimVal, dimVal);
        allocOperands.push_back(rewriter.create<IndexCastOp>(
            loc, loadedVal, rewriter.getIndexType()));
      }
      AllocOp allocateMemref =
          rewriter.create<AllocOp>(loc, memRefType, allocOperands);

      // Make sure to allocate at the beginning of the block if
      // all dimensions are known.
      auto *parentBlock = allocateMemref.getOperation()->getBlock();
      if (insertDealloc) {
        auto dealloc = rewriter.create<DeallocOp>(loc, allocateMemref);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }

      alloc = allocateMemref;
    }

    rewriter.create<KrnlMemcpyOp>(loc, alloc, data, tensorSize);
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXReshapeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpLowering>(ctx);
}
