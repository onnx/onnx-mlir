//===--------------- Squeeze.cpp - Lowering Squeeze Op --------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Squeeze Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXSqueezeOpLowering : public ConversionPattern {
  ONNXSqueezeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeOpOperandAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int outRank = memRefType.getRank();
    Value data = operandAdaptor.data();

    // Assume that `axes` has been validated by shape inference.
    // So, here we just get it.
    ArrayAttr axisAttrs = llvm::dyn_cast<ONNXSqueezeOp>(op).axesAttr();
    SmallVector<int, 4> axes;
    for (auto axisAttr : axisAttrs.getValue()) {
      int axis = axisAttr.cast<IntegerAttr>().getInt();
      axes.emplace_back(axis);
    }

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;

    // Compute size in bytes.
    Value tensorSize = emitConstantOp(rewriter, loc,
        rewriter.getIntegerType(64), getMemRefEltSizeInBytes(memRefType));

    bool insertDealloc = checkInsertDealloc(op);
    auto memRefShape = memRefType.getShape();
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
      for (int i = 0; i < memRefShape.size(); ++i) {
        Value dimVal = emitConstantOp(
            rewriter, loc, rewriter.getIntegerType(64), memRefShape[i]);
        tensorSize = rewriter.create<MulIOp>(loc, tensorSize, dimVal);
      }
    } else {
      // Need to know the input dimension from which the unknown output
      // dimension comes from.
      SmallVector<Value, 4> allocOperands;
      for (int inIdx = 0, outIdx = 0; inIdx < memRefShape.size(); ++inIdx) {
        Value dimVal = nullptr;
        // Squeeze dimension is not in the output, ignore it.
        if (std::find(axes.begin(), axes.end(), inIdx) != axes.end())
          continue;
        // Found effective input dimension.
        if (memRefShape[outIdx] < 0) {
          Value index = rewriter.create<DimOp>(loc, data, inIdx);
          dimVal = rewriter.create<IndexCastOp>(
              loc, index, rewriter.getIntegerType(64));
          allocOperands.emplace_back(index);
        } else {
          dimVal = emitConstantOp(
              rewriter, loc, rewriter.getIntegerType(64), memRefShape[outIdx]);
        }
        tensorSize = rewriter.create<MulIOp>(loc, tensorSize, dimVal);
        // Move to the next output dimension.
        outIdx++;
      }
      alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
      auto *parentBlock = alloc.getDefiningOp()->getBlock();
      if (insertDealloc) {
        auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }
    rewriter.create<KrnlMemcpyOp>(loc, alloc, data, tensorSize);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSqueezeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeOpLowering>(ctx);
}
