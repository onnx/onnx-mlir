//===----- unsqueeze.cpp - Lowering Unsqueeze Op --------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Unsqueeze Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/conversion/onnx_to_krnl/onnx_to_krnl_common.hpp"

using namespace mlir;

struct ONNXUnsqueezeOpLowering : public ConversionPattern {
  ONNXUnsqueezeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXUnsqueezeOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int outRank = memRefType.getRank();

    // Assume that `axes` has been validated by shape inference.
    // So, here we just get it.
    ArrayAttr axisAttrs = llvm::dyn_cast<ONNXUnsqueezeOp>(op).axesAttr();
    SmallVector<int, 4> axes;
    for (auto axisAttr : axisAttrs.getValue()) {
      int axis = axisAttr.cast<IntegerAttr>().getInt();
      axis = axis >= 0 ? axis : (outRank + axis);
      axes.emplace_back(axis);
    }

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;

    // Compute size in bytes.
    Value tensorSize = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                     getMemRefEltSizeInBytes(memRefType)));

    bool insertDealloc = checkInsertDealloc(op);
    auto memRefShape = memRefType.getShape();
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
      for (int i = 0; i < memRefShape.size(); ++i) {
        Value dimVal = rewriter.create<ConstantOp>(
            loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                         memRefShape[i]));
        tensorSize = rewriter.create<MulIOp>(loc, tensorSize, dimVal);
      }
    } else {
      // Unknown dimensions are always the operand's dimensions.
      SmallVector<Value, 4> allocOperands;
      for (int outIdx = 0, inIdx = 0; outIdx < memRefShape.size(); ++outIdx) {
        Value dimVal = nullptr;
        if (memRefShape[outIdx] < 0) {
          Value index = rewriter.create<DimOp>(loc, operands[0], inIdx);
          dimVal = rewriter.create<IndexCastOp>(
              loc, index, rewriter.getIntegerType(64));
          allocOperands.emplace_back(index);
        } else {
          dimVal = rewriter.create<ConstantOp>(
              loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                           memRefShape[outIdx]));
        }
        tensorSize = rewriter.create<MulIOp>(loc, tensorSize, dimVal);
        if (std::find(axes.begin(), axes.end(), outIdx) == axes.end())
          inIdx++;
      }
      alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
      auto *parentBlock = alloc.getDefiningOp()->getBlock();
      if (insertDealloc) {
        auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }
    rewriter.create<KrnlMemcpyOp>(loc, alloc, operands[0], tensorSize);
    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};

void populateLoweringONNXUnsqueezeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXUnsqueezeOpLowering>(ctx);
}
