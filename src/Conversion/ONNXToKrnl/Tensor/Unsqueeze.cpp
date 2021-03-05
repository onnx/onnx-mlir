/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Unsqueeze.cpp - Lowering Unsqueeze Op ----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Unsqueeze Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXUnsqueezeOpLowering : public ConversionPattern {
  ONNXUnsqueezeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXUnsqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXUnsqueezeOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int outRank = memRefType.getRank();
    Value data = operandAdaptor.data();

    // Assume that `axes` has been validated by shape inference.
    // So, here we just get it.
    SmallVector<int, 4> axes;
    if (auto axesOp = llvm::dyn_cast_or_null<KrnlGlobalOp>(
            operandAdaptor.axes().getDefiningOp())) {
      auto axesAttr = axesOp.value()->dyn_cast<DenseElementsAttr>();
      if (!axesAttr) {
        return op->emitError("Axes expected to be a DenseElementAttr");
      }

      for (auto attr : axesAttr.getValues<IntegerAttr>()) {
        int axis = attr.getInt();
        axis = axis >= 0 ? axis : (outRank + axis);
        axes.emplace_back(axis);
      }
    } else {
      return op->emitError("Axes argument expected to be a constant");
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
      // Unknown dimensions are always the operand's dimensions.
      SmallVector<Value, 4> allocOperands;
      for (int outIdx = 0, inIdx = 0; outIdx < memRefShape.size(); ++outIdx) {
        Value dimVal = nullptr;
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
    rewriter.create<KrnlMemcpyOp>(loc, alloc, data, tensorSize);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXUnsqueezeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXUnsqueezeOpLowering>(ctx);
}
