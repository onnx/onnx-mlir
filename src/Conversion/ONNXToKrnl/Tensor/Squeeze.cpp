/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
    ONNXSqueezeOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto memRefShape = memRefType.getShape();
    auto elementSizeInBytes = getMemRefEltSizeInBytes(memRefType);
    Value data = operandAdaptor.data();

    // Assume that `axes` has been validated by shape inference.
    // So, here we just get it.
    ArrayAttr axisAttrs = llvm::dyn_cast<ONNXSqueezeOp>(op).axesAttr();
    SmallVector<int, 4> axes;
    for (auto axisAttr : axisAttrs.getValue()) {
      int axis = axisAttr.cast<IntegerAttr>().getInt();
      axes.emplace_back(axis);
    }

    // Insert an allocation and deallocation for the result of this operation,
    // and compute the output tensor's size in bytes.
    Value alloc, tensorSize;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
      auto tensorSizeInBytes = elementSizeInBytes;
      for (int i = 0; i < memRefShape.size(); ++i) {
        tensorSizeInBytes *= memRefShape[i];
      }
      tensorSize = emitConstantOp(
          rewriter, loc, rewriter.getIntegerType(64), tensorSizeInBytes);
    } else {
      // Need to know the input dimension from which the unknown output
      // dimension comes from.
      SmallVector<Value, 4> allocOperands;
      auto tensorSizeConstant = elementSizeInBytes;
      int64_t inRank = data.getType().cast<ShapedType>().getRank();
      for (decltype(inRank) inIdx = 0, outIdx = 0; inIdx < inRank; ++inIdx) {
        Value dimVal = nullptr;
        // Squeeze dimension is not in the output, ignore it.
        if (std::find(axes.begin(), axes.end(), inIdx) != axes.end())
          continue;
        // Found effective input dimension.
        if (memRefShape[outIdx] < 0) {
          Value index = rewriter.create<DimOp>(loc, data, inIdx);
          allocOperands.emplace_back(index);
        } else {
          // Collect constant dimensions for calculating the output tensor size.
          tensorSizeConstant *= memRefShape[outIdx];
        }
        // Move to the next output dimension.
        outIdx++;
      }
      // Allocate memory.
      alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
      auto *parentBlock = alloc.getDefiningOp()->getBlock();
      if (insertDealloc) {
        auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }

      // Compute the output tensor's size.
      tensorSize = emitConstantOp(
          rewriter, loc, rewriter.getIntegerType(64), tensorSizeConstant);
      for (Value dim : allocOperands) {
        Value dimVal =
            rewriter.create<IndexCastOp>(loc, dim, rewriter.getIntegerType(64));
        tensorSize = rewriter.create<MulIOp>(loc, tensorSize, dimVal);
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
