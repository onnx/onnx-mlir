/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
    ONNXReshapeOp reshapeOp = dyn_cast_or_null<ONNXReshapeOp>(op);

    auto loc = op->getLoc();
    Value data = operandAdaptor.data();
    Value shape = operandAdaptor.shape();
    auto dataShape = data.getType().cast<MemRefType>().getShape();
    // If shape input was promoted to attribute, get its values from the
    // attribute.
    SmallVector<int64_t, 4> shapeAttrValues;
    DenseElementsAttr shapeAttr =
        reshapeOp.getAttr("shape").dyn_cast_or_null<DenseElementsAttr>();
    if (shapeAttr) {
      auto shapeAttrIt = shapeAttr.getValues<IntegerAttr>().begin();
      auto itEnd = shapeAttr.getValues<IntegerAttr>().end();
      for (; shapeAttrIt != itEnd;)
        shapeAttrValues.emplace_back(
            (*shapeAttrIt++).cast<IntegerAttr>().getInt());
    }

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto memRefShape = memRefType.getShape();
    Value alloc;

    // Compute size in bytes using the input tensor.
    Value tensorSizeFromInput = emitConstantOp(rewriter, loc,
        rewriter.getIntegerType(64), getMemRefEltSizeInBytes(memRefType));
    for (int i = 0; i < dataShape.size(); ++i) {
      Value dimVal =
          getDimOrConstant(rewriter, loc, data, i, rewriter.getIntegerType(64));
      tensorSizeFromInput =
          rewriter.create<MulIOp>(loc, tensorSizeFromInput, dimVal);
    }

    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    } else {
      // Calculate the unknown output dimensions using given shape information.
      // Shape information is store in the shape input. However, the shape input
      // will be promoted to attribute if it is a constant. Thus, we need to
      // check both cases.
      //
      // Dimensions in the shape information can be: 0, -1, or N.
      // - If a dimension is 0, the actual dimension value is taken from the
      // input tensor.
      //
      // - If a dimension is -1, the actual dimension value is computed by
      // 'tensorSizeFromInput/tensorSizeFromShape'
      //
      // - If a dimension is N, kept it unchanged.

      Value tensorSizeFromShape = emitConstantOp(rewriter, loc,
          rewriter.getIntegerType(64), getMemRefEltSizeInBytes(memRefType));

      SmallVector<Value, 4> outputDimInfo;
      for (int i = 0; i < memRefShape.size(); ++i) {
        Value outputDimVal;
        if (!shapeAttrValues.empty()) {
          // Compute the output dimension using shape attribute.
          if (shapeAttrValues[i] == 0)
            // Dimension is 0, get its actual value from the input tensor.
            outputDimVal = getDimOrConstant(
                rewriter, loc, data, i, rewriter.getIntegerType(64));
          else
            // If dimension is -1, compute it later.
            // If dimension is N (N != 0 && N != -1), use it.
            // In both cases, just kept it unchanged.
            outputDimVal = emitConstantOp(
                rewriter, loc, rewriter.getIntegerType(64), shapeAttrValues[i]);
        } else {
          // Compute the output dimension using shape tensor.
          Value index =
              emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
          Value loadedVal = rewriter.create<KrnlLoadOp>(loc, shape, index);

          // If dimension is 0, get its actual value from the input tensor.
          // Otherwise, kept it unchanged.
          if (i >= dataShape.size())
            // Dimension cannot be 0 if its position is out-of-bound, e.g. the
            // output rank is greater than the input rank.
            // No need to check 0.
            outputDimVal = loadedVal;
          else {
            // Dimension is potentially 0, check it.
            auto dimVal = getDimOrConstant(
                rewriter, loc, data, i, rewriter.getIntegerType(64));
            auto zero =
                emitConstantOp(rewriter, loc, rewriter.getIntegerType(64), 0);
            auto isZero = rewriter.create<CmpIOp>(
                loc, CmpIPredicate::eq, loadedVal, zero);
            outputDimVal =
                rewriter.create<SelectOp>(loc, isZero, dimVal, loadedVal);
          }
        }

        // Compute tensor size using shape information.
        tensorSizeFromShape =
            rewriter.create<MulIOp>(loc, tensorSizeFromShape, outputDimVal);
        // Store intermediate results to use later.
        outputDimInfo.emplace_back(outputDimVal);
      }

      // tensorSizeFromShape is negative if a dimension is -1. So make it
      // positive by multipling by -1.
      auto negOne =
          emitConstantOp(rewriter, loc, rewriter.getIntegerType(64), -1);
      tensorSizeFromShape =
          rewriter.create<MulIOp>(loc, tensorSizeFromShape, negOne);

      // Obtain operands for AllocOp.
      SmallVector<Value, 4> allocOperands;
      for (int i = 0; i < memRefShape.size(); ++i) {
        if (memRefShape[i] != -1)
          continue;
        auto dimVal = outputDimInfo[i];
        auto isNegOne =
            rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, dimVal, negOne);
        // If dimension is -1, compute its value by
        // 'tensorSizeFromInput/tensorSizeFromShape'
        auto unknownDimVal = rewriter.create<SignedDivIOp>(
            loc, tensorSizeFromInput, tensorSizeFromShape);
        auto actualDimVal =
            rewriter.create<SelectOp>(loc, isNegOne, unknownDimVal, dimVal);
        allocOperands.push_back(rewriter.create<IndexCastOp>(
            loc, actualDimVal, rewriter.getIndexType()));
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

    rewriter.create<KrnlMemcpyOp>(loc, alloc, data, tensorSizeFromInput);
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXReshapeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpLowering>(ctx);
}
