/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Clip.cpp - Lowering Clip Op ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Clip Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXClipOp
//===----------------------------------------------------------------------===//

struct ONNXClipOpLowering : public ConversionPattern {
  ONNXClipOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXClipOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = operands[0];
    Value min = operands[1];
    Value max = operands[2];

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, input);

    SmallVector<Value, 4> loopIVs;
    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      // Create iterateOp & get block within iterate op.
      BuildKrnlLoop loops(rewriter, loc, memRefType.getRank());
      loops.createDefineAndIterateOp(input);
      Block *iterationBlock = loops.getIterateBlock();

      // Insert instructions inside the KernelIterateOp body.
      rewriter.setInsertionPointToStart(iterationBlock);

      // Handle the operation:
      for (auto arg : iterationBlock->getArguments())
        loopIVs.push_back(arg);
    }

    // Load unary first operand.
    Value loadedVal = rewriter.create<KrnlLoadOp>(loc, input, loopIVs);
    Type inputType = loadedVal.getType();
    Value res = loadedVal;
    if (inputType.isa<FloatType>()) {
      if (!min.getType().isa<NoneType>()) {
        Value minVal = rewriter.create<KrnlLoadOp>(loc, min).getResult();
        Value lessThanMin = rewriter.create<arith::CmpFOp>(
            loc, arith::CmpFPredicate::OLT, res, minVal);
        res = rewriter.create<arith::SelectOp>(loc, lessThanMin, minVal, res);
      }
      if (!max.getType().isa<NoneType>()) {
        Value maxVal = rewriter.create<KrnlLoadOp>(loc, max).getResult();
        Value lessThanMax = rewriter.create<arith::CmpFOp>(
            loc, arith::CmpFPredicate::OLT, res, maxVal);
        res = rewriter.create<arith::SelectOp>(loc, lessThanMax, res, maxVal);
      }
    } else if (inputType.isa<IntegerType>()) {
      if (!min.getType().isa<NoneType>()) {
        Value minVal = rewriter.create<KrnlLoadOp>(loc, min).getResult();
        Value lessThanMin = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, res, minVal);
        res = rewriter.create<arith::SelectOp>(loc, lessThanMin, minVal, res);
      }
      if (!max.getType().isa<NoneType>()) {
        Value maxVal = rewriter.create<KrnlLoadOp>(loc, max).getResult();
        Value lessThanMax = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, res, maxVal);
        res = rewriter.create<arith::SelectOp>(loc, lessThanMax, res, maxVal);
      }
    } else {
      llvm_unreachable("unsupported element type");
    }

    // Store result in the resulting array.
    rewriter.create<KrnlStoreOp>(loc, res, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXClipOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXClipOpLowering>(typeConverter, ctx);
}
