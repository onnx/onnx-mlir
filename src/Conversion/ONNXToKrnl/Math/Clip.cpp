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
    ONNXClipOp clipOp = cast<ONNXClipOp>(op);
    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());

    ONNXClipOpAdaptor operandAdaptor(operands);
    ONNXClipOpShapeHelper shapeHelper(&clipOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapeComputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapeComputed));

    Value input = operandAdaptor.input();
    Value min = operandAdaptor.min();
    Value max = operandAdaptor.max();

    // Insert an allocation and deallocation for the result of this operation.
    bool insertDealloc = checkInsertDealloc(op);
    Value alloc =
        (hasAllConstantDimensions(memRefType))
            ? insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc)
            : insertAllocAndDealloc(
                  memRefType, loc, rewriter, insertDealloc, input);

    auto computeResult =
        [&](MultiDialectBuilder<KrnlBuilder, MathBuilder> &create,
            const ValueRange &indices) {
          Value loadedVal = create.krnl.load(input, indices);
          Value res = loadedVal;
          if (!min.getType().isa<NoneType>()) {
            Value minVal = create.krnl.load(min);
            Value lessThanMin = create.math.slt(res, minVal);
            res = create.math.select(lessThanMin, minVal, res);
          }
          if (!max.getType().isa<NoneType>()) {
            Value maxVal = create.krnl.load(max);
            Value lessThanMax = create.math.slt(res, maxVal);
            res = create.math.select(lessThanMax, res, maxVal);
          }
          create.krnl.store(res, alloc, indices);
        };

    // Create a loop only is one of the operands is not a scalar tensor.
    if (!hasAllScalarValues(operands)) {
      KrnlBuilder createKrnl(rewriter, loc);
      uint64_t numLoops = memRefType.getRank();
      ValueRange loopDef = createKrnl.defineLoops(numLoops);

      SmallVector<IndexExpr, 4> lbs(numLoops, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      for (uint64_t i = 0; i < numLoops; ++i)
        ubs.emplace_back(shapeHelper.dimsForOutput()[i]);

      createKrnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange indices) {
            MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
            computeResult(create, indices);
          });
    } else {
      MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
      computeResult(create, {});
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXClipOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXClipOpLowering>(typeConverter, ctx);
}
