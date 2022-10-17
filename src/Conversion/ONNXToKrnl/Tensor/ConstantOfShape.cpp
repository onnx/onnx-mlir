/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ConstantOfShape.cpp - Lowering ConstantOfShape Op -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ConstantOfShape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConstantOfShapeOpLowering : public ConversionPattern {
  ONNXConstantOfShapeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXConstantOfShapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    ONNXConstantOfShapeOpAdaptor operandAdaptor(operands);

    auto valueAttr = llvm::cast<ONNXConstantOfShapeOp>(op)
                         .value()
                         .value()
                         .cast<DenseElementsAttr>();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();
    Type elementType = memRefType.getElementType();
    ArrayRef<int64_t> outputShape = memRefType.getShape();
    size_t rank = outputShape.size();

    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);

    // Allocate memory for the output.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      SmallVector<Value, 2> allocOperands;
      // Load dimensions from the input.
      for (decltype(rank) i = 0; i < rank; ++i) {
        if (outputShape[i] == -1) {
          Value index = create.math.constantIndex(i);
          Value dim = create.krnl.load(operandAdaptor.input(), index);
          Value dimIndex = create.math.castToIndex(dim);
          allocOperands.emplace_back(dimIndex);
        }
      }
      // Allocate memory.
      alloc = create.mem.alignedAlloc(memRefType, allocOperands);
      // Insert deallocation if needed.
      if (insertDealloc) {
        Block *parentBlock = alloc.getDefiningOp()->getBlock();
        memref::DeallocOp dealloc = create.mem.dealloc(alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }

    // Get the constant value from the attribute 'value'.
    Value constantVal;
    if (elementType.isa<IntegerType>()) {
      auto valueIt = valueAttr.getValues<IntegerAttr>().begin();
      auto valueInt = (*valueIt++).cast<IntegerAttr>().getInt();
      constantVal = create.math.constant(elementType, valueInt);
    } else if (elementType.isa<FloatType>()) {
      auto valueIt = valueAttr.getValues<FloatAttr>().begin();
      auto valueFloat = (*valueIt++).cast<FloatAttr>().getValueAsDouble();
      constantVal = create.math.constant(elementType, valueFloat);
    } else
      llvm_unreachable("unsupported element type");

    KrnlBuilder createKrnl(rewriter, loc);
    // Create a Krnl iterate if the output is not a scalar tensor.
    if (!hasAllScalarValues({alloc})) {
      IndexExprScope childScope(&rewriter, loc);
      ValueRange loopDef = createKrnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
      MemRefBoundsIndexCapture allocBounds(alloc);
      SmallVector<IndexExpr, 4> ubs;
      allocBounds.getDimList(ubs);
      createKrnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            createKrnl.store(constantVal, alloc, loopInd);
          });
    } else
      createKrnl.store(constantVal, alloc);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXConstantOfShapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOfShapeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
