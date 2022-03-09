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
                         .getValue()
                         .cast<DenseElementsAttr>();

    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto elementType = memRefType.getElementType();
    size_t rank = memRefType.cast<ShapedType>().getRank();

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
        Value index = create.math.constantIndex(i);
        Value dim = create.krnl.load(operandAdaptor.input(), index);
        Value dimIndex = create.math.castToIndex(dim);
        allocOperands.emplace_back(dimIndex);
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
      constantVal = emitConstantOp(rewriter, loc, elementType, valueInt);
    } else if (elementType.isa<FloatType>()) {
      auto valueIt = valueAttr.getValues<FloatAttr>().begin();
      auto valueFloat = (*valueIt++).cast<FloatAttr>().getValueAsDouble();
      constantVal = emitConstantOp(rewriter, loc, elementType, valueFloat);
    } else
      llvm_unreachable("unsupported element type");

    SmallVector<Value, 4> loopIVs;
    // Create a Krnl iterate if the output is not a scalar tensor.
    if (!hasAllScalarValues({alloc})) {
      BuildKrnlLoop loops(rewriter, loc, rank);
      loops.createDefineAndIterateOp(alloc);
      Block *iterationBlock = loops.getIterateBlock();
      // Get IVs.
      for (auto arg : iterationBlock->getArguments())
        loopIVs.push_back(arg);
      // Insert instructions inside the KernelIterateOp body.
      rewriter.setInsertionPointToStart(iterationBlock);
    }

    // Store the constant value to the output.
    create.krnl.store(constantVal, alloc, loopIVs);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXConstantOfShapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOfShapeOpLowering>(typeConverter, ctx);
}
