/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ConstantOfShape.cpp - Lowering ConstantOfShape Op -------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ConstantOfShape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXConstantOfShapeOpLowering : public ConversionPattern {
  ONNXConstantOfShapeOpLowering(MLIRContext *ctx)
      : ConversionPattern(
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

    // Allocate memory for the output.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      SmallVector<Value, 2> allocOperands;
      // Load dimensions from the input.
      for (decltype(rank) i = 0; i < rank; ++i) {
        auto index = emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
        auto dim =
            rewriter.create<KrnlLoadOp>(loc, operandAdaptor.input(), index);
        auto dimIndex =
            rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), dim);
        allocOperands.emplace_back(dimIndex);
      }
      // Allocate memory.
      alloc = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      // Insert deallocation if needed.
      if (insertDealloc) {
        Block *parentBlock = alloc.getDefiningOp()->getBlock();
        memref::DeallocOp dealloc =
            rewriter.create<memref::DeallocOp>(loc, alloc);
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
    } else {
      llvm_unreachable("unsupported element type");
    }

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
    rewriter.create<KrnlStoreOp>(loc, constantVal, alloc, loopIVs);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXConstantOfShapeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOfShapeOpLowering>(ctx);
}
