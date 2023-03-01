
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Concat.cpp - Lowering ConcatShapeTranspose Op ---------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Concat OperatorShapeTranspose to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConcatShapeTransposeOpLowering : public ConversionPattern {
  ONNXConcatShapeTransposeOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXConcatShapeTransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXConcatShapeTransposeOpAdaptor operandAdaptor(
        operands, op->getAttrDictionary());
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXConcatShapeTransposeOpShapeHelper shapeHelper(
        op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Compute concat output shape.
    unsigned numInputs = op->getNumOperands();
    Value firstInput = operandAdaptor.getInputs().front();
    ArrayRef<int64_t> commonShape =
        firstInput.getType().cast<ShapedType>().getShape();
    // firstInput.getType().cast<ShapedType>().getElementType();
    uint64_t rank = commonShape.size();
    int64_t axis = operandAdaptor.getAxis();

    // Negative axis means values are counted from the opposite side.
    // TODO should be in normalization pass
    if (axis < 0)
      axis += rank;

    IndexExprScope IEScope(&rewriter, loc);
    DimsExpr outputConcatDims(rank);
    for (unsigned dim = 0; dim < rank; dim++) {
      outputConcatDims[dim] = create.krnlIE.getShapeAsDim(firstInput, dim);
    }
    IndexExpr cumulativeAxisSize =
        create.krnlIE.getShapeAsDim(firstInput, axis);

    // Handle the rest of input
    for (unsigned i = 1; i < numInputs; ++i) {
      Value currInput = operandAdaptor.getInputs()[i];
      for (unsigned dim = 0; dim < rank; dim++) {
        if (dim == axis) {
          IndexExpr currentSize = create.krnlIE.getShapeAsDim(currInput, axis);
          cumulativeAxisSize = cumulativeAxisSize + currentSize;
        } else {
          IndexExpr currInputPossiblyLit =
              create.krnlIE.getShapeAsDim(currInput, dim);
          if (currInputPossiblyLit.isLiteral()) {
            // The size of current dimension of current input  is a constant
            outputConcatDims[dim] = currInputPossiblyLit;
          }
        }
      }
    }
    outputConcatDims[axis] = cumulativeAxisSize;

    // Shape for Shape
    int64_t start = operandAdaptor.getStart();
    int64_t end = rank;
    if (operandAdaptor.getEnd().has_value()) {
      end = operandAdaptor.getEnd().value();
    }
    // Handle negative
    if (start < 0)
      start += rank;
    if (end < 0)
      end += rank;
    Type outputShapeType = op->getResultTypes()[0];

    // Alloc and set value for ShapeOp output
    auto convertedShapeType =
        typeConverter->convertType(outputShapeType).cast<MemRefType>();
    Value shapeAlloc = create.mem.alignedAlloc(
        convertedShapeType, shapeHelper.getOutputDims());
    Type elementType = convertedShapeType.getElementType();
    for (int64_t i = start; i < end; i++) {
      Value intVal =
          create.math.cast(elementType, outputConcatDims[i].getValue());
      create.krnl.store(
          intVal, shapeAlloc, create.math.constantIndex(i - start));
    }

    // Convert the output type to MemRefType.
    DimsExpr outputTransposeDims = shapeHelper.getOutputDims(1);
    ArrayAttr permAttr = operandAdaptor.getPermAttr();
    Type t = op->getResultTypes()[1];
    auto outputTransposeType = typeConverter->convertType(t).cast<MemRefType>();
    Value alloc =
        create.mem.alignedAlloc(outputTransposeType, outputTransposeDims);

    // Creates loops, one for each input.
    // Since the each input should have same size for each dimension(except
    // axis), we will try to make the loop upper bound the same for further
    // optimization. Difference may come from constant vs. dynamic, or dynamic
    // dim of different inputs.
    SmallVector<IndexExpr, 4> commonUB = outputConcatDims;
    IndexExpr accumulatedOffset = LiteralIndexExpr(0);
    for (unsigned int i = 0; i < numInputs; ++i) {
      // Since the accumulatedOffsetValue will be used in a nested
      // IndexExprScope, we get the Value of this IndexExpr and pass it as a
      // symbol
      Value accumulatedOffsetValue = accumulatedOffset.getValue();
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Create loop.
      ValueRange loopDef = create.krnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(operands[i], ubs);
      // For each input, only the dimension 'axis' is different
      commonUB[axis] = ubs[axis];
      create.krnl.iterateIE(loopDef, loopDef, lbs, commonUB,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            // Indices for the read and write.
            SmallVector<Value, 4> readIndices, writeIndices;
            for (unsigned int r = 0; r < rank; ++r) {
              if (r != axis || i == 0)
                writeIndices.emplace_back(loopInd[r]);
              else {
                IndexExprScope IEScope(&rewriter, loc);
                IndexExpr writeOffset = DimIndexExpr(loopInd[r]);
                IndexExpr accumulatedOffsetIE =
                    SymbolIndexExpr(accumulatedOffsetValue);
                writeOffset = writeOffset + accumulatedOffsetIE;
                writeIndices.emplace_back(writeOffset.getValue());
              }
            }
            // Insert copy.
            Value loadData = createKrnl.load(operands[i], loopInd);
            SmallVector<Value, 4> transposedWriteIndices(rank);
            for (uint64_t i = 0; i < rank; i++) {
              transposedWriteIndices[i] =
                  writeIndices[ArrayAttrIntVal(permAttr, i)];
            }
            createKrnl.store(loadData, alloc, transposedWriteIndices);
          });
      Value operandJ = operands[i];
      accumulatedOffset =
          accumulatedOffset + create.krnlIE.getShapeAsDim(operandJ, axis);
    }
    SmallVector<Value, 2> outputs;
    rewriter.replaceOp(op, {shapeAlloc, alloc});
    return success();
  }
};

void populateLoweringONNXConcatShapeTransposeOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXConcatShapeTransposeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
