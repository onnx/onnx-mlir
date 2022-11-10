
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Concat.cpp - Lowering ConcatShapeTranspose Op ---------===//
//
// Copyright 2022 The IBM Research Authors.
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
    auto loc = op->getLoc();

    ONNXConcatShapeTransposeOpAdaptor operandAdaptor(
        operands, op->getAttrDictionary());
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);

    // Compute concat output shape.
    unsigned numInputs = op->getNumOperands();
    Value firstInput = operandAdaptor.inputs().front();
    ArrayRef<int64_t> commonShape =
        firstInput.getType().cast<ShapedType>().getShape();
    // Type dataElementType =
    // firstInput.getType().cast<ShapedType>().getElementType();
    uint64_t commonRank = commonShape.size();
    int64_t axisIndex = operandAdaptor.axis();

    // Negative axis means values are counted from the opposite side.
    // TOFIX should be in normalization pass
    if (axisIndex < 0)
      axisIndex += commonRank;

    IndexExprScope IEScope(&rewriter, loc);
    DimsExpr outputConcatDims(commonRank);
    MemRefBoundsIndexCapture firstInputBounds(operandAdaptor.inputs()[0]);
    for (unsigned dim = 0; dim < commonRank; dim++) {
      outputConcatDims[dim] = firstInputBounds.getDim(dim);
    }
    IndexExpr cumulativeAxisSize =
        DimIndexExpr(firstInputBounds.getDim(axisIndex));

    // Handle the rest of input
    for (unsigned i = 1; i < numInputs; ++i) {
      Value currentInput = operandAdaptor.inputs()[i];
      MemRefBoundsIndexCapture currInputBounds(currentInput);
      for (unsigned dim = 0; dim < commonRank; dim++) {
        if (dim == axisIndex) {
          DimIndexExpr currentSize(currInputBounds.getDim(axisIndex));
          cumulativeAxisSize = cumulativeAxisSize + currentSize;
        } else {
          if (currInputBounds.getDim(dim).isLiteral()) {
            // The size of current dimension of current input  is a constant
            outputConcatDims[dim] = currInputBounds.getDim(dim);
          }
        }
      }
    }
    outputConcatDims[axisIndex] = cumulativeAxisSize;

    // Shape for Shape
    uint64_t start = operandAdaptor.start();
    uint64_t end = commonRank;
    if (operandAdaptor.end().has_value()) {
      end = operandAdaptor.end().value();
    }

    // SmallVector<int64_t, 4> outputDims;
    // outputDims.emplace_back(end-start);
    // auto outputShapeType = RankedTensorType::get(outputDims,
    // rewriter.getIntegerType(64));
    auto outputShapeType = op->getResultTypes()[0];

    // Alloc and set value for ShapeOp output
    auto convertedShapeType =
        typeConverter->convertType(outputShapeType).cast<MemRefType>();
    Type elementType = convertedShapeType.getElementType();
    Value shapeAlloc =
        insertAllocAndDealloc(convertedShapeType, loc, rewriter, false);
    for (uint64_t i = start; i < end; i++) {
      Value intVal =
          create.math.cast(elementType, outputConcatDims[i].getValue());
      create.krnl.store(
          intVal, shapeAlloc, create.math.constantIndex(i - start));
    }

    // Convert the output type to MemRefType.
    DimsExpr outputTransposeDims(commonRank);
    auto permAttr = operandAdaptor.perm();
    for (uint64_t i = 0; i < commonRank; i++) {
      auto current = outputConcatDims[ArrayAttrIntVal(permAttr, i)];
      outputTransposeDims[i] = current;
    }
    Type t = op->getResultTypes()[1];
    auto outputTransposeType = typeConverter->convertType(t).cast<MemRefType>();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputTransposeType, loc, outputTransposeDims);
    unsigned int rank = commonRank;

    // Creates loops, one for each input.
    // Since the each input should have same size for each dimension(except
    // axis), we will try to make the loop upper bound the same for futher
    // optimization. Difference may come from constant vs. dynamic, or dynamic
    // dim of different inputs.
    KrnlBuilder createKrnl(rewriter, loc);
    SmallVector<IndexExpr, 4> commonUB = outputConcatDims;
    IndexExpr accumulatedOffset = LiteralIndexExpr(0);
    unsigned int inputNum = operands.size();
    for (unsigned int i = 0; i < inputNum; ++i) {
      // Since the acculatedOffsetValue will be used in a nested IndexExprScope,
      // we get the Value of this IndexExpr and pass it as a symbol
      Value accumulatedOffsetValue = accumulatedOffset.getValue();
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Create loop.
      ValueRange loopDef = createKrnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
      MemRefBoundsIndexCapture bounds(operands[i]);
      SmallVector<IndexExpr, 4> ubs;
      bounds.getDimList(ubs);
      // For each input, only the dimension 'axis' is different
      auto axis = axisIndex;
      commonUB[axis] = ubs[axis];
      createKrnl.iterateIE(loopDef, loopDef, lbs, commonUB,
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
      MemRefBoundsIndexCapture operandJBounds(operands[i]);
      accumulatedOffset = accumulatedOffset + operandJBounds.getDim(axis);
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
