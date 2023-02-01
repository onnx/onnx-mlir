/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Unique.cpp - Lowering Unique Op ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Unique Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "onnx-mlir/Runtime/OMTensor.h"
using namespace mlir;

namespace onnx_mlir {

struct ONNXUniqueOpLowering : public ConversionPattern {
  ONNXUniqueOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXUniqueOp::getOperationName(), 1, ctx) {}

  ///
  /// Intermediate data are presented below for better understanding:
  ///
  /// there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
  /// A: [[1, 1], [1, 1]],
  ///    [[0, 1], [0, 1]],
  ///    [[2, 1], [2, 1]],
  ///    [[0, 1], [0, 1]].
  ///
  /// there are 3 unique subtensors:
  /// [[1, 1], [1, 1]],
  /// [[0, 1], [0, 1]],
  /// [[2, 1], [2, 1]].
  ///
  /// sorted unique subtensors:
  /// B: [[0, 1], [0, 1]],
  ///    [[1, 1], [1, 1]],
  ///    [[2, 1], [2, 1]].
  ///
  /// output_Y is constructed from B:
  /// [[[0. 1.], [1. 1.], [2. 1.]],
  /// [[0. 1.], [1. 1.], [2. 1.]]]
  ///
  /// output_indices is to map from B to A:
  /// [1, 0, 2]
  ///
  /// output_inverse_indices is to map from A to B:
  /// [1, 0, 2, 0]
  ///
  /// output_counts = [2 1 1]
  ///

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTopKOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    ONNXUniqueOp uniqueOp = llvm::cast<ONNXUniqueOp>(op);
    Location loc = op->getLoc();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
        rewriter, loc);
    IndexExprScope scope(create.krnl);
    ONNXUniqueOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    Value X = operandAdaptor.X();
    int64_t rank = create.krnlIE.getShapedTypeRank(X);
    ArrayRef<int64_t> xShape = getShape(X.getType());
    int64_t sorted = operandAdaptor.sorted();
    Optional<int64_t> optionalAxis = operandAdaptor.axis();
    int64_t axis = -1;
    if (optionalAxis.has_value()) {
      axis = optionalAxis.value();
      axis = axis < 0 ? axis + rank : axis;
      assert(axis >= 0 && axis < rank && "axis is out of bound");
    }

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType resMemRefType = convertedType.cast<MemRefType>();

    // Calculate maximum output shapes for ouputs
    DimsExpr outputYBufDims;
    DimsExpr outputIndexBufDims;
    uint64_t inputElementNum = 1;
    for (int64_t i = 0; i < rank; i++) {
      inputElementNum = inputElementNum * xShape[i];
    }
    if (axis < 0) {
      outputYBufDims.emplace_back(LiteralIndexExpr(inputElementNum));
      outputIndexBufDims.emplace_back(LiteralIndexExpr(inputElementNum));
    } else {
      for (int64_t i = 0; i < rank; i++) {
        outputYBufDims.emplace_back(LiteralIndexExpr(xShape[i]));
      }
      outputIndexBufDims.emplace_back(LiteralIndexExpr(inputElementNum));
    }
    // Insert an allocation and deallocation for the results of this operation.
    // For Y output
    bool insertDealloc = false; // XXX = true;
    Type i64Type = rewriter.getI64Type();

    Value outputYBuf = insertAllocAndDeallocSimple(rewriter, op,
        MemRefType::get(xShape, i64Type), loc, outputYBufDims,
        insertDealloc);
    Value indicesBuf = insertAllocAndDeallocSimple(rewriter, op,
        MemRefType::get(xShape, i64Type), loc, outputIndexBufDims,
        insertDealloc);
    Value reverse_indicesBuf = insertAllocAndDeallocSimple(rewriter, op,
        MemRefType::get(xShape, i64Type), loc, outputIndexBufDims,
        insertDealloc);;
    Value countsBuf = insertAllocAndDeallocSimple(rewriter, op,
        MemRefType::get(xShape, i64Type), loc, outputIndexBufDims,
        insertDealloc);;
    // Compute argUnique of X along axis.
    Value total = emitArgUnique(rewriter, loc, X, axis, /*sorted=*/sorted,
        outputYBuf, indicesBuf, reverse_indicesBuf, countsBuf);

    // Calculate output shapes for ouputs according to the results
#if 0
    NonAffineIndexExpr totalDimExpr = NonAffineIndexExpr(total);
    DimsExpr outputYDims;
    DimsExpr outputIndexDims;
    if (axis < 0) {
      outputYBufDims.emplace_back(totalDimExpr);
      outputIndexBufDims.emplace_back(totalDimExpr);
    } else {
      for (int64_t i = 0; i < rank; i++) {
        DimIndexExpr tDimExpr = LiteralIndexExpr(xShape[i]);
        if (i == axis)
          tDimExpr = totalDimExpr;
        outputIndexDims.emplace_back(tDimExpr);
      }
    }
#endif
#if 0
    // Produce the final result.
    SmallVector<IndexExpr> zeroDims(rank, LiteralIndexExpr(0));
    ValueRange loopDef = create.krnl.defineLoops(rank);
    create.krnl.iterateIE(loopDef, loopDef, zeroDims, resDims,
        [&](KrnlBuilder &createKrnl, ValueRange resLoopInd) {
          Value resInd = createKrnl.load(argUnique, resLoopInd);
          SmallVector<Value> resIndexLoopInd(resLoopInd);
          resIndexLoopInd[axis] = resInd;
          // Store value.
          Value val = createKrnl.load(X, resIndexLoopInd);
          createKrnl.store(val, resMemRef, resLoopInd);
          // Store index.
          Value resIndI64 =
              rewriter.create<arith::IndexCastOp>(loc, i64Type, resInd);
          createKrnl.store(resIndI64, resIndexMemRef, resLoopInd);
        });

#endif
    rewriter.replaceOp(
        op, {outputYBuf, indicesBuf, reverse_indicesBuf, countsBuf});
    return success();
  }
};

void populateLoweringONNXUniqueOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXUniqueOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
