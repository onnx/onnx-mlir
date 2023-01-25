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
#include "src/Runtime/OMUnique.h"

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
    Location loc = op->getLoc();
    ONNXTopKOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Value X = operandAdaptor.X();

    // Builders.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
        rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType resMemRefType = convertedType.cast<MemRefType>();

    // Common types.

    // Op's Attributes.
    int64_t rank = resMemRefType.getRank();
    Optional<int64_t> optionalAxis = operandAdaptor.axis();
    int64_t axis = -1;
    if (optionalAxis.has_value()) {
      axis = optionalAxis.value();
      axis = axis < 0 ? axis + rank : axis;
      assert(axis >= 0 && axis < rank && "axis is out of bound");
    }
    int64_t sorted = operandAdaptor.sorted();
    // Insert an allocation and deallocation for the results of this operation.
    bool insertDealloc = checkInsertDealloc(op, /*resultIndex=*/0);
    ArrayRef<int64_t> xShape = getShape(X.getType());
    Type i64Type = rewriter.getI64Type();
    DimsExpr resDims;
    // create.krnlIE.getShapeAsDims(X, resDims);
    MemRefType resMemrefForAllocType = MemRefType::get(xShape, i64Type);
    insertDealloc = checkInsertDealloc(op, /*resultIndex=*/1);
    Value resIndexMemRef = insertAllocAndDeallocSimple(rewriter, op,
        MemRefType::get(resMemrefForAllocType.getShape(), i64Type), loc,
        resDims, insertDealloc);
    Value indices;
    Value reverse_indices;
    Value counts;

    // Compute argUnique of X along axis.
    Value argUnique = emitArgUnique(rewriter, loc, X, axis, /*sorted=*/sorted,
        indices, reverse_indices, counts, OMUNIQUE_FLAG_COUNTONLY);
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
    Value resMemRef = insertAllocAndDeallocSimple(
        rewriter, op, resMemrefForAllocType, loc, resDims, insertDealloc);
    rewriter.replaceOp(
        op, {resMemRef, resIndexMemRef, resIndexMemRef, resIndexMemRef});
    return success();
  }
};

void populateLoweringONNXUniqueOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXUniqueOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
