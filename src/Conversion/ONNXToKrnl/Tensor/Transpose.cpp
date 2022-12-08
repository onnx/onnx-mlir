/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Transpose.cpp - Lowering Transpose Op ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Transpose Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXTransposeOpLowering : public ConversionPattern {
  ONNXTransposeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXTransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTransposeOpAdaptor operandAdaptor(operands);
    ONNXTransposeOp transposeOp = llvm::cast<ONNXTransposeOp>(op);
    Location loc = op->getLoc();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
        rewriter, loc);

    // Operands and attributes.
    Value data = operandAdaptor.data();
    auto permAttr = transposeOp.perm();

    // Convert the input type to MemRefType.
    Type inConvertedType = typeConverter->convertType(data.getType());
    assert(inConvertedType && inConvertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType inMemRefType = inConvertedType.cast<MemRefType>();
    uint64_t inRank = inMemRefType.getShape().size();
    // Convert the output type to MemRefType.
    Type outConvertedType =
        typeConverter->convertType(*op->result_type_begin());
    assert(outConvertedType && outConvertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outMemRefType = outConvertedType.cast<MemRefType>();
    uint64_t outRank = outMemRefType.getShape().size();

    // Get shape.
    ONNXTransposeOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // If the order of the dimensions whose value is not 1 does not change after
    // transpose, it is safe to lower transpose to a view op.
    ArrayRef<int64_t> dims = inMemRefType.getShape();
    SmallVector<int64_t, 4> originalAxes;
    for (uint64_t axis = 0; axis < dims.size(); ++axis)
      if (dims[axis] != 1)
        originalAxes.emplace_back(axis);
    SmallVector<int64_t, 4> permutedAxes;
    for (uint64_t i = 0; i < inRank; ++i) {
      int64_t axis = ArrayAttrIntVal(permAttr, i);
      if (dims[axis] != 1)
        permutedAxes.emplace_back(axis);
    }

    if (originalAxes == permutedAxes) {
      // It is safe to lower to a view op.
      MemRefBuilder createMemRef(rewriter, loc);
      Value view =
          createMemRef.reinterpretCast(data, shapeHelper.getOutputDims());
      rewriter.replaceOp(op, view);
      return success();
    }

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outMemRefType, loc, shapeHelper.getOutputDims());

    ValueRange loopDef = create.krnl.defineLoops(outRank);
    SmallVector<IndexExpr, 4> lbs(outRank, LiteralIndexExpr(0));

    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(data, ubs);
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange indices) {
          // Compute the indices used by the load operation.
          SmallVector<IndexExpr, 4> storeIndices;
          for (uint64_t i = 0; i < outRank; ++i) {
            Value index = indices[ArrayAttrIntVal(permAttr, i)];
            storeIndices.emplace_back(DimIndexExpr(index));
          }

          Value loadData = createKrnl.load(data, indices);
          createKrnl.storeIE(loadData, alloc, storeIndices);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXTransposeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
