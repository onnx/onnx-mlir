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
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Operands and attributes.
    Value data = operandAdaptor.data();
    auto permAttr = transposeOp.perm();

    // Convert the input type to MemRefType.
    Type inConvertedType = typeConverter->convertType(data.getType());
    assert(inConvertedType && inConvertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType inMemRefType = inConvertedType.cast<MemRefType>();
    uint64_t rank = inMemRefType.getShape().size();
    // Convert the output type to MemRefType.
    Type outConvertedType =
        typeConverter->convertType(*op->result_type_begin());
    assert(outConvertedType && outConvertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outMemRefType = outConvertedType.cast<MemRefType>();

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
    for (uint64_t i = 0; i < rank; ++i) {
      int64_t axis = ArrayAttrIntVal(permAttr, i);
      if (dims[axis] != 1)
        permutedAxes.emplace_back(axis);
    }

    if (originalAxes == permutedAxes) {
      // It is safe to lower to a view op.
      Value view =
          create.mem.reinterpretCast(data, shapeHelper.getOutputDims());
      rewriter.replaceOp(op, view);
      return success();
    }

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outMemRefType, loc, shapeHelper.getOutputDims());

    if ((uint64_t)ArrayAttrIntVal(permAttr, rank - 1) == (rank - 1)) {
      // The last dimension is not permuted. Do block copying for the last
      // dimension.
      Type i64Ty = create.math.getBuilder().getI64Type();

      // Input and output upperbounds.
      SmallVector<IndexExpr, 4> inUBs;
      create.krnlIE.getShapeAsDims(data, inUBs);
      SmallVector<IndexExpr, 4> outUBs;
      create.krnlIE.getShapeAsDims(alloc, outUBs);

      // Size to copy.
      Value sizeInBytes =
          create.math.constant(i64Ty, getMemRefEltSizeInBytes(inMemRefType));
      sizeInBytes = create.math.mul(
          sizeInBytes, create.math.cast(i64Ty, inUBs[rank - 1].getValue()));

      // Strides (the last stride is ommitted)
      IndexExpr strideIE = LiteralIndexExpr(1);
      SmallVector<IndexExpr, 4> inStrides, outStrides;
      inStrides.resize_for_overwrite(rank - 1);
      for (uint64_t i = 0; i < rank - 1; ++i) {
        strideIE = strideIE * inUBs[rank - 1 - i];
        inStrides[rank - 2 - i] = strideIE;
      }
      strideIE = LiteralIndexExpr(1);
      outStrides.resize_for_overwrite(rank - 1);
      for (uint64_t i = 0; i < rank - 1; ++i) {
        strideIE = strideIE * outUBs[rank - 1 - i];
        outStrides[rank - 2 - i] = strideIE;
      }

      // Remove the last dimension.
      inUBs.truncate(rank - 1);
      outUBs.truncate(rank - 1);

      ValueRange loopDef = create.krnl.defineLoops(rank - 1);
      SmallVector<IndexExpr, 4> lbs(rank - 1, LiteralIndexExpr(0));
      create.krnl.iterateIE(loopDef, loopDef, lbs, inUBs,
          [&](KrnlBuilder &createKrnl, ValueRange indices) {
            MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createKrnl);
            // Compute destination and source offsets for memcpy.
            IndexExpr destOffsetIE = LiteralIndexExpr(0);
            IndexExpr srcOffsetIE = LiteralIndexExpr(0);
            for (uint64_t i = 0; i < rank - 1; ++i) {
              // source offset
              IndexExpr srcIndex = DimIndexExpr(indices[i]);
              srcOffsetIE =
                  srcOffsetIE + srcIndex * SymbolIndexExpr(inStrides[i]);

              // destination offset
              int64_t k = ArrayAttrIntVal(permAttr, i);
              IndexExpr destIndex = DimIndexExpr(indices[k]);
              destOffsetIE =
                  destOffsetIE + destIndex * SymbolIndexExpr(outStrides[k]);
            }
            // call memcpy.
            Value destOffset = create.math.cast(i64Ty, destOffsetIE.getValue());
            Value srcOffset = create.math.cast(i64Ty, srcOffsetIE.getValue());
            create.krnl.memcpy(alloc, data, sizeInBytes, destOffset, srcOffset);
          });
    } else {
      ValueRange loopDef = create.krnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(data, ubs);

      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange indices) {
            // Compute the indices used by the load operation.
            SmallVector<IndexExpr, 4> storeIndices;
            for (uint64_t i = 0; i < rank; ++i) {
              Value index = indices[ArrayAttrIntVal(permAttr, i)];
              storeIndices.emplace_back(DimIndexExpr(index));
            }

            Value loadData = createKrnl.load(data, indices);
            createKrnl.storeIE(loadData, alloc, storeIndices);
          });
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXTransposeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
