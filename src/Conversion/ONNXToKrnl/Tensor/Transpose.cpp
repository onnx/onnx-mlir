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
  using MDBuider = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MemRefBuilder, MathBuilder>;

  ONNXTransposeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXTransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    MDBuider create(rewriter, loc);

    // Operands and attributes.
    ONNXTransposeOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Value data = operandAdaptor.data();
    Optional<ArrayAttr> permAttr = operandAdaptor.perm();

    // Input and output types.
    MemRefType inMemRefType = data.getType().cast<MemRefType>();
    Type outConvertedType =
        typeConverter->convertType(*op->result_type_begin());
    assert(outConvertedType && outConvertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outMemRefType = outConvertedType.cast<MemRefType>();

    // Get shape.
    ONNXTransposeOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr &outDims = shapeHelper.getOutputDims();

    // If transpose does not permute the non-1 dimensions, it is safe to lower
    // transpose to a view op.
    if (canBeViewOp(inMemRefType, permAttr)) {
      Value view = create.mem.reinterpretCast(data, outDims);
      rewriter.replaceOp(op, view);
      return success();
    }

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc =
        insertAllocAndDeallocSimple(rewriter, op, outMemRefType, loc, outDims);

    // If the last N dimensions are not permuted, do block copying for the last
    // N dimensions. Otherwise, do element-wise copying.
    if (auto numLastDims = unchangedInnerDimensions(permAttr))
      blockTranspose(data, alloc, permAttr, &create, numLastDims);
    else
      scalarTranspose(data, alloc, permAttr, &create);

    rewriter.replaceOp(op, alloc);
    return success();
  }

private:
  // If transpose does not permute the non-1 dimensions, it is safe to lower
  // transpose to a view op.
  bool canBeViewOp(
      MemRefType inMemRefType, Optional<ArrayAttr> permAttr) const {
    ArrayRef<int64_t> dims = inMemRefType.getShape();
    uint64_t rank = inMemRefType.getRank();

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
    return (originalAxes == permutedAxes);
  }

  // Determine how many consecutive inner-most dimensions are not permuted.
  int unchangedInnerDimensions(Optional<ArrayAttr> permAttr) const {
    uint64_t rank = permAttr.value().size();
    int numberOfUnchangedInnerDims = 0;
    for (int i = rank - 1; i >= 0; --i) {
      if (ArrayAttrIntVal(permAttr, i) == i)
        numberOfUnchangedInnerDims++;
      else
        break;
    }

    return (numberOfUnchangedInnerDims > 0) ? numberOfUnchangedInnerDims : NULL;
  }

  // Do transpose by copying elements one-by-one.
  void scalarTranspose(Value inputMemRef, Value outputMemRef,
      Optional<ArrayAttr> permAttr, MDBuider *create) const {
    uint64_t rank = outputMemRef.getType().cast<MemRefType>().getRank();
    ValueRange loopDef = create->krnl.defineLoops(rank);
    SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> ubs;
    create->krnlIE.getShapeAsDims(inputMemRef, ubs);

    create->krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange indices) {
          // Compute the indices used by the load operation.
          SmallVector<IndexExpr, 4> storeIndices;
          for (uint64_t i = 0; i < rank; ++i) {
            Value index = indices[ArrayAttrIntVal(permAttr, i)];
            storeIndices.emplace_back(DimIndexExpr(index));
          }
          Value loadData = createKrnl.load(inputMemRef, indices);
          createKrnl.storeIE(loadData, outputMemRef, storeIndices);
        });
  }

  // Do transpose by copying block of consecutive elements in the inner-most
  // dimensions.
  void blockTranspose(Value inputMemRef, Value outputMemRef,
      Optional<ArrayAttr> permAttr, MDBuider *create, int numLastDims) const {
    Type i64Ty = create->math.getBuilder().getI64Type();
    MemRefType inMemRefType = inputMemRef.getType().cast<MemRefType>();
    uint64_t rank = inMemRefType.getRank();
    uint64_t outerRank = rank - numLastDims;

    // Input and output upperbounds.
    SmallVector<IndexExpr, 4> inUBs;
    create->krnlIE.getShapeAsDims(inputMemRef, inUBs);
    SmallVector<IndexExpr, 4> outUBs;
    create->krnlIE.getShapeAsDims(outputMemRef, outUBs);

    // Strides
    SmallVector<IndexExpr, 4> inStrides, outStrides;
    inStrides.resize_for_overwrite(rank);
    inStrides[rank - 1] = LiteralIndexExpr(1);
    IndexExpr strideIE = LiteralIndexExpr(1);
    for (int i = rank - 2; i >= 0; --i) {
      strideIE = strideIE * inUBs[i + 1];
      inStrides[i] = strideIE;
    }
    strideIE = LiteralIndexExpr(1);
    outStrides.resize_for_overwrite(rank);
    for (int i = rank - 2; i >= 0; --i) {
      strideIE = strideIE * outUBs[i + 1];
      outStrides[i] = strideIE;
    }

    // Block size to copy, computed for the last N dimensions.
    IndexExpr eltSizeInBytes =
        LiteralIndexExpr(getMemRefEltSizeInBytes(inMemRefType));
    IndexExpr sizeInBytesIE = eltSizeInBytes;
    for (uint64_t i = rank - numLastDims; i < rank; ++i)
      sizeInBytesIE = sizeInBytesIE * inUBs[i];
    Value sizeInBytes = create->math.cast(i64Ty, sizeInBytesIE.getValue());

    // Remove the last N dimensions in strides and bounds.
    inStrides.truncate(outerRank);
    outStrides.truncate(outerRank);
    inUBs.truncate(outerRank);
    outUBs.truncate(outerRank);

    // Main loop defined over the outer-most dimensions.
    ValueRange loopDef = create->krnl.defineLoops(outerRank);
    SmallVector<IndexExpr, 4> lbs(outerRank, LiteralIndexExpr(0));
    create->krnl.iterateIE(loopDef, loopDef, lbs, inUBs,
        [&](KrnlBuilder &createKrnl, ValueRange indices) {
          MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createKrnl);
          // Compute destination and source offsets for memcpy.
          IndexExpr destOffsetIE = LiteralIndexExpr(0);
          IndexExpr srcOffsetIE = LiteralIndexExpr(0);
          for (uint64_t i = 0; i < outerRank; ++i) {
            // source offset
            DimIndexExpr srcIndex(indices[i]);
            srcOffsetIE =
                srcOffsetIE + srcIndex * SymbolIndexExpr(inStrides[i]);

            // destination offset
            DimIndexExpr destIndex(indices[ArrayAttrIntVal(permAttr, i)]);
            // Note: index for outStrides is not the permuted index.
            destOffsetIE =
                destOffsetIE + destIndex * SymbolIndexExpr(outStrides[i]);
          }
          IndexExpr destOffsetInBytes = eltSizeInBytes * destOffsetIE;
          // eltSizeInBytes, create.math.cast(i64Ty, destOffsetIE.getValue()));
          IndexExpr srcOffsetInBytes = eltSizeInBytes * srcOffsetIE;
          // call memcpy.
          Value destOffsetVal =
              create.math.cast(i64Ty, destOffsetInBytes.getValue());
          Value srcOffsetVal =
              create.math.cast(i64Ty, srcOffsetInBytes.getValue());
          create.krnl.memcpy(outputMemRef, inputMemRef, sizeInBytes,
              destOffsetVal, srcOffsetVal);
        });
  }
};

void populateLoweringONNXTransposeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
