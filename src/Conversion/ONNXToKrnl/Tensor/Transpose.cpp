/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Transpose.cpp - Lowering Transpose Op ---------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
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
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MemRefBuilder, MathBuilder>;

  ONNXTransposeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXTransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    MDBuilder create(rewriter, loc);

    // Operands and attributes.
    ONNXTransposeOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Value data = operandAdaptor.getData();
    auto permAttr = operandAdaptor.getPerm();

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
    Value alloc = create.mem.alignedAlloc(outMemRefType, outDims);

    // If the last N dimensions are not permuted, do block copying for the last
    // N dimensions. Input and Output's MemRefs must use an identity layout to
    // make sure the block's elements are consecutive.
    //
    // Otherwise, do element-wise copying.

    if (auto numLastDims =
            unchangedInnerDimensions(inMemRefType, outMemRefType, permAttr))
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
  // Only apply to MemRefs whose affine layout is identity.
  int unchangedInnerDimensions(MemRefType inputMemRefType,
      MemRefType outputMemRefType, Optional<ArrayAttr> permAttr) const {
    // Verify that the input's affine layout is identity.
    AffineMap im = inputMemRefType.getLayout().getAffineMap();
    if (im.getNumResults() != 1 && !im.isIdentity())
      return 0;
    // Verify that the output's affine layout is identity.
    AffineMap om = outputMemRefType.getLayout().getAffineMap();
    if (om.getNumResults() != 1 && !om.isIdentity())
      return 0;

    int numberOfUnchangedInnerDims = 0;
    uint64_t rank = inputMemRefType.getRank();
    for (int i = rank - 1; i >= 0; --i) {
      if (ArrayAttrIntVal(permAttr, i) == i)
        numberOfUnchangedInnerDims++;
      else
        break;
    }

    return numberOfUnchangedInnerDims;
  }

  // Do transpose by copying elements one-by-one.
  void scalarTranspose(Value inputMemRef, Value outputMemRef,
      Optional<ArrayAttr> permAttr, MDBuilder *create) const {
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
      Optional<ArrayAttr> permAttr, MDBuilder *create, int numLastDims) const {
    Type i64Ty = create->math.getBuilder().getI64Type();
    MemRefType inMemRefType = inputMemRef.getType().cast<MemRefType>();
    uint64_t rank = inMemRefType.getRank();
    uint64_t outerRank = rank - numLastDims;

    // Input and output upper bounds.
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

    // The number of elements in a block to copy, computed for the last N
    // dimensions.
    IndexExpr elemsToCopy = LiteralIndexExpr(1);
    for (uint64_t i = rank - numLastDims; i < rank; ++i)
      elemsToCopy = elemsToCopy * inUBs[i];
    Value elemsToCopyI64 = create->math.cast(i64Ty, elemsToCopy.getValue());

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
          // call memcpy.
          create.krnl.memcpy(outputMemRef, inputMemRef, elemsToCopyI64,
              destOffsetIE.getValue(), srcOffsetIE.getValue());
        });
  }
};

void populateLoweringONNXTransposeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
