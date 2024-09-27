/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Transpose.cpp - Lowering Transpose Op ---------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Transpose Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXTransposeOpLowering : public OpConversionPattern<ONNXTransposeOp> {
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MemRefBuilder, MathBuilder>;
  bool enableParallel = false;

  ONNXTransposeOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXTransposeOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ONNXTransposeOp transposeOp,
      ONNXTransposeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = transposeOp.getOperation();
    Location loc = ONNXLoc<ONNXTransposeOp>(op);
    ValueRange operands = adaptor.getOperands();
    MDBuilder create(rewriter, loc);

    // Operands and attributes.
    Value data = adaptor.getData();
    auto permAttr = adaptor.getPerm();

    // Input and output types.
    MemRefType inMemRefType = mlir::cast<MemRefType>(data.getType());
    Type outConvertedType =
        typeConverter->convertType(*op->result_type_begin());
    assert(outConvertedType && mlir::isa<MemRefType>(outConvertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outMemRefType = mlir::cast<MemRefType>(outConvertedType);

    // Get shape.
    ONNXTransposeOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr &outDims = shapeHelper.getOutputDims();

    // If transpose does not permute the non-1 dimensions, it is safe to lower
    // transpose to a view op.
    if (canBeViewOp(inMemRefType, permAttr)) {
      Value view = create.mem.reinterpretCast(data, outDims);
      rewriter.replaceOp(op, view);
      // No work, no need to report on SIMD.
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
      blockTranspose(
          op, data, alloc, permAttr, &create, numLastDims, enableParallel);
    else
      scalarTranspose(op, data, alloc, permAttr, &create, enableParallel);

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }

private:
  // If transpose does not permute the non-1 dimensions, it is safe to lower
  // transpose to a view op.
  bool canBeViewOp(
      MemRefType inMemRefType, std::optional<ArrayAttr> permAttr) const {
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
      MemRefType outputMemRefType, std::optional<ArrayAttr> permAttr) const {
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
  void scalarTranspose(Operation *op, Value inputMemRef, Value outputMemRef,
      std::optional<ArrayAttr> permAttr, MDBuilder *create,
      bool enableParallel) const {
    uint64_t rank = mlir::cast<MemRefType>(outputMemRef.getType()).getRank();
    ValueRange loopDef = create->krnl.defineLoops(rank);
    SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));
    SmallVector<IndexExpr, 4> ubs;
    create->krnlIE.getShapeAsDims(inputMemRef, ubs);

    if (enableParallel) {
      int64_t parId;
      // TODO: consider flattening the outer dims, or along inner dims.
      if (findSuitableParallelDimension(lbs, ubs, 0, 2, parId, 8)) {
        create->krnl.parallel(loopDef[parId]);
        onnxToKrnlParallelReport(
            op, true, parId, lbs[parId], ubs[parId], "scalar transpose");
      } else {
        onnxToKrnlParallelReport(
            op, false, -1, -1, "no dim with enough work in scalar transpose");
      }
    }

    create->krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          // Compute the indices used by the load operation.
          SmallVector<IndexExpr, 4> storeIndices;
          for (uint64_t i = 0; i < rank; ++i) {
            Value index = indices[ArrayAttrIntVal(permAttr, i)];
            storeIndices.emplace_back(DimIE(index));
          }
          Value loadData = createKrnl.load(inputMemRef, indices);
          createKrnl.storeIE(loadData, outputMemRef, storeIndices);
        });
  }

  // Do transpose by copying block of consecutive elements in the inner-most
  // dimensions.
  void blockTranspose(Operation *op, Value inputMemRef, Value outputMemRef,
      std::optional<ArrayAttr> permAttr, MDBuilder *create, int numLastDims,
      bool enableParallel) const {
    Type i64Ty = create->math.getBuilder().getI64Type();
    MemRefType inMemRefType = mlir::cast<MemRefType>(inputMemRef.getType());
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
    inStrides[rank - 1] = LitIE(1);
    IndexExpr strideIE = LitIE(1);
    for (int i = rank - 2; i >= 0; --i) {
      strideIE = strideIE * inUBs[i + 1];
      inStrides[i] = strideIE;
    }
    strideIE = LitIE(1);
    outStrides.resize_for_overwrite(rank);
    for (int i = rank - 2; i >= 0; --i) {
      strideIE = strideIE * outUBs[i + 1];
      outStrides[i] = strideIE;
    }

    // The number of elements in a block to copy, computed for the last N
    // dimensions.
    IndexExpr elemsToCopy = LitIE(1);
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
    SmallVector<IndexExpr, 4> lbs(outerRank, LitIE(0));
    if (enableParallel) {
      int64_t parId;
      // Note that if there is only 1 dim, lastExclusiveDim is automatically
      // reduced to 1 in the findSuitableParallelDimension call.
      if (findSuitableParallelDimension(lbs, inUBs, 0, 2, parId, 8)) {
        create->krnl.parallel(loopDef[parId]);
        onnxToKrnlParallelReport(
            op, true, parId, lbs[parId], inUBs[parId], "block transpose");
      } else {
        onnxToKrnlParallelReport(
            op, false, -1, -1, "no dim with enough work in block transpose");
      }
    }
    create->krnl.iterateIE(loopDef, loopDef, lbs, inUBs,
        [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createKrnl);
          IndexExprScope loopScope(createKrnl);
          // Compute destination and source offsets for memcpy.
          IndexExpr destOffsetIE = LitIE(0);
          IndexExpr srcOffsetIE = LitIE(0);
          for (uint64_t i = 0; i < outerRank; ++i) {
            // source offset
            DimIndexExpr srcIndex(indices[i]);
            srcOffsetIE = srcOffsetIE + srcIndex * SymIE(inStrides[i]);
            // destination offset
            DimIndexExpr destIndex(indices[ArrayAttrIntVal(permAttr, i)]);
            // Note: index for outStrides is not the permuted index.
            destOffsetIE = destOffsetIE + destIndex * SymIE(outStrides[i]);
          }
          // call memcpy.
          create.krnl.memcpy(outputMemRef, inputMemRef, elemsToCopyI64,
              destOffsetIE.getValue(), srcOffsetIE.getValue());
        });
  }
};

void populateLoweringONNXTransposeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXTransposeOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
