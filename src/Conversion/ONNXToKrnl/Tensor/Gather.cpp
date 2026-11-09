/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gather.cpp - Lowering Gather Op ---------------------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXGatherOpLowering : public OpConversionPattern<ONNXGatherOp> {
  ONNXGatherOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableParallel, bool enableCollapse)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXGatherOp::getOperationName());
    // Not and-ed with this->enableParallel: see Slice.cpp for why the two bools
    // stay independent.
    this->enableCollapse = enableCollapse;
  }
  bool enableParallel = false;
  bool enableCollapse = false;

  LogicalResult matchAndRewrite(ONNXGatherOp gatherOp,
      ONNXGatherOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = gatherOp.getOperation();
    Location loc = ONNXLoc<ONNXGatherOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXGatherOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Operands and attributes.
    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();
    int64_t axisLit = adaptor.getAxis();
    int64_t dataRank = mlir::cast<MemRefType>(data.getType()).getRank();
    int64_t indicesRank = mlir::cast<MemRefType>(indices.getType()).getRank();

    // Determine whether indices may be negative.
    bool indicesMayBeNegative = !indicesAreNonNegativeConstants(indices);

    // Negative value means counting dimensions from the back.
    axisLit = axisLit < 0 ? axisLit + dataRank : axisLit;

    // Check the value of indices and change it to zero if out-of-bound
    genSafeCodeForGatherAlike(rewriter, loc, op, data, indices, axisLit);

    int64_t outputRank = shapeHelper.getOutputDims().size();
    int iIndexStart = 0;
    int jIndexStart = iIndexStart + axisLit;
    int kIndexStart = jIndexStart + indicesRank - (axisLit + 1);

    LiteralIndexExpr zeroIE(0);
    DimsExpr dataDims;
    create.krnlIE.getShapeAsDims(data, dataDims);

    /*
      The pattern that we are using is that of numpy.take.

      Ni, Nk = data.shape[:axis], data.shape[axis+1:]
      Nj = indices.shape
      for ii in ndindex(Ni):
        for jj in ndindex(Nj):
          for kk in ndindex(Nk):
            out[ii + jj + kk] = data[ii + (indices[jj],) + kk]
    */
    // Fast path: when the data slice kk is contiguous in both data and output,
    // iterate only over the outer (ii, jj) levels and copy each slice of K
    // elements with one memcpy.
    if (canCopyRows(data, alloc, axisLit)) {
      emitRowCopies(op, create, data, indices, alloc, axisLit, indicesRank,
          dataDims, shapeHelper.getOutputDims(), indicesMayBeNegative);
      rewriter.replaceOp(op, alloc);
      onnxToKrnlSimdReport(op);
      return success();
    }

    // Define loops and iteration trip counts (equivalent to size of output)
    ValueRange loopDef = create.krnl.defineLoops(outputRank);
    DimsExpr lbs(outputRank, zeroIE);
    DimsExpr ubs = shapeHelper.getOutputDims();
    // Enable parallelism if required.
    // out[ii + jj + kk] = data[ii + (indices[jj],) + kk] writes each output
    // element exactly once and reads only operands, so the whole nest is
    // order-independent and the collapse claim can be the whole window.
    //
    // bodyCost 10: one innermost iteration loads an index, clamps it with a
    // compare and a select or two, loads through the resulting non-affine
    // access function, and stores -- a handful of ops, not a plain copy.
    KrnlParallelPlan plan(loopDef, enableCollapse, /*parFirstInclusiveDim=*/0,
        /*parLastExclusiveDim=*/outputRank,
        /*collapseLastExclusiveDim=*/outputRank,
        {.minTripCountForParallel = 4, .bodyCost = 10});
    if (enableParallel)
      plan.tryCreateParallel(create.krnl, op, "gather", lbs, ubs);
    create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);
          SymbolIndexExpr axisDim(dataDims[axisLit]);

          // compute the loop indices for the output
          SmallVector<IndexExpr, 4> outputAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, outputAccessFct);

          // Compute access function for indices[jjs].
          SmallVector<IndexExpr, 4> indicesAccessFct;
          for (int j = 0; j < indicesRank; ++j)
            indicesAccessFct.emplace_back(outputAccessFct[jIndexStart + j]);
          Value indexVal = createKrnl.loadIE(indices, indicesAccessFct);
          // Loaded value is an index that is not affine
          IndexExpr index = NonAffineIndexExpr(indexVal);

          // When index may be negative, add axis Dim to it.
          if (indicesMayBeNegative)
            index = index.selectOrSelf(index < zeroIE, index + axisDim);

          if (enableSafeCodeGen) {
            index = index.selectOrSelf(index < 0, zeroIE);
            index = index.selectOrSelf(index >= axisDim, axisDim - 1);
          }

          // Compute access function of data: data[ii + (indices[jj],) + kk]
          SmallVector<IndexExpr, 4> dataAccessFct;
          // First add indices iis
          for (int i = 0; i < axisLit; ++i)
            dataAccessFct.emplace_back(outputAccessFct[iIndexStart + i]);
          // Then add indices[jj] (indexVal).
          dataAccessFct.emplace_back(index);
          // Then add kks.
          for (int k = axisLit + 1; k < dataRank; ++k)
            dataAccessFct.emplace_back(outputAccessFct[kIndexStart + k]);
          Value dataVal = createKrnl.loadIE(data, dataAccessFct);

          // Save data into output
          createKrnl.storeIE(dataVal, alloc, outputAccessFct);
        });
    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }

private:
  // Smallest slice, in bytes, worth a memcpy call per row instead of an
  // element-wise copy that LLVM can inline.
  static constexpr int64_t minRowCopyBytes = 64;

  // The data slice after the axis, data[ii, idx, *], is K = prod(dataDims[axis
  // + 1 :]) consecutive elements in data and lands as K consecutive elements
  // in the output, provided both memrefs have an identity layout (no strides,
  // no offset: krnl.memcpy addresses the aligned pointer plus an element
  // offset). Restricted to int/float elements, and to a literal K large enough
  // to amortize a call; K literal also lets the parallel plan know the true
  // body cost.
  bool canCopyRows(Value data, Value output, int64_t axisLit) const {
    auto dataType = mlir::cast<MemRefType>(data.getType());
    auto outputType = mlir::cast<MemRefType>(output.getType());
    int64_t dataRank = dataType.getRank();
    // Need a non-empty slice, and at least one outer level to iterate over.
    if (axisLit + 1 >= dataRank ||
        outputType.getRank() <= dataRank - axisLit - 1)
      return false;
    if (!dataType.getLayout().isIdentity() ||
        !outputType.getLayout().isIdentity())
      return false;
    Type elementType = dataType.getElementType();
    if (!elementType.isIntOrFloat())
      return false;
    // Read K off the static type, not dataDims: an IndexExpr product over a
    // dynamic dim emits ops, which would linger when this check fails.
    int64_t K = 1;
    for (int64_t d = axisLit + 1; d < dataRank; ++d) {
      if (dataType.isDynamicDim(d))
        return false;
      K *= dataType.getDimSize(d);
    }
    return K * getMemRefEltSizeInBytes(dataType) >= minRowCopyBytes;
  }

  // for (ii, jj) in outer output levels:      // collapse-eligible, parallel
  //   idx = indices[jj] (normalized, clamped)
  //   memcpy(&out[ii, jj, 0...], &data[ii, idx, 0...], K)
  void emitRowCopies(Operation *op,
      MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
          MathBuilder> &create,
      Value data, Value indices, Value alloc, int64_t axisLit,
      int64_t indicesRank, const DimsExpr &dataDims, const DimsExpr &outDims,
      bool indicesMayBeNegative) const {
    int64_t dataRank = dataDims.size();
    int64_t outerRank = axisLit + indicesRank;
    int64_t jIndexStart = axisLit;

    // K, the slice length, and strides of the levels the nest walks: data
    // levels [0, axis] and output levels [0, outerRank), all in elements.
    IndexExpr K = LitIE(1);
    for (int64_t d = axisLit + 1; d < dataRank; ++d)
      K = K * dataDims[d];
    DimsExpr dataStrides(axisLit + 1);
    dataStrides[axisLit] = K;
    for (int64_t d = axisLit - 1; d >= 0; --d)
      dataStrides[d] = dataStrides[d + 1] * dataDims[d + 1];
    DimsExpr outStrides(outerRank);
    outStrides[outerRank - 1] = K;
    for (int64_t d = outerRank - 2; d >= 0; --d)
      outStrides[d] = outStrides[d + 1] * outDims[d + 1];
    Value kI64 = create.math.constant(
        create.krnl.getBuilder().getI64Type(), K.getLiteral());

    ValueRange loopDef = create.krnl.defineLoops(outerRank);
    DimsExpr lbs(outerRank, LitIE(0));
    DimsExpr ubs(outDims.begin(), outDims.begin() + outerRank);
    // Each (ii, jj) writes its own disjoint K-element slice of the output and
    // reads only operands, so the whole outer nest is order-independent and
    // the collapse claim is the whole window. bodyCost is K: one iteration
    // copies K elements (plus one index load and clamp, negligible against a
    // literal K of at least minRowCopyBytes).
    KrnlParallelPlan plan(loopDef, enableCollapse, /*parFirstInclusiveDim=*/0,
        /*parLastExclusiveDim=*/outerRank,
        /*collapseLastExclusiveDim=*/outerRank,
        {.minTripCountForParallel = 4, .bodyCost = K.getLiteral()});
    if (enableParallel)
      plan.tryCreateParallel(create.krnl, op, "gather rows", lbs, ubs);
    create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          IndexExprScope innerLoopScope(createKrnl);
          LiteralIndexExpr zeroIE(0);
          SymbolIndexExpr axisDim(dataDims[axisLit]);
          SmallVector<IndexExpr, 4> outerInd;
          getIndexExprList<DimIndexExpr>(loopInd, outerInd);

          SmallVector<IndexExpr, 4> indicesAccessFct;
          for (int64_t j = 0; j < indicesRank; ++j)
            indicesAccessFct.emplace_back(outerInd[jIndexStart + j]);
          Value indexVal = createKrnl.loadIE(indices, indicesAccessFct);
          IndexExpr index = NonAffineIndexExpr(indexVal);
          if (indicesMayBeNegative)
            index = index.selectOrSelf(index < zeroIE, index + axisDim);
          if (enableSafeCodeGen) {
            index = index.selectOrSelf(index < 0, zeroIE);
            index = index.selectOrSelf(index >= axisDim, axisDim - 1);
          }

          // srcOffset = sum_{i<axis} ii_i * dataStride_i + index * K.
          IndexExpr srcOffset = index * SymIE(dataStrides[axisLit]);
          for (int64_t i = 0; i < axisLit; ++i)
            srcOffset = srcOffset + outerInd[i] * SymIE(dataStrides[i]);
          // destOffset = sum_{o<outerRank} o_o * outStride_o.
          IndexExpr destOffset = LitIE(0);
          for (int64_t o = 0; o < outerRank; ++o)
            destOffset = destOffset + outerInd[o] * SymIE(outStrides[o]);
          createKrnl.memcpy(
              alloc, data, kI64, destOffset.getValue(), srcOffset.getValue());
        });
  }
};

void populateLoweringONNXGatherOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel,
    bool enableCollapse) {
  patterns.insert<ONNXGatherOpLowering>(
      typeConverter, ctx, enableParallel, enableCollapse);
}

} // namespace onnx_mlir
