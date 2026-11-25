/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ScatterND.cpp - Lowering ScatterND Op ----------------===//
//
// Copyright 2022-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ScatterND Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXScatterNDOpLowering : public OpConversionPattern<ONNXScatterNDOp> {
  ONNXScatterNDOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableSIMD, bool enableParallel, bool enableCollapse)
      : OpConversionPattern(typeConverter, ctx), enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXScatterNDOp::getOperationName());
    // Not and-ed with this->enableParallel: see Slice.cpp for why the two bools
    // stay independent.
    this->enableCollapse = enableCollapse;
  }
  bool enableSIMD;
  bool enableParallel;
  bool enableCollapse = false;

  // When `data` is Expand(scalar constant), e.g. a zero tensor shared by many
  // scatters, return that scalar as a value of the element type; null
  // otherwise. Looks at the original ONNX operand, not the converted one.
  static Value getFillValue(
      ONNXScatterNDOp scatterNDOp, Type elementType, MathBuilder &createMath) {
    auto expandOp = scatterNDOp.getData().getDefiningOp<ONNXExpandOp>();
    if (!expandOp || !isScalarConstantTensor(expandOp.getInput()))
      return nullptr;
    auto constOp = expandOp.getInput().getDefiningOp<ONNXConstantOp>();
    if (!constOp || !mlir::isa<FloatType, IntegerType>(elementType))
      return nullptr;
    return createMath.constant(elementType, getScalarValue<double>(constOp));
  }

  // Row form of the scatter: for each of the indices.shape[:-1] updates, load
  // its k index values once (normalized if they may be negative), then
  // combine its slice of S contiguous elements with the output, VL elements at
  // a time. Fails (emitting nothing) when the slice size is not static, k is
  // the rank of data (one element per update), or there is no SIMD.
  LogicalResult emitRows(ONNXScatterNDOp scatterNDOp, Operation *op,
      ONNXScatterNDOpAdaptor adaptor, Value output, DimsExpr &dataDims,
      MemRefType outputMemRefType, StringRef reduction,
      bool indicesMayBeNegative, ConversionPatternRewriter &rewriter,
      Location loc) const {
    Value updates = adaptor.getUpdates();
    Value indices = adaptor.getIndices();
    auto indicesType = mlir::cast<ShapedType>(indices.getType());
    int64_t dataRank = outputMemRefType.getRank();
    int64_t indicesRank = indicesType.getRank();
    int64_t k = indicesType.getShape()[indicesRank - 1];
    if (!enableSIMD || ShapedType::isDynamic(k) || k >= dataRank)
      return failure();
    Type elementType = outputMemRefType.getElementType();
    int64_t VL = VectorMachineSupport::getArchVectorLength(elementType);
    if (VL <= 1)
      return failure();
    int64_t S = 1;
    for (int64_t d = k; d < dataRank; ++d) {
      if (!dataDims[d].isLiteral())
        return failure();
      S *= dataDims[d].getLiteral();
    }
    if (S < 1)
      return failure();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);
    // Views with the slice flattened: output [data.shape[:k], S], updates
    // [indices.shape[:-1], S].
    int64_t sliceRank = dataRank - k;
    DimsExpr outFlatDims, updDims, updFlatDims;
    Value outFlat = create.mem.reshapeToFlatInnermost(
        output, dataDims, outFlatDims, sliceRank);
    create.krnlIE.getShapeAsDims(updates, updDims);
    Value updFlat = create.mem.reshapeToFlatInnermost(
        updates, updDims, updFlatDims, sliceRank);

    // One iteration per update slice. Parallel for reduction none only: with
    // a reduction, duplicate indices are allowed and two slices may combine
    // into the same output row, so rows stay in their original order.
    int64_t rowRank = indicesRank - 1;
    ValueRange loopDef = create.krnl.defineLoops(rowRank);
    DimsExpr lbs(rowRank, LitIE(0)), ubs;
    for (int64_t d = 0; d < rowRank; ++d)
      ubs.emplace_back(create.krnlIE.getShapeAsDim(indices, d));
    KrnlParallelPlan plan(loopDef, enableCollapse, /*parFirstInclusiveDim=*/0,
        /*parLastExclusiveDim=*/rowRank, /*collapseLastExclusiveDim=*/rowRank,
        {.minTripCount = 4,
            .bodyCost = std::max<int64_t>(10, S + 3 * k),
            .mustRepayFork = true});
    // No reduction: guaranteed to have distinct output rows. If actual
    // reduction, we may have duplicate output rows, so the nest stays
    // sequential.
    if (enableParallel && reduction == "none")
      plan.tryCreateParallel(create.krnl, op, "scatterND rows", lbs, ubs);
    bool fullySimd = (S % VL == 0);
    onnxToKrnlSimdReport(op, /*successful*/ true, VL, S,
        "scatterND rows with one index load per row");
    create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange rowInd) {
          IndexExprScope innerScope(createKrnl);
          DimsExpr rowAF;
          getIndexExprList<DimIndexExpr>(rowInd, rowAF);
          // Output access: the k index values, then offset 0 in the slice.
          DimsExpr outAF;
          for (int64_t i = 0; i < k; ++i) {
            DimsExpr indicesAF(rowAF);
            indicesAF.emplace_back(LitIE(i));
            Value indexVal = createKrnl.loadIE(indices, indicesAF);
            IndexExpr index = NonAffineIndexExpr(indexVal);
            if (indicesMayBeNegative)
              index = index.selectOrSelf(
                  index < LitIE(0), index + SymIE(dataDims[i]));
            outAF.emplace_back(index);
          }
          outAF.emplace_back(LitIE(0));
          DimsExpr updAF(rowAF);
          updAF.emplace_back(LitIE(0));
          createKrnl.simdIterateIE(LitIE(0), LitIE(S), VL, fullySimd,
              /*useParallel=*/false, {updFlat, outFlat}, {updAF, outAF},
              {outFlat}, {outAF},
              {[&](const KrnlBuilder &b, ArrayRef<Value> vals,
                   int64_t) -> Value {
                MathBuilder createMath(b);
                Value upd = vals[0], cur = vals[1];
                if (reduction == "add")
                  return createMath.add(cur, upd);
                if (reduction == "mul")
                  return createMath.mul(cur, upd);
                if (reduction == "max")
                  return createMath.max(cur, upd);
                if (reduction == "min")
                  return createMath.min(cur, upd);
                assert(reduction == "none" && "Unknown reduction type");
                return upd;
              }});
        });
    return success();
  }

  LogicalResult matchAndRewrite(ONNXScatterNDOp scatterNDOp,
      ONNXScatterNDOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = scatterNDOp.getOperation();
    Location loc = ONNXLoc<ONNXScatterNDOp>(op);

    // Operands and attributes.
    Value data = adaptor.getData();
    Value updates = adaptor.getUpdates();
    Value indices = adaptor.getIndices();
    auto dataType = mlir::cast<ShapedType>(data.getType());
    auto indicesType = mlir::cast<ShapedType>(indices.getType());
    auto updatesType = mlir::cast<ShapedType>(updates.getType());
    int64_t dataRank = dataType.getRank();
    int64_t updatesRank = updatesType.getRank();
    int64_t indicesRank = indicesType.getRank();

    assert(dataRank >= 1 && "The rank of 'data' must be >= 1");
    assert(indicesRank >= 1 && "The rank of 'indices' must be >= 1");
    StringRef reduction = adaptor.getReduction();

    // Determine whether indices may be negative.
    bool indicesMayBeNegative = !indicesAreNonNegativeConstants(indices);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getShape().size();
    assert(outputRank == dataRank && "Output rank not equal to data rank");

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);
    IndexExprScope indexScope(create.krnl);
    DimsExpr dataDims;
    create.krnlIE.getShapeAsDims(data, dataDims);

    // Step1: make `output` hold the values of `data`, either by scattering
    // into the buffer of `data` itself, or by copying `data` into a new one.
    Value output;
    if (canWriteInPlace(rewriter, op, /*operandIndex=*/0, outputMemRefType)) {
      output = data;
    } else {
      output = create.mem.alignedAlloc(outputMemRefType, dataDims);
      // In development (--test-compiler-opt): when `data` is a broadcast
      // scalar, fill the output with it instead of copying the shared buffer.
      // The fill only writes; the shared buffer, if no one else reads it, is
      // then removed as write-only after lowering.
      Value fillVal = debugTestCompilerOpt
                          ? getFillValue(scatterNDOp,
                                outputMemRefType.getElementType(), create.math)
                          : nullptr;
      if (fillVal)
        create.krnl.memset(output, fillVal);
      else
        emitMemcpy(rewriter, loc, op, output, data, dataDims, enableParallel,
            "scatterND copy");
    }

    // In development (--test-compiler-opt): the row form. With k =
    // indices.shape[-1] < rank(data), each update is a whole slice of S =
    // prod(data.shape[k:]) contiguous elements, in both `updates` and the
    // output. Load the k index values once per slice, then combine the slice
    // with SIMD.
    if (debugTestCompilerOpt &&
        succeeded(emitRows(scatterNDOp, op, adaptor, output, dataDims,
            outputMemRefType, reduction, indicesMayBeNegative, rewriter,
            loc))) {
      rewriter.replaceOp(op, output);
      return success();
    }

    // Step2: scatter the updates values into the output.
    //   update_indices = indices.shape[:-1]
    //   for idx in np.ndindex(update_indices):
    //     output[indices[idx]] = updates[idx]
    //
    ValueRange loopDef = create.krnl.defineLoops(updatesRank);
    DimsExpr lbs(updatesRank, LitIE(0)), ubs;
    create.krnlIE.getShapeAsDims(updates, ubs);
    // Enable parallelism if required, for reduction "none" only.
    // With no reduction the spec requires indices to have no duplicate
    // entries, so distinct iterations store to distinct output elements and
    // the whole nest is order-independent: the collapse claim can be the whole
    // window. With a reduction, duplicate indices are allowed and two
    // iterations of the leading (q-1) loops may read-modify-write the same
    // output element, so the nest stays sequential.
    //
    // bodyCost 2 + 3 k (at least 10): one innermost iteration loads k index
    // values and normalizes each with a compare and a select, loads the
    // update, and stores through a non-affine access function. k, the index
    // depth, is the static last dimension of 'indices' (asserted below).
    bool parallelizable = (reduction == "none");
    int64_t indexDepthForCost = indicesType.getShape()[indicesRank - 1];
    int64_t bodyCost = ShapedType::isDynamic(indexDepthForCost)
                           ? 10
                           : std::max<int64_t>(10, 2 + 3 * indexDepthForCost);
    KrnlParallelPlan plan(loopDef, enableCollapse, /*parFirstInclusiveDim=*/0,
        /*parLastExclusiveDim=*/updatesRank,
        /*collapseLastExclusiveDim=*/updatesRank,
        {.minTripCount = 4, .bodyCost = bodyCost, .mustRepayFork = true});
    if (enableParallel && parallelizable)
      plan.tryCreateParallel(create.krnl, op, "scatterND", lbs, ubs);
    create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for 'indices'. Let q = rank(indices).
          // The first (q-1) indexes traverse the iteration space defined by
          // indices.shape[:-1], which corresponds to the first (q-1) induction
          // variable in the loop iteration space.
          DimsExpr indicesAccessFctFirst;
          getIndexExprList<DimIndexExpr>(loopInd, indicesAccessFctFirst);
          indicesAccessFctFirst.truncate(indicesRank - 1);

          // Access function for the output. Let r=rank(data), q=rank(indices),
          // k=indices.shape[-1]. The first k indexes are given by looking up
          // the 'indices' tensor. The remaining (r-k) indexes are given by
          // the trailing (r-k) induction variables of the loop iteration
          // space, i.e. those past the first (q-1) that were consumed above
          // by the 'indices' access function.
          int64_t indexDepth = indicesType.getShape()[indicesRank - 1];
          assert(indexDepth != ShapedType::kDynamic &&
                 "indices tensor's last dimension must be static");
          DimsExpr outputAccessFct;
          for (int64_t i = 0; i < dataRank; ++i) {
            if (i < indexDepth) {
              IndexExpr ind = LitIE(i);
              DimsExpr indicesAccessFct(indicesAccessFctFirst);
              indicesAccessFct.emplace_back(ind);
              Value indexVal = createKrnl.loadIE(indices, indicesAccessFct);
              IndexExpr index = NonAffineIndexExpr(indexVal);
              // Index values are allowed to be negative, counting dimensions
              // from the back; normalize by adding the data dim in that case.
              if (indicesMayBeNegative)
                index = index.selectOrSelf(
                    index < LitIE(0), index + SymIE(dataDims[i]));
              outputAccessFct.emplace_back(index);
            } else {
              int64_t loopIdx = (indicesRank - 1) + (i - indexDepth);
              IndexExpr index = SymIE(loopInd[loopIdx]);
              outputAccessFct.emplace_back(index);
            }
          }

          // Scatter 'update' values into the output tensor with the specified
          // reduction.
          Value updateVal = createKrnl.load(updates, loopInd);
          Value result = updateVal;
          if (reduction == "add") {
            Value current = createKrnl.loadIE(output, outputAccessFct);
            result = create.math.add(current, updateVal);
          } else if (reduction == "mul") {
            Value current = createKrnl.loadIE(output, outputAccessFct);
            result = create.math.mul(current, updateVal);
          } else if (reduction == "max") {
            Value current = createKrnl.loadIE(output, outputAccessFct);
            result = create.math.max(current, updateVal);
          } else if (reduction == "min") {
            Value current = createKrnl.loadIE(output, outputAccessFct);
            result = create.math.min(current, updateVal);
          } else if (reduction != "none") {
            llvm_unreachable("Unknown reduction type");
          }
          createKrnl.storeIE(result, output, outputAccessFct);
        });

    rewriter.replaceOp(op, output);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXScatterNDOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD,
    bool enableParallel, bool enableCollapse) {
  patterns.insert<ONNXScatterNDOpLowering>(
      typeConverter, ctx, enableSIMD, enableParallel, enableCollapse);
}

} // namespace onnx_mlir
