/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ScatterND.cpp - Lowering ScatterND Op ----------------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ScatterND Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXScatterNDOpLowering : public OpConversionPattern<ONNXScatterNDOp> {
  ONNXScatterNDOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableParallel, bool enableCollapse)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXScatterNDOp::getOperationName());
    // Not and-ed with this->enableParallel: see Slice.cpp for why the two bools
    // stay independent.
    this->enableCollapse = enableCollapse;
  }
  bool enableParallel;
  bool enableCollapse = false;

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
      emitMemcpy(rewriter, loc, op, output, data, dataDims, enableParallel,
          "scatterND copy");
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
    // bodyCost 10: one innermost iteration loads k index values, optionally
    // normalizes each with a compare and a select, loads the update, and
    // stores through a non-affine access function.
    bool parallelizable = (reduction == "none");
    KrnlParallelPlan plan(loopDef, enableCollapse, /*parFirstInclusiveDim=*/0,
        /*parLastExclusiveDim=*/updatesRank,
        /*collapseLastExclusiveDim=*/updatesRank,
        {.minTripCountForParallel = 4, .bodyCost = 10});
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
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel,
    bool enableCollapse) {
  patterns.insert<ONNXScatterNDOpLowering>(
      typeConverter, ctx, enableParallel, enableCollapse);
}

} // namespace onnx_mlir
