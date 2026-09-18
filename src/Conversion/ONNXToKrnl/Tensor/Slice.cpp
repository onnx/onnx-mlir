/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Slice.cpp - Lowering Slice Op ----------------------=== //
//
// Copyright 2020-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Slice Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSliceOpLowering : public OpConversionPattern<ONNXSliceOp> {
  ONNXSliceOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableParallel, bool enableCollapse)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXSliceOp::getOperationName());
    // Not and-ed with this->enableParallel: a collapse-eligible plan is built
    // either way, and it is tryCreateParallel -- called only when parallelism
    // is on -- that can act on it. Keeping the two bools independent is what
    // makes the plan construction one unconditional statement.
    this->enableCollapse = enableCollapse;
  }

  LogicalResult matchAndRewrite(ONNXSliceOp sliceOp, ONNXSliceOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = sliceOp.getOperation();
    Location loc = ONNXLoc<ONNXSliceOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    ONNXSliceOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getShape().size();

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    ValueRange loopDef = create.krnl.defineLoops(outputRank);
    SmallVector<IndexExpr, 4> lbs(outputRank, LitIE(0));
    DimsExpr ubs = shapeHelper.getOutputDims();

    // Enable parallelism if required.
    // bodyCost 1: one innermost iteration is a strided load and a store.
    KrnlParallelPlan plan(loopDef, enableCollapse, /*parFirstInclusiveDim=*/0,
        /*parLastExclusiveDim=*/2, /*collapseLastExclusiveDim=*/2,
        {.minTripCountForParallel = 4, .bodyCost = 1});
    if (enableParallel)
      plan.tryCreateParallel(create.krnl, op, "slice", lbs, ubs);

    create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          IndexExprScope loopScope(createKrnl);

          // Compute indices for the load and store op.
          // Load: "i * step + start" for all dim.
          // Store: "i" for all dims.
          SmallVector<IndexExpr, 4> loadIndices, storeIndices;
          for (int ii = 0; ii < outputRank; ++ii) {
            DimIndexExpr inductionIndex(loopInd[ii]);
            IndexExpr start = SymIE(shapeHelper.starts[ii]);
            IndexExpr step = SymIE(shapeHelper.steps[ii]);
            loadIndices.emplace_back((step * inductionIndex) + start);
            storeIndices.emplace_back(inductionIndex);
          }
          // Load data and store in alloc data.
          Value loadVal = createKrnl.loadIE(adaptor.getData(), loadIndices);
          createKrnl.storeIE(loadVal, alloc, storeIndices);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }

private:
  bool enableParallel = false;
  bool enableCollapse = false;
};

void populateLoweringONNXSliceOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel,
    bool enableCollapse) {
  patterns.insert<ONNXSliceOpLowering>(
      typeConverter, ctx, enableParallel, enableCollapse);
}

} // namespace onnx_mlir
