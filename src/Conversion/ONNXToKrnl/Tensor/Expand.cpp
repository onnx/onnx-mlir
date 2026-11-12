/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Expand.cpp - Lowering Expand Op----------------------=== //
//
// Copyright 2020-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Expand Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXExpandOpLowering : public OpConversionPattern<ONNXExpandOp> {
  ONNXExpandOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableParallel, bool enableCollapse)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXExpandOp::getOperationName());
    // Not and-ed with this->enableParallel: see Slice.cpp for why the two bools
    // stay independent.
    this->enableCollapse = enableCollapse;
  }

  LogicalResult matchAndRewrite(ONNXExpandOp expandOp,
      ONNXExpandOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = expandOp.getOperation();
    Location loc = ONNXLoc<ONNXExpandOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value input = adaptor.getInput();

    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXExpandOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getRank();

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Iterate over the output values.
    ValueRange outputLoopDef = create.krnl.defineLoops(outputRank);
    LiteralIndexExpr zeroIE(0);
    SmallVector<IndexExpr, 4> lbs(outputRank, zeroIE);
    DimsExpr ubs = shapeHelper.getOutputDims();

    // Enable parallelism if required.
    // The nest is order-independent all the way down -- every output element is
    // written once and reads only from the input -- so the collapse claim may
    // safely be the whole search window.
    // bodyCost 1: one innermost iteration is a load and a store, or just a
    // store when the input is a scalar hoisted out of the nest.
    //
    // Window: the top two levels, as before, widened to three for rank 4 and
    // up (still leaving out the innermost level). Two levels suffice for a
    // [B, H, ...] tensor, but an attention mask expanded to [B, 1, S, S] at
    // batch 1 has two literal-1 levels on top, and a two-level window then
    // finds nothing to parallelize. Reaching level 2 lets the plan put the
    // region on S at batch 1, or fuse B*1*S when B is dynamic. Ranks 2 and 3
    // keep their two-level window unchanged.
    int64_t parLevels =
        std::max((int64_t)2, std::min(outputRank - 1, (int64_t)3));
    KrnlParallelPlan plan(outputLoopDef, enableCollapse,
        /*parFirstInclusiveDim=*/0, /*parLastExclusiveDim=*/parLevels,
        /*collapseLastExclusiveDim=*/parLevels,
        {.minTripCountForParallel = 4, .bodyCost = 1});
    if (enableParallel)
      plan.tryCreateParallel(create.krnl, op, "expand", lbs, ubs);

    // If input is a scalar, load its value outside the loop.
    Value val = nullptr;
    if (isScalarTensor(input))
      val = create.krnl.load(input);

    create.krnl.iterateIE(outputLoopDef, plan.optimizedLoopDef(), lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange outputLoopInd) {
          if (!val) {
            IndexExprScope outputScope(createKrnl, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputLoopIndices, lhsAccessExprs;
            getIndexExprList<DimIndexExpr>(outputLoopInd, outputLoopIndices);
            LogicalResult res = shapeHelper.getAccessExprs(
                input, 0, outputLoopIndices, lhsAccessExprs);
            assert(succeeded(res) && "Could not compute access indices");
            val = createKrnl.loadIE(input, lhsAccessExprs);
          }
          createKrnl.store(val, alloc, outputLoopInd);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }

private:
  bool enableParallel = false;
  bool enableCollapse = false;
};

void populateLoweringONNXExpandOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel,
    bool enableCollapse) {
  patterns.insert<ONNXExpandOpLowering>(
      typeConverter, ctx, enableParallel, enableCollapse);
}

} // namespace onnx_mlir
