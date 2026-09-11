/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Concat Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConcatOpLowering : public OpConversionPattern<ONNXConcatOp> {
  ONNXConcatOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableParallel, bool enableCollapse)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXConcatOp::getOperationName());
    // Not and-ed with this->enableParallel: see Slice.cpp for why the two bools
    // stay independent.
    this->enableCollapse = enableCollapse;
  }

  bool enableParallel = false;
  bool enableCollapse = false;

  LogicalResult matchAndRewrite(ONNXConcatOp concatOp,
      ONNXConcatOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = concatOp.getOperation();
    Location loc = ONNXLoc<ONNXConcatOp>(op);
    ValueRange operands = adaptor.getOperands();

    // Gather info.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXConcatOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    auto axis = concatOp.getAxis();
    assert(axis >= 0 && "negative axis is supposed to have been normalized");
    unsigned int inputNum = operands.size();

    // Convert the output type to MemRefType.
    Type outputTensorType = *op->result_type_begin();
    Type convertedType = typeConverter->convertType(outputTensorType);
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    auto resultShape = outputMemRefType.getShape();
    unsigned int rank = resultShape.size();

    // Alloc and dealloc.
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), alignment);

    // Creates loops, one for each input.
    // Since the each input should have same size for each dimension(except
    // axis), we will try to make the loop upper bound the same for further
    // optimization. Difference may come from constant vs. dynamic, or dynamic
    // dim of different inputs.
    SmallVector<IndexExpr, 4> commonUB(shapeHelper.getOutputDims());
    // IndexExprScope IEScope(&rewriter, loc);
    IndexExpr accumulatedOffset = LitIE(0);
    for (unsigned int i = 0; i < inputNum; ++i) {
      // Since the accumulatedOffsetValue will be used in a nested
      // IndexExprScope, we get the Value of this IndexExpr and pass it as a
      // symbol
      Value accumulatedOffsetValue = accumulatedOffset.getValue();
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Create loop.
      ValueRange loopDef = create.krnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));
      // For each input, only the dimension 'axis' is different, so all the
      // other dims keep the output's (possibly literal, and shared) values.
      IndexExpr axisDim = create.krnlIE.getShapeAsDim(operands[i], axis);
      commonUB[axis] = axisDim;

      // Explore the first two outermost dims, giving up if the found one is
      // 'axis'. Plan is per-input like loopDef, so no ref leaks between inputs
      // and the destructor's consumed-check fires once per input.
      //
      // The collapse claim is the whole window minus 'axis', which is what the
      // exclusion already expresses: away from 'axis' every input element lands
      // at one output element, so the levels are individually parallel and safe
      // to fuse. When 'axis' is 0 or 1 the exclusion leaves no run of two
      // adjacent safe levels, so the frame quick-exits to STEP 0 and the IR is
      // unchanged -- collapse only ever engages here for axis >= 2.
      // bodyCost 1: one innermost iteration is a load and a store, plus an
      // offset add on the axis level.
      KrnlParallelPlan plan(loopDef, enableCollapse,
          /*parFirstInclusiveDim=*/0, /*parLastExclusiveDim=*/2,
          /*collapseLastExclusiveDim=*/2,
          {.minTripCountForParallel = 4, .bodyCost = 1},
          /*excl dims*/ {axis});
      if (enableParallel)
        plan.tryCreateParallel(create.krnl, op, "concat", lbs, commonUB);

      create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), lbs, commonUB,
          [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
            // Indices for the read and write.
            SmallVector<Value, 4> writeIndices;
            for (unsigned int r = 0; r < rank; ++r) {
              if (r != axis || i == 0)
                writeIndices.emplace_back(loopInd[r]);
              else {
                IndexExprScope IEScope(&rewriter, loc);
                IndexExpr writeOffset = DimIE(loopInd[r]);
                IndexExpr accumulatedOffsetIE = SymIE(accumulatedOffsetValue);
                writeOffset = writeOffset + accumulatedOffsetIE;
                writeIndices.emplace_back(writeOffset.getValue());
              }
            }
            // Insert copy.
            Value loadData = createKrnl.load(operands[i], loopInd);
            createKrnl.store(loadData, alloc, writeIndices);
          });
      accumulatedOffset = accumulatedOffset + axisDim;
    }
    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXConcatOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel,
    bool enableCollapse) {
  patterns.insert<ONNXConcatOpLowering>(
      typeConverter, ctx, enableParallel, enableCollapse);
}

} // namespace onnx_mlir
