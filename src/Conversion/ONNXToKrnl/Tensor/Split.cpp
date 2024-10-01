/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Split.cpp - Lowering Split Op -----------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Split Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <typename OP_TYPE, typename OP_ADAPTOR>
LogicalResult ONNXSplitOpLoweringCommon(OP_TYPE splitOp, OP_ADAPTOR adaptor,
    ConversionPatternRewriter &rewriter, const TypeConverter *typeConverter) {
  // Gather info.
  Operation *op = splitOp.getOperation();
  Location loc = ONNXLoc<OP_TYPE>(op);
  ValueRange operands = adaptor.getOperands();

  MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder>
      create(rewriter, loc);

  Value input = adaptor.getInput();
  uint64_t rank = create.krnlIE.getShapedTypeRank(input);
  unsigned outputNum = splitOp.getNumResults();
  unsigned axis = splitOp.getAxis();

  // Get shape.
  ONNXCommonSplitOpShapeHelper<OP_TYPE> shapeHelper(
      op, operands, &create.krnlIE);
  shapeHelper.computeShapeAndAssertOnFailure();

  // Alloc and dealloc.
  SmallVector<Value, 4> allocs;
  for (unsigned i = 0; i < outputNum; ++i) {
    // Convert the output type to MemRefType.
    Type convertedType =
        typeConverter->convertType(splitOp.getOutputs()[i].getType());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);
    Value alloc =
        create.mem.alignedAlloc(memRefType, shapeHelper.getOutputDims(i));
    allocs.emplace_back(alloc);
  }

  // Creates loops, one for each output.
  for (unsigned i = 0; i < outputNum; ++i) {
    OpBuilder::InsertionGuard insertGuard(rewriter);

    // Scope for krnl ops
    IndexExprScope childScope(&rewriter, shapeHelper.getScope());
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
        rewriter, loc);

    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));

    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(allocs[i], ubs);
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          SmallVector<IndexExpr, 4> readIndices;
          for (uint64_t r = 0; r < rank; ++r) {
            DimIndexExpr readIndex(indices[r]);
            // Compute read index for the split axis.
            if (r == axis)
              for (unsigned k = 0; k < i; ++k) {
                SymbolIndexExpr splitDim(shapeHelper.getOutputDims(k)[r]);
                readIndex = readIndex + splitDim;
              }

            readIndices.emplace_back(readIndex);
          }

          // Insert copy.
          Value loadData = createKrnl.loadIE(input, readIndices);
          createKrnl.store(loadData, allocs[i], indices);
        });
  }

  rewriter.replaceOp(op, allocs);
  onnxToKrnlSimdReport(op);
  return success();
}

struct ONNXSplitOpLowering : public OpConversionPattern<ONNXSplitOp> {
  ONNXSplitOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXSplitOp splitOp, ONNXSplitOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSplitOpLoweringCommon<ONNXSplitOp, ONNXSplitOpAdaptor>(
        splitOp, adaptor, rewriter, typeConverter);
  }
};

struct ONNXSplitV11OpLowering : public OpConversionPattern<ONNXSplitV11Op> {
  ONNXSplitV11OpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXSplitV11Op splitOp,
      ONNXSplitV11OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    return ONNXSplitOpLoweringCommon<ONNXSplitV11Op, ONNXSplitV11OpAdaptor>(
        splitOp, adaptor, rewriter, typeConverter);
  }
};

void populateLoweringONNXSplitOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLowering>(typeConverter, ctx);
}

void populateLoweringONNXSplitV11OpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSplitV11OpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
