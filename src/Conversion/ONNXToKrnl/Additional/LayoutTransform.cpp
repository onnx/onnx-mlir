
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- LayoutTransform.cpp - Lowering Layout Transform Op --------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Layout Transform Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "layout-tranform"

using namespace mlir;

namespace onnx_mlir {

struct ONNXLayoutTransformOpLowering
    : public OpConversionPattern<ONNXLayoutTransformOp> {
  bool enableParallel = false;

  ONNXLayoutTransformOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx),
        enableParallel(enableParallel) {}

  LogicalResult matchAndRewrite(ONNXLayoutTransformOp layoutOp,
      ONNXLayoutTransformOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = layoutOp.getOperation();
    Location loc = ONNXLoc<ONNXLayoutTransformOp>(op);

    // Operands and attributes.
    Value data = adaptor.getData();

    // Convert the input type to MemRefType.
    Type inConvertedType = typeConverter->convertType(data.getType());
    assert(inConvertedType && inConvertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType inMemRefType = inConvertedType.cast<MemRefType>();
    // Convert the output type to MemRefType.
    Type outputTensorType = *op->result_type_begin();
    Type outConvertedType = typeConverter->convertType(outputTensorType);
    assert(outConvertedType && outConvertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outMemRefType = outConvertedType.cast<MemRefType>();

    // Note that by definition the input and output of LayoutTransformOp have
    // the same logical rank. The only difference between them should be their
    // layout. Currently defined layout may increase the dimensionality of the
    // mapped data by 1 or 2 dimensions compared to the original layout. Note
    // that these higher dimensionality will only manifest themselves once the
    // memref are normalized.
    uint64_t rank = inMemRefType.getShape().size();

    // Transform simply copy the input data to the output data. Both must have
    // the same logical size so use the input ones (arbitrary).
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope outerScope(create.krnl);
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(data, ubs);

    // Insert an allocation and deallocation for the result of this
    // operation.
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    Value alloc = create.mem.alignedAlloc(outMemRefType, ubs, alignment);

    // Insert loop over all inputs.
    ValueRange loopDef = create.krnl.defineLoops(rank);
    if (enableParallel) {
      create.krnl.parallel(loopDef);
      onnxToKrnlParallelReport(op, /*successful*/ true, rank, -1,
          "LayoutTransform op fully parallelized with perfectly nested loops");
    }
    SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange indices) {
          // Simply copy the input into the output.
          Value val = createKrnl.load(data, indices);
          createKrnl.store(val, alloc, indices);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXLayoutTransformOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXLayoutTransformOpLowering>(
      typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
