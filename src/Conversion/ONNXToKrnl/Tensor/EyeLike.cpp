/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ EyeLike.cpp - Lowering EyeLike Op ----------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX EyeLike Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXEyeLikeOpLowering : public OpConversionPattern<ONNXEyeLikeOp> {
  ONNXEyeLikeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXEyeLikeOp eyeLikeOp,
      ONNXEyeLikeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = eyeLikeOp.getOperation();
    Location loc = ONNXLoc<ONNXEyeLikeOp>(op);
    int64_t k = adaptor.getK();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);
    Type elementType = memRefType.getElementType();
    int64_t rank = memRefType.getRank();

    // Use the (unary) shape helper to get the output dims as IndexExprs
    // (output has the same shape as the input); the op's verifier already
    // guarantees rank == 2.
    ONNXEyeLikeOpShapeHelper shapeHelper(
        op, adaptor.getOperands(), &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr &outputDims = shapeHelper.getOutputDims();

    // Allocate memory for the output.
    Value alloc = create.mem.alignedAlloc(memRefType, outputDims);

    Value zero = create.math.constant(elementType, 0);
    Value one = create.math.constant(elementType, 1);
    Value kVal = create.math.constantIndex(k);

    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr, 2> lbs(rank, LitIE(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, outputDims,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
          // Value is 1 when col - row == k, 0 otherwise.
          Value diff = create.math.sub(loopInd[1], loopInd[0]);
          Value isDiag = create.math.eq(diff, kVal);
          Value result = create.math.select(isDiag, one, zero);
          createKrnl.store(result, alloc, loopInd);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXEyeLikeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXEyeLikeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
