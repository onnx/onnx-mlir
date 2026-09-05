/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ EyeLike.cpp - Lowering EyeLike Op ----------------===//
//
// This file lowers the ONNX EyeLike Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

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
    Value input = adaptor.getInput();
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
    ArrayRef<int64_t> outputShape = memRefType.getShape();
    int64_t rank = outputShape.size();
    assert(rank == 2 && "EyeLike only supports 2D tensors");

    // Allocate memory for the output. Output has the same shape as input.
    Value alloc;
    if (hasAllConstantDimensions(memRefType))
      alloc = create.mem.alignedAlloc(memRefType);
    else {
      SmallVector<Value, 2> allocOperands;
      for (int64_t i = 0; i < rank; ++i) {
        if (outputShape[i] == ShapedType::kDynamic) {
          Value dim = create.mem.dim(input, i);
          allocOperands.emplace_back(dim);
        }
      }
      alloc = create.mem.alignedAlloc(memRefType, allocOperands);
    }

    Value zero = create.math.constant(elementType, 0);
    Value one = create.math.constant(elementType, 1);
    Value kVal = create.math.constantIndex(k);

    IndexExprScope ieScope(create.krnl);
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr, 2> lbs(rank, LitIE(0));
    SmallVector<IndexExpr, 2> ubs;

    create.krnlIE.getShapeAsDims(alloc, ubs);
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
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
