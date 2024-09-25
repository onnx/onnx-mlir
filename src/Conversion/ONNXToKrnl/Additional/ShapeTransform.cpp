/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- ShapeTransform.cpp - Lowering ShapeTransform Op--------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNXShapeTransformOp to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXShapeTransformOpLowering : public ConversionPattern {
  ONNXShapeTransformOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXShapeTransformOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXShapeTransformOpAdaptor operandAdaptor(
        operands, op->getAttrDictionary());
    Value input = operandAdaptor.getInput();
    AffineMap indexMap = operandAdaptor.getIndexMap();

    // Helper builders.
    MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    // Get shape.
    ONNXShapeTransformOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Input and output types.
    MemRefType inputMemRefType = mlir::cast<MemRefType>(input.getType());
    MemRefType outputMemRefType = mlir::cast<MemRefType>(
        typeConverter->convertType(*op->result_type_begin()));
    uint64_t inputRank = inputMemRefType.getRank();
    uint64_t outputRank = outputMemRefType.getRank();

    // Allocate a buffer for the result MemRef.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Element-wise moving of data.
    ValueRange loopDef = create.krnl.defineLoops(inputRank);
    SmallVector<IndexExpr, 4> lbs(inputRank, LitIE(0));
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(input, ubs);

    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange inputIndices) {
          Value loadedVal = createKrnl.load(input, inputIndices);
          // Compute output indices by using affine map.
          SmallVector<Value, 4> outputIndices;
          for (uint64_t i = 0; i < outputRank; ++i) {
            AffineMap dimMap = indexMap.getSubMap(i);
            Value dimIndex = create.affine.apply(dimMap, inputIndices);
            outputIndices.emplace_back(dimIndex);
          }
          // Store result in the resulting array.
          createKrnl.store(loadedVal, alloc, outputIndices);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXShapeTransformOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXShapeTransformOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
