/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Slice.cpp - Lowering Slice Op ----------------------=== //
//
// Copyright 2020-2023 The IBM Research Authors.
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
  ONNXSliceOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

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
    create.krnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.getOutputDims(),
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
};

void populateLoweringONNXSliceOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSliceOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
