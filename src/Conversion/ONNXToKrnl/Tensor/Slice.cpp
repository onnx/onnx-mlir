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

struct ONNXSliceOpLowering : public ConversionPattern {
  ONNXSliceOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSliceOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSliceOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    ONNXSliceOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    int64_t outputRank = outputMemRefType.getShape().size();

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    ValueRange loopDef = create.krnl.defineLoops(outputRank);
    SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.getOutputDims(),
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          IndexExprScope loopScope(createKrnl);

          // Compute indices for the load and store op.
          // Load: "i * step + start" for all dim.
          // Store: "i" for all dims.
          SmallVector<IndexExpr, 4> loadIndices, storeIndices;
          for (int ii = 0; ii < outputRank; ++ii) {
            DimIndexExpr inductionIndex(loopInd[ii]);
            IndexExpr start = SymbolIndexExpr(shapeHelper.starts[ii]);
            IndexExpr step = SymbolIndexExpr(shapeHelper.steps[ii]);
            loadIndices.emplace_back((step * inductionIndex) + start);
            storeIndices.emplace_back(inductionIndex);
          }
          // Load data and store in alloc data.
          Value loadVal =
              createKrnl.loadIE(operandAdaptor.getData(), loadIndices);
          createKrnl.storeIE(loadVal, alloc, storeIndices);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSliceOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSliceOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
