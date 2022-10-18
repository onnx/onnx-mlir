
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- LayoutTransform.cpp - Lowering Layout Transform Op --------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Layout Transform Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXLayoutTransformOpLowering : public ConversionPattern {
  ONNXLayoutTransformOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXLayoutTransformOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXLayoutTransformOpAdaptor operandAdaptor(operands);
    ONNXLayoutTransformOp LayoutTransformOp =
        llvm::cast<ONNXLayoutTransformOp>(op);
    auto loc = op->getLoc();

    // Operands and attributes.
    Value data = operandAdaptor.data();

    // Convert the input type to MemRefType.
    Type inConvertedType = typeConverter->convertType(data.getType());
    assert(inConvertedType && inConvertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType inMemRefType = inConvertedType.cast<MemRefType>();
    uint64_t rank = inMemRefType.getShape().size();
    // Convert the output type to MemRefType.
    Type outConvertedType =
        typeConverter->convertType(*op->result_type_begin());
    assert(outConvertedType && outConvertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outMemRefType = outConvertedType.cast<MemRefType>();

    // Transform simply copy the input data to the output data. Both must have
    // the same logical size.
    KrnlBuilder createKrnl(rewriter, loc);
    IndexExprScope outerScope(createKrnl);
    MemRefBoundsIndexCapture dataBounds(data);
    SmallVector<IndexExpr, 4> ubs;
    dataBounds.getDimList(ubs);

    // Insert an allocation and deallocation for the result of this
    // operation.
    Value alloc =
        insertAllocAndDeallocSimple(rewriter, op, outMemRefType, loc, ubs);

    // Insert loop over all inputs.
    ValueRange loopDef = createKrnl.defineLoops(rank);
    SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
    createKrnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange indices) {
          // Simply copy the input into the output.
          Value val = createKrnl.load(data, indices);
          createKrnl.store(val, alloc, indices);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXLayoutTransformOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLayoutTransformOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
