/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Reshape.cpp - Lowering Reshape Op -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reshape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "reshape_onnx_to_krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXReshapeOpLowering : public OpConversionPattern<ONNXReshapeOp> {
  ONNXReshapeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXReshapeOp reshapeOp,
      ONNXReshapeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = reshapeOp.getOperation();
    Location loc = ONNXLoc<ONNXReshapeOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value data = adaptor.getData();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();
    LLVM_DEBUG(llvm::dbgs() << "memRefType: " << memRefType << "\n");

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXReshapeOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Lower to ReinterpretCastOp so that the data is never copied or modified.
    Value newView = emitMemRefReinterpretCastOp(
        rewriter, loc, data, shapeHelper.getOutputDims(), convertedType);
    LLVM_DEBUG(llvm::dbgs() << "newView: " << newView << "\n");

    rewriter.replaceOp(op, newView);
    return success();
  }
};

void populateLoweringONNXReshapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
