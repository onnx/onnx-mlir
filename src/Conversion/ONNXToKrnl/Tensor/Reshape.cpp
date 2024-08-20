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
  DimAnalysis *dimAnalysis;

  ONNXReshapeOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, DimAnalysis *dimAnalysis)
      : OpConversionPattern(typeConverter, ctx), dimAnalysis(dimAnalysis) {}

  LogicalResult matchAndRewrite(ONNXReshapeOp reshapeOp,
      ONNXReshapeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = reshapeOp.getOperation();
    Location loc = ONNXLoc<ONNXReshapeOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value data = adaptor.getData();

    // If reshape does not change dimensions or it is an identity, just replace
    // the output with the input.
    if (isIdentityReshape(reshapeOp, dimAnalysis)) {
      LLVM_DEBUG(llvm::dbgs() << "Lowering reshape to identity\n");
      rewriter.replaceOp(op, data);
      return success();
    }

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);
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
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXReshapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, DimAnalysis *dimAnalysis) {
  patterns.insert<ONNXReshapeOpLowering>(typeConverter, ctx, dimAnalysis);
}

} // namespace onnx_mlir
