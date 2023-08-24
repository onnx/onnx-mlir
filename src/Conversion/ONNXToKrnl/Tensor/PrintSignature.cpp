/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Print Signature Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXPrintSignatureLowering
    : public OpConversionPattern<ONNXPrintSignatureOp> {
  ONNXPrintSignatureLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXPrintSignatureOp printSignatureOp,
      ONNXPrintSignatureOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Operation *op = printSignatureOp.getOperation();
    Location loc = ONNXLoc<ONNXPrintSignatureOp>(op);
    MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);

    std::string opName(printSignatureOp.getOpName().data());
    std::string str = "==SIGNATURE==, " + opName;
    create.krnl.printf(str);
    std::string msg = ", %t";
    for (Value oper : adaptor.getInput())
      if (!oper.getType().isa<NoneType>())
        create.krnl.printTensor(msg, oper);
    Value noneValue;
    rewriter.replaceOpWithNewOp<KrnlPrintOp>(op, "\n", noneValue);
    // For debug, no need to report on SIMD.
    return success();
  }
};

void populateLoweringONNXPrintSignaturePattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXPrintSignatureLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
