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

    // First message.
    std::string opName(printSignatureOp.getOpName().data());
    std::string msg =
        "%i==SIG-REPORT==, " + opName + ", sig"; // meaningless secondary key.
    // Discover the values to print, setting aside the last one.
    llvm::SmallVector<Value, 4> printVal;
    for (Value oper : adaptor.getInput())
      if (!oper.getType().isa<NoneType>())
        printVal.emplace_back(oper);
    int64_t printNum = printVal.size();
    Value lastVal = printVal.pop_back_val();
    // Print all but the last one.
    for (Value oper : printVal) {
      create.krnl.printTensor(msg + ", %t%e", oper);
      msg = "%i";
    }
    // Print the last one with replace with new op.
    if (printNum > 0) {
      rewriter.replaceOpWithNewOp<KrnlPrintTensorOp>(
          op, msg + ", %t\n%e", lastVal);
    } else {
      Value noneVal = nullptr;
      rewriter.replaceOpWithNewOp<KrnlPrintTensorOp>(op, msg + "\n%e", noneVal);
    }
    return success();
  }
};

void populateLoweringONNXPrintSignaturePattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXPrintSignatureLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
