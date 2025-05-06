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
      if (!mlir::isa<NoneType>(oper.getType()))
        printVal.emplace_back(oper);
    int64_t printNum = printVal.size();
    if (printNum == 0) {
      // Print tensor without any valid tensor.
      Value noneVal = nullptr;
      rewriter.replaceOpWithNewOp<KrnlPrintOp>(
          op, msg + "(no tensors)\n%e", noneVal);
      return success();
    }
    // Control how the tensor will be printed
    // Print the only the shape.
    std::string printControl = ", %t%e";
    if (printSignatureOp.getPrintData() == 1) {
      // The data of tensor will be printed
      printControl = "%t%d\n";
      msg += "\n";
    }
    Value lastVal = printVal.pop_back_val();
    // Print all but the last one.
    for (Value oper : printVal) {
      create.krnl.printTensor(msg + printControl, oper);
      msg = "%i";
    }
    // Print the last one with replace with new op.
    if (printSignatureOp.getPrintData() == 0) {
      printControl = ", %t\n%e";
    }
    rewriter.replaceOpWithNewOp<KrnlPrintTensorOp>(
        op, msg + printControl, lastVal);
    return success();
  }
};

void populateLoweringONNXPrintSignaturePattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXPrintSignatureLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
