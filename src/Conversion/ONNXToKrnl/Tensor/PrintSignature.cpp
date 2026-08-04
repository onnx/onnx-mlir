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
    // Per-input label (e.g. "in0", "out1"), only used when io_labels is
    // present and aligned 1:1 with (pre-filtering) getInput(), so a
    // caller-selected subset of operands/results can still be told apart in
    // the printed output. Kept in lockstep with printVal below, since a
    // NoneType operand is dropped from printVal but must also drop its label.
    ArrayAttr labelsAttr = printSignatureOp.getIoLabelsAttr();
    bool hasLabels =
        labelsAttr && labelsAttr.size() == adaptor.getInput().size();
    // Discover the values to print, setting aside the last one.
    llvm::SmallVector<Value, 4> printVal;
    llvm::SmallVector<std::string, 4> printLabel;
    for (auto it : llvm::enumerate(adaptor.getInput())) {
      Value oper = it.value();
      if (mlir::isa<NoneType>(oper.getType()))
        continue;
      printVal.emplace_back(oper);
      printLabel.emplace_back(
          hasLabels
              ? mlir::cast<StringAttr>(labelsAttr[it.index()]).getValue().str()
              : std::string());
    }
    int64_t printNum = printVal.size();
    if (printNum == 0) {
      // Print tensor without any valid tensor.
      Value noneVal = nullptr;
      rewriter.replaceOpWithNewOp<KrnlPrintOp>(
          op, msg + "(no tensors)\n%e", noneVal);
      return success();
    }
    // Prefix a tensor's format string with its label, e.g. "in0, ", when one
    // was given; otherwise leave the format string untouched.
    auto withLabel = [](const std::string &label, const std::string &fmt) {
      return label.empty() ? fmt : label + ", " + fmt;
    };
    // Control how the tensor will be printed
    // Print the only the shape.
    std::string printControl = ", %t%e";
    if (printSignatureOp.getPrintData() == 1) {
      // The data of tensor will be printed
      printControl = "%t%d\n";
      msg += "\n";
    }
    Value lastVal = printVal.pop_back_val();
    std::string lastLabel = printLabel.pop_back_val();
    // Print all but the last one.
    for (size_t i = 0; i < printVal.size(); ++i) {
      create.krnl.printTensor(
          msg + withLabel(printLabel[i], printControl), printVal[i]);
      msg = "%i";
    }
    // Print the last one with replace with new op.
    if (printSignatureOp.getPrintData() == 0) {
      printControl = ", %t\n%e";
    }
    rewriter.replaceOpWithNewOp<KrnlPrintTensorOp>(
        op, msg + withLabel(lastLabel, printControl), lastVal);
    return success();
  }
};

void populateLoweringONNXPrintSignaturePattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXPrintSignatureLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
