//===---------------- Constant.cpp - Lowering Constant Op -----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Constant Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXConstantOpLowering : public ConversionPattern {
  static int constantID;

  ONNXConstantOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConstantOp::getOperationName(), 1, ctx) {
        constantID = 0;
      }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto constantOp = llvm::dyn_cast<ONNXConstantOp>(op);

    if (constantOp.sparse_value().hasValue()) {
      emitError(loc, "Only support dense values at this time");
    }

    auto memRefType = convertToMemRefType(*op->result_type_begin());

    // Shape based computations.
    auto shape = memRefType.getShape();
    int64_t numElements = 1;
    for (int i=0; i<shape.size(); ++i)
      numElements *= shape[i];

    // Emit the constant global in Krnl dialect.
    auto constantGlobal = rewriter.create<KrnlGlobalOp>(loc,
        memRefType,
        rewriter.getI64ArrayAttr(shape),
        constantOp.value().getValue(),
        rewriter.getStringAttr("constant_" + std::to_string(constantID)));

    // Increment constant ID:
    constantID++;

    // Replace this operation with the generated alloc.
    // rewriter.replaceOp(op, alloc);
    rewriter.replaceOp(op, constantGlobal.getResult());

    return success();
  }
};

int ONNXConstantOpLowering::constantID;

void populateLoweringONNXConstantOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOpLowering>(ctx);
}
