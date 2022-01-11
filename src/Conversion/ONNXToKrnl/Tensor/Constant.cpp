/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Constant.cpp - Lowering Constant Op -----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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

  ONNXConstantOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXConstantOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = ONNXLoc<ONNXConstantOp>(op);
    auto constantOp = llvm::dyn_cast<ONNXConstantOp>(op);
    assert(constantOp && "Op does not have type ONNXConstantOp");

    if (constantOp.sparse_value().hasValue())
      return emitError(loc, "Only support dense values at this time");

    auto memRefType = convertToMemRefType(*op->result_type_begin());

    // Shape based computations.
    auto shape = memRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < shape.size(); ++i)
      numElements *= shape[i];

    // Emit the constant global in Krnl dialect.
    auto constantGlobal = rewriter.create<KrnlGlobalOp>(loc, memRefType,
        /*shape=*/rewriter.getI64ArrayAttr(shape),
        /*name=*/
        rewriter.getStringAttr("constant_" + std::to_string(constantID)),
        /*value=*/constantOp.value().getValue(),
        /*offset=*/nullptr,
        /*alignment=*/nullptr);

    // Increment constant ID:
    constantID++;

    // Replace this operation with the generated krnl.global.
    rewriter.replaceOp(op, constantGlobal.getResult());

    return success();
  }
};

int ONNXConstantOpLowering::constantID = 0;

void populateLoweringONNXConstantOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOpLowering>(typeConverter, ctx);
}
