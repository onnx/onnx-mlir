/*
 * SPDX-License-Identifier: Apache-2.0
 */

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

bool checkOpResultIsReturned(ONNXConstantOp *constantOp) {
  FuncOp function = getContainingFunction(constantOp->getOperation());

  bool opIsReturned = false;
  function.walk([&opIsReturned, constantOp](ReturnOp op) {
    auto result = constantOp->getResult();
    for (const auto &operand : op.getOperands())
      if (operand == result)
        opIsReturned = true;
  });

  return opIsReturned;
}

struct ONNXConstantOpLowering : public ConversionPattern {
  static int constantID;

  ONNXConstantOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConstantOp::getOperationName(), 1, ctx) {
    constantID = 0;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = ONNXLoc<ONNXConstantOp>(op);
    auto constantOp = llvm::dyn_cast<ONNXConstantOp>(op);

    if (constantOp.sparse_value().hasValue())
      return emitError(loc, "Only support dense values at this time");

    auto memRefType = convertToMemRefType(*op->result_type_begin());

    // Shape based computations.
    auto shape = memRefType.getShape();
    int64_t numElements = 1;
    for (int i = 0; i < shape.size(); ++i)
      numElements *= shape[i];

    // Emit the constant global in Krnl dialect.
    auto constantGlobal = rewriter.create<KrnlGlobalOp>(loc, memRefType,
        /*shape=*/rewriter.getI64ArrayAttr(shape),
        /*name=*/
        rewriter.getStringAttr("constant_" + std::to_string(constantID)),
        /*value=*/constantOp.value().getValue(),
        /*offset=*/nullptr);

    // Increment constant ID:
    constantID++;

    // Check if the variable is returned.
    if (checkOpResultIsReturned(&constantOp)) {
      // In this case, use an AllocOp for the constant since krnl.Global
      // operations are not mean to be returned.
      AllocOp alloc = rewriter.create<AllocOp>(loc, memRefType);

      // Compute size in bytes using the input tensor.
      Value tensorSize = emitConstantOp(rewriter, loc,
          rewriter.getIntegerType(64), getMemRefEltSizeInBytes(memRefType));
      auto numElementsValue = emitConstantOp(
          rewriter, loc, rewriter.getIntegerType(64), numElements);
      tensorSize = rewriter.create<MulIOp>(loc, tensorSize, numElementsValue);

      // Copy the value in the AllocOp.
      rewriter.create<KrnlMemcpyOp>(
          loc, alloc, constantGlobal.getResult(), tensorSize);

      // Since the value is returned we need to only work with the AllocOp
      // not the KrnlGlobalOp. Globals cannot be returned.
      rewriter.replaceOp(op, alloc.getResult());
    } else {
      // Replace this operation with the generated krnl.global.
      rewriter.replaceOp(op, constantGlobal.getResult());
    }

    return success();
  }
};

int ONNXConstantOpLowering::constantID;

void populateLoweringONNXConstantOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOpLowering>(ctx);
}
