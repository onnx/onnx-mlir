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
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "reshape_onnx_to_krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXReshapeOpLowering : public ConversionPattern {
  ONNXReshapeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXReshapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXReshapeOpAdaptor operandAdaptor(operands);
    ONNXReshapeOp reshapeOp = dyn_cast_or_null<ONNXReshapeOp>(op);

    auto loc = op->getLoc();
    Value data = operandAdaptor.data();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    LLVM_DEBUG(llvm::dbgs() << "memRefType: " << memRefType << "\n");

    ONNXReshapeOpShapeHelper shapeHelper(&reshapeOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    LogicalResult shapeComputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapeComputed;
    assert(succeeded(shapeComputed) && "Could not compute shape");

    // Lower to ReinterpretCastOp so that the data is never copied or modified.
    Value newView = emitMemRefReinterpretCastOp(
        rewriter, loc, data, memRefType, shapeHelper.dimsForOutput(0));
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
