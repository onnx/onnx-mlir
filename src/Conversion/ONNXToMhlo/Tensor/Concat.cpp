/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Concat Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "onnx_to_mhlo"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXConcatOpLoweringToMhlo : public ConversionPattern {
  ONNXConcatOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConcatOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    ONNXConcatOpAdaptor operandAdaptor(operands);
    ONNXConcatOp concatOp = llvm::cast<ONNXConcatOp>(op);

    assert(op->getNumResults() == 1 && "ONNXConcatOp shoule have 1 result");
    Type resultType = op->getResult(0).getType();
    if (!onnx_mlir::isRankedShapedType(resultType)) {
      LLVM_DEBUG(llvm::dbgs() << "Concat Output Is Not Ranked\n");
      return failure();
    }
    int64_t rank = onnx_mlir::getRank(resultType);
    int64_t axis = concatOp.axis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1 && "Axis out of rank range");

    ValueRange inputs = operandAdaptor.inputs();
    Value result = rewriter.create<mhlo::ConcatenateOp>(
        loc, op->getResultTypes(), inputs, rewriter.getI64IntegerAttr(axis));
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXConcatOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConcatOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
