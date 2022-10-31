/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Unsqueeze.cpp - Lowering Unsqueeze Op ----------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Unsqueeze Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXUnsqueezeOp(A) is implemented using MHLO reshapeOp
struct ONNXUnsqueezeOpLoweringToMhlo : public ConversionPattern {
  ONNXUnsqueezeOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXUnsqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXUnsqueezeOpAdaptor operandAdaptor(operands);
    ONNXUnsqueezeOp unsqueezeOp = llvm::cast<ONNXUnsqueezeOp>(op);
    Location loc = op->getLoc();
    Value data = unsqueezeOp.data();

    IndexExprScope scope(&rewriter, loc);
    ONNXUnsqueezeOpShapeHelper shapeHelper(&unsqueezeOp, &scope);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    SmallVector<Value, 4> dims;
    IndexExpr::getValues(shapeHelper.dimsForOutput(), dims);

    Type outputShapeType =
        RankedTensorType::get({(int64_t)dims.size()}, rewriter.getIndexType());
    Value newShapeValue = rewriter.create<shape::FromExtentsOp>(loc, dims);
    newShapeValue = rewriter.create<shape::ToExtentTensorOp>(
        loc, outputShapeType, newShapeValue);
    Type outputType = *op->result_type_begin();
    Value result = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, outputType, data, newShapeValue);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXUnsqueezeOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXUnsqueezeOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
