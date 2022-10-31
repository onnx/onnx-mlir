/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Squeeze.cpp - Lowering Squeeze Op ----------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Squeeze Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXSqueezeOp(A) is implemented using MHLO reshapeOp
struct ONNXSqueezeOpLoweringToMhlo : public ConversionPattern {
  ONNXSqueezeOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeOpAdaptor operandAdaptor(operands);
    ONNXSqueezeOp squeezeOp = llvm::cast<ONNXSqueezeOp>(op);
    Location loc = op->getLoc();
    Value data = squeezeOp.data();

    IndexExprScope scope(&rewriter, loc);
    ONNXSqueezeOpShapeHelper shapeHelper(&squeezeOp, &scope);
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

void populateLoweringONNXSqueezeOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
