/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Shape.cpp - Lowering Shape Op ----------------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Shape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXShapeOpLowering : public ConversionPattern {
  ONNXShapeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXShapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXShapeOpAdaptor operandAdaptor(operands);
    ONNXShapeOp shapeOp = cast<ONNXShapeOp>(op);
    Location loc = op->getLoc();
    ONNXShapeOpShapeHelper shapeHelper(&shapeOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // TODO: if the dimensions are known at compile time
    // (shapeHelper.dimsForOutput literal), then we could use a constant array.
    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput());

    // Compute the data selected by the Shape operator.
    DimsExpr selectedData = computeSelectedData(operandAdaptor);

    // Iterate along the data shape storing dim value to result.
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
    for (uint64_t i = 0; i < selectedData.size(); ++i) {
      Value val = selectedData[i].getValue();
      Value intVal = create.math.cast(elementType, val);
      create.krnl.storeIE(intVal, alloc, {LiteralIndexExpr(i)});
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXShapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXShapeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
