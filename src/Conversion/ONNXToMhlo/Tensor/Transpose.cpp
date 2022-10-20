/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Transpose.cpp - Lowering Transpose Op ---------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Transpose Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXTransposeOpLoweringToMhlo : public ConversionPattern {
  ONNXTransposeOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTransposeOpAdaptor operandAdaptor(operands);
    ONNXTransposeOp transposeOp = llvm::cast<ONNXTransposeOp>(op);
    Location loc = op->getLoc();

    // Operands
    Value data = operandAdaptor.data();

    // Convert the output type
    Type outputType = *op->result_type_begin();
    assert(outputType.isa<ShapedType>() && "Expected ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    int64_t rank = outputShapedType.getShape().size();

    // Attributes
    llvm::Optional<ArrayAttr> permAttr = transposeOp.perm();
    DenseIntElementsAttr permAxis;
    RankedTensorType permAxisType =
        RankedTensorType::get({rank}, rewriter.getI64Type());
    SmallVector<int64_t, 4> permAxisList;
    if (permAttr.has_value()) {
      for (int64_t i = 0; i < rank; ++i)
        permAxisList.push_back(ArrayAttrIntVal(permAttr, i));
      permAxis = DenseIntElementsAttr::get(permAxisType, permAxisList);
    } else {
      for (int64_t i = 0; i < rank; ++i)
        permAxisList.push_back(rank - 1 - i);
      permAxis = DenseIntElementsAttr::get(permAxisType, permAxisList);
    }

    // Get a shape helper.
    ONNXTransposeOpShapeHelper shapeHelper(&transposeOp);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    Value transposeValue =
        rewriter.create<mhlo::TransposeOp>(loc, outputType, data, permAxis);
    rewriter.replaceOp(op, transposeValue);

    return success();
  }
};

} // namespace

void populateLoweringONNXTransposeOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
