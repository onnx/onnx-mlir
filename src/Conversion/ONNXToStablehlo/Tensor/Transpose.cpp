/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Transpose.cpp - Lowering Transpose Op ---------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Transpose Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXTransposeOpLoweringToStablehlo : public ConversionPattern {
  ONNXTransposeOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTransposeOpAdaptor operandAdaptor(operands);
    ONNXTransposeOp transposeOp = llvm::cast<ONNXTransposeOp>(op);
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();

    // Operands
    Value data = operandAdaptor.getData();

    // Convert the output type
    Type outputType = *op->result_type_begin();
    assert(mlir::isa<ShapedType>(outputType) && "Expected ShapedType");
    ShapedType outputShapedType = mlir::cast<ShapedType>(outputType);
    int64_t rank = outputShapedType.getShape().size();

    // Attributes
    std::optional<ArrayAttr> permAttr = transposeOp.getPerm();
    DenseI64ArrayAttr permAxis;
    SmallVector<int64_t, 4> permAxisList;
    if (permAttr.has_value()) {
      for (int64_t i = 0; i < rank; ++i)
        permAxisList.push_back(ArrayAttrIntVal(permAttr, i));
      permAxis = DenseI64ArrayAttr::get(context, permAxisList);
    } else {
      for (int64_t i = 0; i < rank; ++i)
        permAxisList.push_back(rank - 1 - i);
      permAxis = DenseI64ArrayAttr::get(context, permAxisList);
    }

    // Get a shape helper: unused, needed?
    IndexExprBuilderForAnalysis createIE(loc);
    ONNXTransposeOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Value transposeValue = rewriter.create<stablehlo::TransposeOp>(
        loc, outputType, data, permAxis);
    rewriter.replaceOp(op, transposeValue);

    return success();
  }
};

} // namespace

void populateLoweringONNXTransposeOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
