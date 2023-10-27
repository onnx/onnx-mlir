/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Transpose.cpp - Lowering Transpose Op ---------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Transpose Operator to StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStableHlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStableHlo/ONNXToStableHloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXTransposeOpLoweringToStableHlo : public ConversionPattern {
  ONNXTransposeOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTransposeOpAdaptor operandAdaptor(operands);
    ONNXTransposeOp transposeOp = llvm::cast<ONNXTransposeOp>(op);
    Location loc = op->getLoc();

    // Operands
    Value data = operandAdaptor.getData();

    // Convert the output type
    Type outputType = *op->result_type_begin();
    assert(outputType.isa<ShapedType>() && "Expected ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    int64_t rank = outputShapedType.getShape().size();

    // Attributes
    std::optional<ArrayAttr> permAttr = transposeOp.getPerm();
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

void populateLoweringONNXTransposeOpToStableHloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLoweringToStableHlo>(ctx);
}

} // namespace onnx_mlir
