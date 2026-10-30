//===--------- Transpose.cpp - Lowering Transpose Op to Linalg ----------===//
//
// Copyright 2026 HighTec-EDV Systeme Gmbh Authors.
//
// =============================================================================
//
// This file lowers the ONNX Transpose Operator to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToLinalg/DialectBuilder.hpp"
#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXTransposeOpLoweringToLinalg
    : public OpRewritePattern<ONNXTransposeOp> {
  ONNXTransposeOpLoweringToLinalg(
      MLIRContext *ctx, const std::string &linalgOps, bool useLinalgPath)
      : OpRewritePattern<ONNXTransposeOp>(ctx), linalgOps(linalgOps),
        useLinalgPath(useLinalgPath) {}

  LogicalResult matchAndRewrite(
      ONNXTransposeOp transposeOp, PatternRewriter &rewriter) const final {

    if (!shouldConvertToLinalg(
            transposeOp.getOperation(), linalgOps, useLinalgPath)) {
      return rewriter.notifyMatchFailure(
          transposeOp, "operation not selected for Linalg conversion");
    }

    Value input = transposeOp.getData();
    Location loc = transposeOp->getLoc();

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(
          transposeOp, "expected ranked input tensor type");

    int64_t rank = inputType.getRank();
    IndexExprBuilderForTensor ieBuilder{rewriter, loc};
    ONNXTransposeOpShapeHelper shapeHelper{
        transposeOp.getOperation(), transposeOp->getOperands(), &ieBuilder};

    if (failed(shapeHelper.computeShape()))
      return rewriter.notifyMatchFailure(
          transposeOp, "failed to compute output shape");

    auto outputType =
        dyn_cast<RankedTensorType>(transposeOp.getResult().getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(
          transposeOp, "expected ranked output tensor type");

    if (outputType.getRank() != rank)
      return rewriter.notifyMatchFailure(
          transposeOp, "input and output ranks must match");

    // The shape helper materializes ONNX's default reversed permutation when
    // the perm attribute is omitted.
    ArrayAttr permAttr = transposeOp.getPermAttr();
    if (!permAttr || static_cast<int64_t>(permAttr.size()) != rank)
      return rewriter.notifyMatchFailure(
          transposeOp, "permutation size must equal the input rank");

    SmallVector<int64_t> permutation;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t axis = ArrayAttrIntVal(permAttr, i);
      if (axis < 0)
        axis += rank;
      permutation.push_back(axis);
    }

    const DimsExpr &outputDims = shapeHelper.getOutputDims();

    SmallVector<Value> dynamicSizes;
    for (int64_t axis = 0; axis < outputType.getRank(); ++axis) {
      if (outputType.isDynamicDim(axis))
        dynamicSizes.push_back(outputDims[axis].getValue());
    }

    Value emptyTensor = tensor::EmptyOp::create(rewriter, loc,
        outputType.getShape(), outputType.getElementType(), dynamicSizes);

    Value result = linalg::TransposeOp::create(
        rewriter, loc, input, emptyTensor, permutation)
                       ->getResult(0);

    rewriter.replaceOp(transposeOp, result);
    return success();
  }

private:
  std::string linalgOps;
  bool useLinalgPath;
};

void populateLoweringONNXTransposeOpToLinalgPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx,
    const std::string &linalgOps, bool useLinalgPath) {
  patterns.add<ONNXTransposeOpLoweringToLinalg>(ctx, linalgOps, useLinalgPath);
}

} // namespace onnx_mlir
