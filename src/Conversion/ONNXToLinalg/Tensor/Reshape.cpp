//===------------ Reshape.cpp - Lowering Reshape Op to Linalg -------------===//
//
// Copyright 2026 HighTec-EDV Systeme Gmbh Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reshape Operator to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "src/Conversion/ONNXToLinalg/DialectBuilder.hpp"
#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include <cstdint>

using namespace mlir;
using namespace onnx_mlir;

namespace {
// IndexExpr adapter for tensor-based lowerings. The Reshape shape helper uses
// it to materialize runtime shape values without introducing Krnl operations.

struct ONNXReshapeOpLoweringToLinalg : public OpRewritePattern<ONNXReshapeOp> {
  ONNXReshapeOpLoweringToLinalg(
      MLIRContext *ctx, const std::string &linalgOps, bool useLinalgPath)
      : OpRewritePattern<ONNXReshapeOp>(ctx), linalgOps(linalgOps),
        useLinalgPath(useLinalgPath) {}

  LogicalResult matchAndRewrite(
      ONNXReshapeOp reshapeOp, PatternRewriter &rewriter) const final {
    if (!shouldConvertToLinalg(
            reshapeOp.getOperation(), linalgOps, useLinalgPath)) {
      return rewriter.notifyMatchFailure(
          reshapeOp, "operation not selected for Linalg conversion");
    }

    if (!isa<RankedTensorType>(reshapeOp.getData().getType()))
      return rewriter.notifyMatchFailure(
          reshapeOp, "expected ranked input tensor");

    // ONNXReshapeOpShapeHelper currently does not account for allowzero.
    if (reshapeOp.getAllowzero() != 0)
      return rewriter.notifyMatchFailure(
          reshapeOp, "allowzero = 1 is not supported");

    auto shapeType = dyn_cast<RankedTensorType>(reshapeOp.getShape().getType());
    if (!shapeType || shapeType.getRank() != 1 || shapeType.isDynamicDim(0))
      return rewriter.notifyMatchFailure(
          reshapeOp, "expected a rank-one shape tensor with static length");

    Location loc = reshapeOp->getLoc();
    IndexExprBuilderForTensor createIE(rewriter, loc);
    ONNXReshapeOpShapeHelper shapeHelper(
        reshapeOp.getOperation(), reshapeOp->getOperands(), &createIE);
    if (failed(shapeHelper.computeShape()))
      return rewriter.notifyMatchFailure(
          reshapeOp, "failed to compute output shape");

    DimsExpr outputDims = shapeHelper.getOutputDims();
    if (llvm::any_of(
            outputDims, [](IndexExpr dim) { return dim.isQuestionmark(); }))
      return rewriter.notifyMatchFailure(
          reshapeOp, "could not materialize all output dimensions");

    SmallVector<Value> dims;
    IndexExpr::getValues(outputDims, dims);
    auto dimensionsType = RankedTensorType::get(
        {static_cast<int64_t>(dims.size())}, rewriter.getIndexType());
    Value dimensionsArr =
        tensor::FromElementsOp::create(rewriter, loc, dimensionsType, dims);

    Value result = tensor::ReshapeOp::create(rewriter, loc,
        reshapeOp->getResult(0).getType(), reshapeOp.getData(), dimensionsArr);
    rewriter.replaceOp(reshapeOp, result);
    return success();
  }

private:
  std::string linalgOps;
  bool useLinalgPath;
};
} // namespace

namespace onnx_mlir {
void populateLoweringONNXReshapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx,
    const std::string &linalgOps, bool useLinalgPath) {
  patterns.add<ONNXReshapeOpLoweringToLinalg>(ctx, linalgOps, useLinalgPath);
}
} // namespace onnx_mlir
