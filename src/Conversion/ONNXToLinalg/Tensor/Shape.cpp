//===--------------- Shape.cpp - Lowering Shape Op to Linalg --------------===//
//
// Copyright 2026 HighTec-EDV Systeme Gmbh Authors.
//
// =============================================================================
//
// This file lowers the ONNX Shape operator to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace onnx_mlir {

struct ONNXShapeOpLoweringToLinalg : public OpRewritePattern<ONNXShapeOp> {
  ONNXShapeOpLoweringToLinalg(
      MLIRContext *ctx, const std::string &linalgOps, bool useLinalgPath)
      : OpRewritePattern<ONNXShapeOp>(ctx), linalgOps(linalgOps),
        useLinalgPath(useLinalgPath) {}

  LogicalResult matchAndRewrite(
      ONNXShapeOp shapeOp, PatternRewriter &rewriter) const final {
    if (!shouldConvertToLinalg(shapeOp, linalgOps, useLinalgPath)) {
      return rewriter.notifyMatchFailure(
          shapeOp, "operation not selected for Linalg conversion");
    }
    Value input = shapeOp.getData();
    Location loc = shapeOp->getLoc();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto outputType =
        dyn_cast<RankedTensorType>(shapeOp->getOpResult(0).getType());

    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(
          shapeOp, "expected ranked tensor types");

    int64_t start, end;
    ONNXShapeOpShapeHelper::getStartEndValues(shapeOp, start, end);
    if (start > end)
      return rewriter.notifyMatchFailure(
          shapeOp, "normalized start must not be greater than end");

    int64_t resultLength = end - start;
    if (outputType.getRank() != 1 || outputType.isDynamicDim(0) ||
        outputType.getDimSize(0) != resultLength)
      return rewriter.notifyMatchFailure(
          shapeOp, "expected shape-inferred result type");

    Type elementType = outputType.getElementType();
    llvm::SmallVector<Value> tensorDims;
    for (int64_t tensorDimIdx = start; tensorDimIdx < end; ++tensorDimIdx) {
      Value dim = tensor::DimOp::create(rewriter, loc, input, tensorDimIdx);
      tensorDims.push_back(
          arith::IndexCastOp::create(rewriter, loc, elementType, dim));
    }

    Value result =
        tensor::FromElementsOp::create(rewriter, loc, outputType, tensorDims);

    rewriter.replaceOp(shapeOp, result);
    return success();
  }

private:
  std::string linalgOps;
  bool useLinalgPath;
};

void populateLoweringONNXShapeOpToLinalgPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx,
    const std::string &linalgOps, bool useLinalgPath) {
  patterns.add<ONNXShapeOpLoweringToLinalg>(ctx, linalgOps, useLinalgPath);
}

} // namespace onnx_mlir
