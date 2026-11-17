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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/ADT/SmallVector.h"

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
    Location loc = ONNXLoc<ONNXTransposeOp>(transposeOp);

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(
          transposeOp, "expected ranked input tensor type");

    int64_t rank = inputType.getRank();
    auto outputType =
        dyn_cast<RankedTensorType>(transposeOp.getResult().getType());

    if (!outputType)
      return rewriter.notifyMatchFailure(
          transposeOp, "expected ranked output tensor type");

    if (outputType.getRank() != rank)
      return rewriter.notifyMatchFailure(
          transposeOp, "input and output ranks must match");

    ArrayAttr permAttr = transposeOp.getPermAttr();
    if (permAttr && static_cast<int64_t>(permAttr.size()) != rank)
      return rewriter.notifyMatchFailure(
          transposeOp, "permutation size must equal the input rank");

    SmallVector<int64_t> permutation;
    SmallVector<bool> seen(rank, false);
    if (!permAttr) {
      for (int64_t i = rank - 1; i >= 0; --i)
        permutation.push_back(i);
    } else {
      for (int64_t i = 0; i < rank; ++i) {
        int64_t axis = ArrayAttrIntVal(permAttr, i);
        if (axis < 0)
          axis += rank;
        if (axis < 0 || axis >= rank || seen[axis])
          return rewriter.notifyMatchFailure(
              transposeOp, "permutation axis is out of bounds or duplicated");

        permutation.push_back(axis);
        seen[axis] = true;
      }
    }

    llvm::SmallVector<Value> tensorDims;
    for (int64_t outputAxis = 0; outputAxis < rank; ++outputAxis) {
      if (outputType.isDynamicDim(outputAxis))
        tensorDims.push_back(tensor::DimOp::create(
            rewriter, loc, input, permutation[outputAxis]));
    }

    Value emptyTensor =
        tensor::EmptyOp::create(rewriter, loc, outputType, tensorDims);
    Value linalgTranspose = linalg::TransposeOp::create(
        rewriter, loc, input, emptyTensor, permutation)
                                ->getResult(0);
    rewriter.replaceOp(transposeOp, linalgTranspose);
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
