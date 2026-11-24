/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Concat.cpp - Lowering Concat Op to Linalg -------------===//
//
// Copyright 2026 HighTec-EDV Systeme Gmbh Authors.
//
// =============================================================================
//
// This file lowers the ONNX Concat Operator to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#include <algorithm>

using namespace mlir;

namespace onnx_mlir {

struct ONNXConcatOpLoweringToLinalg : public OpRewritePattern<ONNXConcatOp> {
  ONNXConcatOpLoweringToLinalg(
      MLIRContext *ctx, const std::string &linalgOps, bool useLinalgPath)
      : OpRewritePattern<ONNXConcatOp>(ctx), linalgOps(linalgOps),
        useLinalgPath(useLinalgPath) {}

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const final {
    if (!shouldConvertToLinalg(concatOp, linalgOps, useLinalgPath)) {
      return rewriter.notifyMatchFailure(
          concatOp, "operation not selected for Linalg conversion");
    }

    if (concatOp.getInputs().empty())
      return rewriter.notifyMatchFailure(
          concatOp, "expected at least one input");

    auto outputType =
        dyn_cast<RankedTensorType>(concatOp->getResult(0).getType());

    if (std::any_of(concatOp.getInputs().begin(), concatOp.getInputs().end(),
            [](const Value &input) {
              return !isa<RankedTensorType>(input.getType());
            }) ||
        !outputType)
      return rewriter.notifyMatchFailure(
          concatOp, "expected ranked tensor types");

    // ONNX treats a rank-one tensor<0xT> as an empty input and ignores it,
    // even when the other inputs have a different rank. tensor.concat instead
    // requires every input to have the result rank, so omit these operands.
    SmallVector<Value> concatInputs;
    for (Value input : concatOp.getInputs()) {
      auto inputType = cast<RankedTensorType>(input.getType());
      if (inputType.getRank() == 1 && inputType.getDimSize(0) == 0)
        continue;
      concatInputs.push_back(input);
    }

    // Keep the original operands in the all-empty case so tensor.concat still
    // receives at least one input.
    if (concatInputs.empty())
      concatInputs.append(
          concatOp.getInputs().begin(), concatOp.getInputs().end());

    int64_t joinAxis = concatOp.getAxis();
    if (joinAxis < 0)
      joinAxis += outputType.getRank();

    Location loc = concatOp->getLoc();
    Value tensorConcat = tensor::ConcatOp::create(
        rewriter, loc, outputType, joinAxis, concatInputs);
    rewriter.replaceOp(concatOp, tensorConcat);
    return success();
  }

private:
  std::string linalgOps;
  bool useLinalgPath;
};

void populateLoweringONNXConcatOpToLinalgPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx,
    const std::string &linalgOps, bool useLinalgPath) {
  patterns.add<ONNXConcatOpLoweringToLinalg>(ctx, linalgOps, useLinalgPath);
}

} // namespace onnx_mlir
