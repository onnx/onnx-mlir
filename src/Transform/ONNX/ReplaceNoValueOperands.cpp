/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- ReplaceNoValueOperands.cpp - ReplaceNoValueOperands Op------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This pass replaces onnx.NoValue ops with the standard value constant
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {
struct ReplaceNoValuePass : public mlir::PassWrapper<ReplaceNoValuePass,
                                OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceNoValuePass)

  ReplaceNoValuePass() = default;
  ReplaceNoValuePass(const ReplaceNoValuePass &pass)
      : PassWrapper<ReplaceNoValuePass, OperationPass<ModuleOp>>() {}

  StringRef getArgument() const override { return "onnx-replace-novalue"; }

  StringRef getDescription() const override {
    return "Replaces NoValue op with a constant op of the default shape "
           "and value for the op.";
  }
  void runOnOperation() final;
};

bool isNotNoValue(Value &value) { return !value.getType().isa<NoneType>(); }

Value createONNXConstFromFloatValue(PatternRewriter &rewriter,
    const Location &loc, ArrayRef<int64_t> shape, float floatValue) {
  DenseElementsAttr newConstantAttr = DenseElementsAttr::get(
      RankedTensorType::get(shape, rewriter.getF32Type()), {floatValue});
  auto constOp = OnnxBuilder(rewriter, loc).constant(newConstantAttr);
  return constOp;
}

class PadReplaceNoValue : public OpRewritePattern<ONNXPadOp> {
public:
  using OpRewritePattern<ONNXPadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXPadOp op, PatternRewriter &rewriter) const override {
    auto constValue = op.getConstantValue();
    if (isNotNoValue(constValue)) {
      return rewriter.notifyMatchFailure(op, "has no NoValue operand");
    }

    constValue =
        createONNXConstFromFloatValue(rewriter, op.getLoc(), {1}, 0.0F);
    op.setOperand(2, constValue);
    return success();
  }
};

class GemmReplaceNoValue : public OpRewritePattern<ONNXGemmOp> {
public:
  using OpRewritePattern<ONNXGemmOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXGemmOp op, PatternRewriter &rewriter) const override {
    auto matrixC = op.getC();
    if (isNotNoValue(matrixC)) {
      return rewriter.notifyMatchFailure(op, "has no NoValue operand");
    }
    ArrayRef<int64_t> cshape(
        op.getResult().getType().cast<TensorType>().getShape()[1]);

    matrixC =
        createONNXConstFromFloatValue(rewriter, op.getLoc(), cshape, 0.0F);
    op.setOperand(2, matrixC);
    return success();
  }
};

class Conv2DReplaceNoValue : public OpRewritePattern<ONNXConvOp> {
public:
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXConvOp op, PatternRewriter &rewriter) const override {

    auto bias = op.getB();
    if (isNotNoValue(bias)) {
      return rewriter.notifyMatchFailure(op, "has no NoValue operand");
    }

    auto weight = op.getW();
    auto weightType = weight.getType().cast<ShapedType>();
    auto weightShape = weightType.getShape();

    bias = createONNXConstFromFloatValue(
        rewriter, op.getLoc(), {weightShape[0]}, 0.0F);

    op->setOperand(2, bias);
    return success();
  }
};

void ReplaceNoValuePass::runOnOperation() {
  auto module = getOperation();

  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<Conv2DReplaceNoValue, GemmReplaceNoValue, PadReplaceNoValue>(
      context);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}
} // namespace
} // namespace onnx_mlir

namespace onnx_mlir {
std::unique_ptr<Pass> createReplaceNoValuePass() {
  return std::make_unique<ReplaceNoValuePass>();
}
} // namespace onnx_mlir