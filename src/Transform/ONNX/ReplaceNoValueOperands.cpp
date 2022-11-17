/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- MaxPoolSingleOut.cpp - MaxPoolSingleOut Op-----------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file replace onnx.NoValue ops with the standard value constant
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
    return "Invoke passes iteratively that transform ONNX operation.";
  }
  void runOnOperation() final;
};

class Conv2DReplaceNoValue : public OpRewritePattern<ONNXConvOp> {
public:
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXConvOp op, PatternRewriter &rewriter) const override {

    auto bias = op.B();
    if (!bias.getType().isa<NoneType>()) {
      return success();
    }

    auto weight = op.W();
    auto weightType = weight.getType().cast<ShapedType>();
    auto weightShape = weightType.getShape();

    DenseElementsAttr newBiasAttr = DenseElementsAttr::get(
        RankedTensorType::get({weightShape[0]}, rewriter.getF32Type()), {0.0F});
    bias =
        createONNXConstantOpWithDenseAttr(rewriter, op->getLoc(), newBiasAttr);

    op->setOperand(2, bias);
    return success();
  }
};

void ReplaceNoValuePass::runOnOperation() {
  auto module = getOperation();

  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<Conv2DReplaceNoValue>(context);

  // Define pattern rewriter
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.maxIterations = 1;

  (void)applyPatternsAndFoldGreedily(module, std::move(patterns), config);
}
} // namespace
} // namespace onnx_mlir

namespace onnx_mlir {
std::unique_ptr<Pass> createReplaceNoValuePass() {
  return std::make_unique<ReplaceNoValuePass>();
}
} // namespace onnx_mlir