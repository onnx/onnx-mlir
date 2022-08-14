/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- RewriteONNXForZHigh.cpp - Rewrite ONNX ops for ZHigh lowering ----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements pass for rewriting of ONNX operations to generate
// combination of ONNX and ZHigh operations.
//
// - `ONNXBatchNormalizationInferenceModeOp`
// In this pass, `ONNXBatchNormalizationInferenceModeOp` is converted into
// `ZHigh.BatchNorm`, generating `ONNX.Add`, `ONNX.Sub`, `ONNX.Mul`, `ONNX.Div`,
// and `ONNX.Sqrt` to calculate inputs(`a` and `b`) for `ZHigh.BatchNorm`.
// `ONNXToZHighLoweringPass`(`--convert-onnx-to-zhigh`) is also able to generate
// the ONNX ops, but,they are lowered to ZHigh ops. So, constant
// propagation(`--constprop-onnx`) doesn't work. To enable to work it, this
// separate pass is needed. By using this pass, constant propagation works by
// running it just after this pass.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

/// Calculate sqrt(var + epsilon) for batchnorm op A.
/// A = scale / sqrt(var + epsilon)
Value getSqrtResultBatchNormA(
    Location loc, PatternRewriter &rewriter, Value var, FloatAttr epsilon) {
  Type elementType = var.getType().cast<ShapedType>().getElementType();

  // epsilon
  RankedTensorType epsilonType = RankedTensorType::get({1}, elementType);
  DenseElementsAttr epsilonConstAttr =
      DenseElementsAttr::get<float>(epsilonType, epsilon.getValueAsDouble());
  Value epsilonConst = rewriter.create<ONNXConstantOp>(loc, epsilonType,
      nullptr, epsilonConstAttr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr);

  // sqrt(var + epsilon)
  Value var_plus_epsilon = rewriter.create<ONNXAddOp>(loc, var, epsilonConst);
  Value sqrtResult =
      rewriter.create<ONNXSqrtOp>(loc, var.getType(), var_plus_epsilon);

  return sqrtResult;
}

/// Check if A is unidirectionally broadcastable to B, e.g.
/// A: [256], B: [128x256]
/// A: [1], B: [128x256]
/// More info about unidirectional broadcasting:
/// https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
/// Note: being differenct from ONNX broadcasting, we return false if A and B
/// have exactly the same static shape.
bool isUniBroadcatableFirstToSecond(Value A, Value B) {
  if (!hasStaticShape(A.getType()) || !hasStaticShape(B.getType()))
    return false;
  ArrayRef<int64_t> aDims = getShape(A.getType());
  ArrayRef<int64_t> bDims = getShape(B.getType());
  // A and B have exactly the same static shape.
  if (aDims == bDims)
    return false;
  // aDims size > bDims size: not unidirectional broadcasting from A to B, but B
  // to A.
  if (aDims.size() > bDims.size())
    return false;
  // Pre-pad A's shape with dims 1 so that two shapes have the same size.
  SmallVector<int64_t> paddedADims(bDims.size(), 1);
  for (unsigned i = 0; i < aDims.size(); ++i)
    paddedADims[i + bDims.size() - aDims.size()] = aDims[i];
  // Check unidirectional broadcasting.
  bool isUniBroadcasting = true;
  for (unsigned i = 0; i < paddedADims.size(); ++i)
    if (paddedADims[i] > bDims[i]) {
      isUniBroadcasting = false;
      break;
    }
  return isUniBroadcasting;
}

/// Check a value is defined by ONNXConstantOp or not.
bool isDefinedByONNXConstantOp(Value v) {
  if (v.isa<BlockArgument>())
    return false;
  return isa<ONNXConstantOp>(v.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// Rewrite ONNX ops to ZHigh ops and ONNX ops for ZHigh.
//===----------------------------------------------------------------------===//

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXRewriteONNXForZHigh.inc"

struct RewriteONNXForZHighPass
    : public PassWrapper<RewriteONNXForZHighPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "rewrite-onnx-for-zhigh"; }

  StringRef getDescription() const override {
    return "Rewrite ONNX ops for ZHigh.";
  }

  RewriteONNXForZHighPass() = default;
  RewriteONNXForZHighPass(mlir::ArrayRef<std::string> execNodesOnCpu)
      : execNodesOnCpu(execNodesOnCpu) {}
  void runOnOperation() final;

public:
  mlir::ArrayRef<std::string> execNodesOnCpu = mlir::ArrayRef<std::string>();
};

void RewriteONNXForZHighPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<ONNXDialect, zhigh::ZHighDialect, func::FuncDialect>();

  // Single ONNX to ZHigh operation lowering.
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);

  // `ONNXBatchNormalizationInferenceModeOp` to `ZHigh.BatchNorm`,
  // generating `ONNX.Add`, `ONNX.Sub`, `ONNX.Mul`, `ONNX.Div`,
  // and `ONNX.Sqrt` to calculate inputs(`a` and `b`)
  addDynamicallyLegalOpFor<ONNXBatchNormalizationInferenceModeOp>(
      &target, execNodesOnCpu);

  target.addDynamicallyLegalOp<ONNXAddOp>([](ONNXAddOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });

  target.addDynamicallyLegalOp<ONNXDivOp>([](ONNXDivOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });

  target.addDynamicallyLegalOp<ONNXMulOp>([](ONNXMulOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });

  target.addDynamicallyLegalOp<ONNXSubOp>([](ONNXSubOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createRewriteONNXForZHighPass() {
  return std::make_unique<RewriteONNXForZHighPass>();
}

std::unique_ptr<Pass> createRewriteONNXForZHighPass(
    mlir::ArrayRef<std::string> execNodesOnCpu) {
  return std::make_unique<RewriteONNXForZHighPass>(execNodesOnCpu);
}

} // namespace onnx_mlir
