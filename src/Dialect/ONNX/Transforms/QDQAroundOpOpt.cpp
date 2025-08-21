//===- QDQAroundOpOpt.cpp - Remove DQ, Q operations around data movement ops
//--------*- C++ -*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <src/Dialect/ONNX/ONNXOps.hpp>
#include <src/Dialect/ONNX/ONNXOps/OpHelper.hpp>

using namespace mlir;
using namespace onnx_mlir;
struct InputAndOutput {
  Value input;
  Value output;
};

InputAndOutput getDataInputOutput(ONNXTransposeOp transposeOp) {
  return {transposeOp.getData(), transposeOp.getTransposed()};
}
InputAndOutput getDataInputOutput(ONNXUnsqueezeOp unsqueezeOp) {
  return {unsqueezeOp.getData(), unsqueezeOp.getExpanded()};
}
InputAndOutput getDataInputOutput(ONNXSqueezeOp squeezeOp) {
  return {squeezeOp.getData(), squeezeOp.getSqueezed()};
}
InputAndOutput getDataInputOutput(ONNXReshapeOp reshapeOp) {
  return {reshapeOp.getData(), reshapeOp.getReshaped()};
}
InputAndOutput getDataInputOutput(ONNXGatherOp gatherOp) {
  return {gatherOp.getData(), gatherOp.getOutput()};
}
InputAndOutput getDataInputOutput(ONNXSliceOp sliceOp) {
  return {sliceOp.getData(), sliceOp.getOutput()};
}
InputAndOutput getDataInputOutput(ONNXResizeOp resizeOp) {
  return {resizeOp.getX(), resizeOp.getY()};
}
InputAndOutput getDataInputOutput(ONNXFlattenOp flattenOp) {
  return {flattenOp.getInput(), flattenOp.getOutput()};
}
namespace {
template <typename T>
class RemoveQDQAroundOpPattern : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      T op, PatternRewriter &rewriter) const override {
    if (llvm::isa<ONNXResizeOp>(op)) {
      auto &resizeOp = llvm::cast<ONNXResizeOp>(op);
      if (resizeOp.getMode() != "nearest") {
        return failure();
      }
    }
    InputAndOutput opIO = getDataInputOutput(op);

    auto dqOp = opIO.input.getDefiningOp<ONNXDequantizeLinearOp>();
    // Only run this pass if Quantizelization is on tensor
    if (!dqOp || !isScalarConstantTensor(dqOp.getXScale()) ||
        !isScalarConstantTensor(dqOp.getXZeroPoint())) {
      return failure();
    }
    if (!opIO.output.hasOneUse()) {
      return failure();
    }

    Operation *firstOp = *(opIO.output.getUsers().begin());
    if (auto qOp = dyn_cast<ONNXQuantizeLinearOp>(firstOp)) {
      if (!isScalarConstantTensor(qOp.getYScale()) ||
          !isScalarConstantTensor(qOp.getYZeroPoint())) {
        return failure();
      }
      if (!isDequantQuantSame(dqOp, qOp))
        return failure();

      // Map dqOp inputs to dqOp's inputs
      IRMapping irMapping;
      irMapping.map(dqOp, dqOp.getX());

      SmallVector<Value> newInputs;
      transform(op->getOperands(), std::back_inserter(newInputs),
          [&](Value operand) { return irMapping.lookupOrDefault(operand); });

      auto newOp =
          rewriter.create<T>(op.getLoc(), TypeRange{qOp.getResult().getType()},
              ValueRange{newInputs}, op->getAttrs());
      rewriter.replaceOp(qOp, newOp.getResult());
      return success();
    }
  };
};
struct QDQAroundOpOptONNXToONNXPass
    : public PassWrapper<QDQAroundOpOptONNXToONNXPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QDQAroundOpOptONNXToONNXPass)
  StringRef getArgument() const override {
    return "qdq-around-op-opt-onnx-to-onnx";
  }
  StringRef getDescription() const override {
    return "Remove QDQ around ops if safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // ONNXReduceSumOp is expecting high precision value, it failed to compile
    // during applying this pass, so for now there is no dq, q removal around
    // ReduceSum
    patterns.add<RemoveQDQAroundOpPattern<ONNXTransposeOp>,
        RemoveQDQAroundOpPattern<ONNXUnsqueezeOp>,
        RemoveQDQAroundOpPattern<ONNXSqueezeOp>,
        RemoveQDQAroundOpPattern<ONNXReshapeOp>,
        RemoveQDQAroundOpPattern<ONNXResizeOp>,
        RemoveQDQAroundOpPattern<ONNXGatherOp>,
        RemoveQDQAroundOpPattern<ONNXSliceOp>,
        RemoveQDQAroundOpPattern<ONNXFlattenOp>>(patterns.getContext());
    if (failed(applyPatternsGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createQDQAroundOpOptONNXToONNXPass() {
  return std::make_unique<QDQAroundOpOptONNXToONNXPass>();
}
} // namespace onnx_mlir