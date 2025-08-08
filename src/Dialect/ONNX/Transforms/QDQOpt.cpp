//===- QDQOpt.cpp - Remove QDQ operations --------*- C++ -*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
float getFloatFromConstant(Value val) {
  auto constOp = val.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return 0.0f;
  auto attr = constOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  if (!attr || attr.getNumElements() != 1)
    return 0.0f;
  auto floatAttr = (*attr.getValues<FloatAttr>().begin());
  return floatAttr.getValueAsDouble();
}

int64_t getIntFromConstant(Value val) {
  auto constOp = val.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return 0;
  auto attr = constOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  if (!attr || attr.getNumElements() != 1)
    return 0;
  auto elementType = attr.getType().getElementType();
  auto it = attr.getValues<APInt>().begin();
  if (it == attr.getValues<APInt>().end())
    return 0;
  APInt apInt = *it;
  return apInt.getSExtValue();
}

bool quantizationParamsMatch(
    Value scale1, Value zp1, Value scale2, Value zp2, float tolerance = 1e-5f) {
  float s1 = getFloatFromConstant(scale1);
  float s2 = getFloatFromConstant(scale2);
  int64_t z1 = getIntFromConstant(zp1);
  int64_t z2 = getIntFromConstant(zp2);
  llvm::outs() << z1 << z2 << "\n";
  bool zeroPointMatch = (z1 == z2);
  bool scaleClose = std::fabs(s1 - s2) < tolerance;
  return zeroPointMatch && scaleClose;
}

struct RemoveQDQPattern : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {
    auto dqOp = qOp.getX().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp) {
      return failure();
    }
    if (!quantizationParamsMatch(dqOp.getXScale(), dqOp.getXZeroPoint(),
            qOp.getYScale(), qOp.getYZeroPoint())) {
      return failure();
    }
    rewriter.replaceOp(qOp, dqOp.getX());
    return success();
  }
};

struct QDQOptONNXToONNXPass
    : public PassWrapper<QDQOptONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QDQOptONNXToONNXPass)
  StringRef getArgument() const override { return "qdq-opt-onnx-to-onnx"; }
  StringRef getDescription() const override {
    return "Remove QDQ ops and surrounding QDQ if safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveQDQPattern>(&getContext());
    if (failed(applyPatternsGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createQDQOptONNXToONNXPass() {
  return std::make_unique<QDQOptONNXToONNXPass>();
}
} // namespace onnx_mlir