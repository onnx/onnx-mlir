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


static std::tuple<Value /*input*/, Value /*output*/> getDataInputs(ONNXTransposeOp transposeOp) {
  return {transposeOp.getData(), transposeOp.getTransposed()};
}

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

template <typename T>
class RemoveQDQAroundOpPattern : public OpRewritePattern<T> {
    // if (isa<ONNXMatMulOp, ONNXMatMulIntegerOp>(op)) {

  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      T op, PatternRewriter &rewriter) const override {
    Value value = getDataInputs(op);
    auto dqOp = value.getDefiningOp<ONNXDequantizeLinearOp>();
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

struct QDQAroundOpOptONNXToONNXPass
    : public PassWrapper<QDQAroundOpOptONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QDQAroundOpOptONNXToONNXPass)
  StringRef getArgument() const override { return "qdq-around-op-opt-onnx-to-onnx"; }
  StringRef getDescription() const override {
    return "Remove QDQ around ops if safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveQDQAroundOpPattern<ONNXTransposeOp>>(&getContext());
    patterns.add<RemoveQDQAroundOpPattern<ONNXUnsqueezeOp>>(&getContext());
    patterns.add<RemoveQDQAroundOpPattern<ONNXSqueezeOp>>(&getContext());
    patterns.add<RemoveQDQAroundOpPattern<ONNXReshapeOp>>(&getContext());
    patterns.add<RemoveQDQAroundOpPattern<ONNXGatherOp>>(&getContext());
    patterns.add<RemoveQDQAroundOpPattern<ONNXReduceSumOp>>(&getContext());
    patterns.add<RemoveQDQAroundOpPattern<ONNXSliceOp>>(&getContext());
    patterns.add<RemoveQDQAroundOpPattern<ONNXResizeOp>>(&getContext());
    patterns.add<RemoveQDQAroundOpPattern<ONNXFlattenOp>>(&getContext());


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