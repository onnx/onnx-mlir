//===- QDQOpt.cpp - Remove QDQ operations --------*- C++ -*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include "mlir/Transforms/DialectConversion.h"
#include <cmath>

using namespace mlir;
using namespace onnx_mlir;
std::tuple<Value /*input*/, Value /*output*/> getDataInputOutput(
    ONNXTransposeOp transposeOp) {
  return {transposeOp.getData(), transposeOp.getTransposed()};
}
std::tuple<Value /*input*/, Value /*output*/> getDataInputOutput(
    ONNXUnsqueezeOp unsqueezeOp) {
  return {unsqueezeOp.getData(), unsqueezeOp.getExpanded()};
}
std::tuple<Value /*input*/, Value /*output*/> getDataInputOutput(
    ONNXSqueezeOp squeezeOp) {
  return {squeezeOp.getData(), squeezeOp.getSqueezed()};
}
std::tuple<Value /*input*/, Value /*output*/> getDataInputOutput(
    ONNXReshapeOp reshapeOp) {
  return {reshapeOp.getData(), reshapeOp.getReshaped()};
}
std::tuple<Value /*input*/, Value /*output*/> getDataInputOutput(
    ONNXGatherOp gatherOp) {
  return {gatherOp.getData(), gatherOp.getOutput()};
}
std::tuple<Value /*input*/, Value /*output*/> getDataInputOutput(
    ONNXReduceSumOp reduceOp) {
  return {reduceOp.getData(), reduceOp.getReduced()};
}
std::tuple<Value /*input*/, Value /*output*/> getDataInputOutput(
    ONNXSliceOp sliceOp) {
  return {sliceOp.getData(), sliceOp.getOutput()};
}
std::tuple<Value /*input*/, Value /*output*/> getDataInputOutput(
    ONNXResizeOp resizeOp) {
  return {resizeOp.getX(), resizeOp.getY()};
}
std::tuple<Value /*input*/, Value /*output*/> getDataInputOutput(
    ONNXFlattenOp flattenOp) {
  return {flattenOp.getInput(), flattenOp.getOutput()};
}
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

namespace {
template <typename T>
class RemoveQDQAroundOpPattern : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      T op, PatternRewriter &rewriter) const override {
    Value input, output;
    std::tie(input, output) = getDataInputOutput(op);

    auto dqOp = input.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp) {
      return failure();
    }
    if (output.hasOneUse()) {
      Operation *firstOp = *(output.getUsers().begin());
      if (mlir::isa<ONNXQuantizeLinearOp>(firstOp)) {
        // auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        // if (!(op2)){
        // }
        auto qOp = ::llvm::dyn_cast<ONNXQuantizeLinearOp>(firstOp);
        // if (!(castedOp2)){
        if (!quantizationParamsMatch(dqOp.getXScale(), dqOp.getXZeroPoint(),
                qOp.getYScale(), qOp.getYZeroPoint())) {
          return failure();
        }
        rewriter.replaceOp(dqOp, dqOp.getX());
        rewriter.replaceOp(qOp, qOp.getY());
        return success();
      }
    };
  }
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
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // patterns.add<RemoveQDQAroundOpPattern<ONNXTransposeOp>,
    //     RemoveQDQAroundOpPattern<ONNXUnsqueezeOp>,
    //     RemoveQDQAroundOpPattern<ONNXSqueezeOp>,
    //     RemoveQDQAroundOpPattern<ONNXReshapeOp>,
    //     RemoveQDQAroundOpPattern<ONNXGatherOp>,
    //     RemoveQDQAroundOpPattern<ONNXReduceSumOp>,
    //     RemoveQDQAroundOpPattern<ONNXSliceOp>,
    //     RemoveQDQAroundOpPattern<ONNXResizeOp>,
    //     RemoveQDQAroundOpPattern<ONNXFlattenOp>>(patterns.getContext());
    patterns.add<RemoveQDQAroundOpPattern<ONNXTransposeOp>>(ctx);
    // if (failed(applyPatternsGreedily(function, std::move(patterns))))
    //   signalPassFailure();
  }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createQDQAroundOpOptONNXToONNXPass() {
  return std::make_unique<QDQAroundOpOptONNXToONNXPass>();
}
} // namespace onnx_mlir