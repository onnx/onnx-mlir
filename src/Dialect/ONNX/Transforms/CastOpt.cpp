//===- CastOpt.cpp - Remove Cast operations --------*- C++ -*-===//
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
  auto intAttr = (*attr.getValues<IntegerAttr>().begin());
  return intAttr.getInt();
}

bool quantizationParamsMatch(
    Value scale1, Value zp1, Value scale2, Value zp2, float tolerance = 1e-5f) {
  float s1 = getFloatFromConstant(scale1);
  float s2 = getFloatFromConstant(scale2);
  int64_t z1 = getIntFromConstant(zp1);
  int64_t z2 = getIntFromConstant(zp2);
  bool zeroPointMatch = (z1 == z2);
  bool scaleClose = std::fabs(s1 - s2) < tolerance;
  return zeroPointMatch && scaleClose;
}

struct RemoveCastPattern : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {

    auto castOp = qOp.getX().getDefiningOp<ONNXCastOp>();
    if (!castOp)
      return failure();

    auto dqOp = castOp.getInput().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return failure();

    if (!quantizationParamsMatch(dqOp.getXScale(), dqOp.getXZeroPoint(),
            qOp.getYScale(), qOp.getYZeroPoint()))
      return failure();

    mlir::Type targetElementType = castOp.getTo();
    mlir::Type inputElementType =
        castOp.getInput().getType().cast<mlir::ShapedType>().getElementType();
    if (inputElementType == targetElementType) {
      rewriter.replaceOp(qOp, dqOp.getX());
      return success();
    } else {
      return failure();
    }
  }
};

struct CastOptONNXToONNXPass
    : public PassWrapper<CastOptONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CastOptONNXToONNXPass)

  StringRef getArgument() const override { return "cast-opt-onnx-to-onnx"; }
  StringRef getDescription() const override {
    return "Remove Cast ops and surrounding QDQ if safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveCastPattern>(&getContext());
    if (failed(applyPatternsGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createCastOptONNXToONNXPass() {
  return std::make_unique<CastOptONNXToONNXPass>();
}
} // namespace onnx_mlir
