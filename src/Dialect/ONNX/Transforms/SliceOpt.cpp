//===- SliceOpt.cpp - Remove Slice operations --------*- C++ -*-===//
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
// Helper: extract float from ONNX ConstantOp
float getFloatFromConstant(mlir::Value val) {
  auto constOp = val.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return 0.0f;

  auto valueAttr = constOp.getValueAttr();
  if (!valueAttr)
    return 0.0f;

  auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr);
  if (!denseAttr || denseAttr.getNumElements() != 1)
    return 0.0f;

  auto it = denseAttr.getValues<FloatAttr>().begin();
  return (*it).getValueAsDouble();
}

int64_t getIntFromConstant(mlir::Value val) {
  auto constOp = val.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return 0;

  auto valueAttr = constOp.getValueAttr();
  if (!valueAttr)
    return 0;

  auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr);
  if (!denseAttr || denseAttr.getNumElements() != 1)
    return 0;

  auto it = denseAttr.getValues<IntegerAttr>().begin();
  return (*it).getInt();
}

bool quantizationParamsMatch(mlir::Value scale1, mlir::Value zp1,
    mlir::Value scale2, mlir::Value zp2, float tolerance = 1e-5f) {
  float s1 = getFloatFromConstant(scale1);
  float s2 = getFloatFromConstant(scale2);
  int64_t z1 = getIntFromConstant(zp1);
  int64_t z2 = getIntFromConstant(zp2);
  bool zeroPointMatch = (z1 == z2);
  bool scaleClose = std::fabs(s1 - s2) < tolerance;
  return zeroPointMatch && scaleClose;
}

struct RemoveSlicePattern : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {
    auto sliceOp = qOp.getX().getDefiningOp<ONNXSliceOp>();
    if (!sliceOp)
      return failure();

    auto dqOp = sliceOp.getData().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return failure();

    if (!quantizationParamsMatch(dqOp.getXScale(), dqOp.getXZeroPoint(),
            qOp.getYScale(), qOp.getYZeroPoint()))
      return failure();

    auto inputType =
        mlir::dyn_cast<RankedTensorType>(sliceOp.getData().getType());
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(sliceOp.getResult().getType());
    if (inputType != outputType)
      return failure();

    if (sliceOp->getNumOperands() >= 5) {
      Value stepsValue = sliceOp->getOperand(4);
      auto stepsDefOp = stepsValue.getDefiningOp();
      if (!stepsDefOp || !llvm::isa<ONNXConstantOp>(stepsDefOp)) {
        return failure();
      }
      auto stepsConstOp = llvm::cast<ONNXConstantOp>(stepsDefOp);
      auto valueAttr = stepsConstOp.getValueAttr();
      if (!valueAttr) {
        return failure();
      }
      if (auto elementsAttr = valueAttr.dyn_cast<mlir::ElementsAttr>()) {
        bool allOne = true;
        for (auto val : elementsAttr.getValues<llvm::APInt>()) {
          if (val != 1) {
            allOne = false;
            break;
          }
        }
        if (!allOne) {
          return failure();
        }
      }
    }
    rewriter.replaceOp(qOp, dqOp.getX());
    return success();
  }
};

struct SliceOptONNXToONNXPass
    : public PassWrapper<SliceOptONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SliceOptONNXToONNXPass)

  StringRef getArgument() const override { return "slice-opt-onnx-to-onnx"; }
  StringRef getDescription() const override {
    return "Remove Slice ops and surrounding QDQ if safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveSlicePattern>(&getContext());
    if (failed(applyPatternsGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createSliceOptONNXToONNXPass() {
  return std::make_unique<SliceOptONNXToONNXPass>();
}
} // namespace onnx_mlir