// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <utility>

using namespace mlir;

namespace {

/// Extract scale and zero-point from a `!quant.uniform` type, or return
/// defaults if the element type is not quantized.
static std::pair<double, int64_t> extractQuantParams(Type type) {
  if (auto qType = dyn_cast<quant::UniformQuantizedType>(type))
    return {qType.getScale(), qType.getZeroPoint()};
  return {1.0, 0};
}

/// Extract float data from an `onnx.Constant`'s `value` attribute.
static SmallVector<float> extractFloatData(Operation *op) {
  if (!op || op->getName().getStringRef() != "onnx.Constant")
    return {};
  auto valueAttr = op->getAttr("value");
  if (!valueAttr)
    return {};
  SmallVector<float> result;
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    for (auto v : denseAttr.getValues<float>())
      result.push_back(v);
  }
  return result;
}

/// Pattern:
///   onnx.Conv -> onnx.Reshape -> onnx.Transpose -> onnx.BatchNormalization
///
/// Merges BatchNormalization parameters into Conv by computing and attaching
/// attributes on the Conv op and removing BatchNormalization.
struct MergeBatchnormToConvPattern : public RewritePattern {
  MergeBatchnormToConvPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "onnx.Conv")
      return failure();

    Operation *convOp = op;
    if (!convOp->hasOneUse())
      return failure();

    Value convOutput = convOp->getResult(0);
    Operation *reshapeOp = *convOutput.getUsers().begin();
    if (!reshapeOp || reshapeOp->getName().getStringRef() != "onnx.Reshape")
      return failure();
    if (!reshapeOp->hasOneUse())
      return failure();

    Value reshapeOutput = reshapeOp->getResult(0);
    Operation *transposeOp = *reshapeOutput.getUsers().begin();
    if (!transposeOp ||
        transposeOp->getName().getStringRef() != "onnx.Transpose")
      return failure();
    if (!transposeOp->hasOneUse())
      return failure();

    Value transposeOutput = transposeOp->getResult(0);
    Operation *batchnormOp = *transposeOutput.getUsers().begin();
    if (!batchnormOp ||
        batchnormOp->getName().getStringRef() != "onnx.BatchNormalization")
      return failure();

    // BatchNormalization operands: X, scale, B, input_mean, input_var.
    if (batchnormOp->getNumOperands() < 5)
      return failure();
    Value scaleInput = batchnormOp->getOperand(1);
    Value biasInput = batchnormOp->getOperand(2);
    Value meanInput = batchnormOp->getOperand(3);
    Value varInput = batchnormOp->getOperand(4);

    Operation *scaleOp = scaleInput.getDefiningOp();
    Operation *biasOp = biasInput.getDefiningOp();
    Operation *meanOp = meanInput.getDefiningOp();
    Operation *varOp = varInput.getDefiningOp();
    if (!scaleOp || !biasOp || !meanOp || !varOp)
      return failure();

    if (scaleOp->getName().getStringRef() != "onnx.Constant" ||
        biasOp->getName().getStringRef() != "onnx.Constant" ||
        meanOp->getName().getStringRef() != "onnx.Constant" ||
        varOp->getName().getStringRef() != "onnx.Constant")
      return failure();

    auto scaleData = extractFloatData(scaleOp);
    auto biasData = extractFloatData(biasOp);
    auto meanData = extractFloatData(meanOp);
    auto varData = extractFloatData(varOp);
    if (scaleData.empty() || biasData.empty() || meanData.empty() ||
        varData.empty())
      return failure();

    // Infer channel count from conv output shape (XCompiler behavior).
    auto convType = dyn_cast<ShapedType>(convOutput.getType());
    if (!convType || !convType.hasStaticShape())
      return failure();
    ArrayRef<int64_t> convShape = convType.getShape();
    if (convShape.empty())
      return failure();
    int64_t numChannels = convShape.back();

    // XCompiler doubles these arrays (model-specific quirk).
    scaleData.append(scaleData.begin(), scaleData.end());
    biasData.append(biasData.begin(), biasData.end());
    meanData.append(meanData.begin(), meanData.end());
    varData.append(varData.begin(), varData.end());

    // Output quantization params from BatchNorm output.
    float outputScale = 1.0f;
    int32_t outputZeroPoint = 0;
    if (auto shapedType =
            dyn_cast<ShapedType>(batchnormOp->getResult(0).getType())) {
      auto [s, zp] = extractQuantParams(shapedType.getElementType());
      outputScale = static_cast<float>(s);
      outputZeroPoint = static_cast<int32_t>(zp);
    }

    // Input activation scale from Conv input.
    float s_a = 1.0f;
    if (auto shapedType =
            dyn_cast<ShapedType>(convOp->getOperand(0).getType())) {
      auto [s, _zp] = extractQuantParams(shapedType.getElementType());
      s_a = static_cast<float>(s);
    }

    // Approximate weight scale from Conv output quantization (XCompiler logic).
    float s_w = 1.0f;
    float convOutputScale = 1.0f;
    int32_t convOutputZeroPoint = 0;
    if (auto shapedType = dyn_cast<ShapedType>(convOutput.getType())) {
      auto [s, zp] = extractQuantParams(shapedType.getElementType());
      convOutputScale = static_cast<float>(s);
      convOutputZeroPoint = static_cast<int32_t>(zp);
      if (s_a != 0.0f)
        s_w = convOutputScale / s_a;
    }

    SmallVector<float> c2_f(numChannels);
    SmallVector<float> c3_f(numChannels);
    SmallVector<int32_t> batchnorm_out_q(numChannels);
    SmallVector<int32_t> batchnorm_out_q_255(numChannels);

    float inv_output_scale =
        (outputScale == 0.0f) ? 1.0f : (1.0f / outputScale);

    for (int64_t i = 0; i < numChannels; ++i) {
      float std_dev = std::sqrt(varData[i]);
      c2_f[i] = scaleData[i] * inv_output_scale / (s_a * s_w * std_dev);
      c3_f[i] = (-meanData[i]) / std_dev * scaleData[i] * inv_output_scale +
                biasData[i] * inv_output_scale + outputZeroPoint;

      float dq_output = -float(convOutputZeroPoint) / convOutputScale;
      float dq_output_255 =
          (255.0f - float(convOutputZeroPoint)) / convOutputScale;

      float batchnorm_out_f =
          (dq_output - meanData[i]) / std_dev * scaleData[i] + biasData[i];
      float batchnorm_out_f_255 =
          (dq_output_255 - meanData[i]) / std_dev * scaleData[i] + biasData[i];

      batchnorm_out_q[i] = std::clamp(
          static_cast<int32_t>(
              std::round(batchnorm_out_f * inv_output_scale) + outputZeroPoint),
          0, 255);
      batchnorm_out_q_255[i] =
          std::clamp(static_cast<int32_t>(
                         std::round(batchnorm_out_f_255 * inv_output_scale) +
                         outputZeroPoint),
              0, 255);
    }

    // Attach attributes to Conv (backend consumption).
    convOp->setAttr("nonlinear", rewriter.getStringAttr("BATCHNORM"));
    convOp->setAttr("c2_f", rewriter.getF32ArrayAttr(c2_f));
    convOp->setAttr("c3_f", rewriter.getF32ArrayAttr(c3_f));
    convOp->setAttr(
        "batchnorm_out_q", rewriter.getI32ArrayAttr(batchnorm_out_q));
    convOp->setAttr(
        "batchnorm_out_q_255", rewriter.getI32ArrayAttr(batchnorm_out_q_255));

    // Replace BatchNormalization output uses with transpose output and erase
    // BN.
    rewriter.replaceAllUsesWith(batchnormOp->getResult(0), transposeOutput);
    rewriter.eraseOp(batchnormOp);

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct MergeBatchnormToConvPass : public PassWrapper<MergeBatchnormToConvPass,
                                      OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "merge-batchnorm-to-conv"; }
  StringRef getDescription() const override {
    return "Merge BatchNormalization parameters into Conv as attributes";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<MergeBatchnormToConvPattern>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createMergeBatchnormToConvPass() {
  return std::make_unique<MergeBatchnormToConvPass>();
}

} // namespace onnx_mlir
