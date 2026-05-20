// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
// ConvertQDQToRequantizePass
//
// Pre-`quant-types` cleanup for back-to-back Q/DQ pairs at the f32 boundary.
//
// Pattern A (FoldEqualQDQ):
//   Q(DQ(x, s1, zp1), s1, zp1) -> x
//   (mirrors FoldQDQPattern from QDQCanonicalize.cpp; reuses
//   isDequantQuantSame for the scale/zero-point/storage-type check.)
//
// Pattern B (InsertRequantizeBetweenQDQ):
//   When the parameters differ (scale, zero-point or storage type),
//   insert an XCOMPILERRequantize op on the f32 edge between DQ and Q:
//     Before: ... -> DQ(s1, zp1) ->                      Q(s2, zp2) -> ...
//     After:  ... -> DQ(s1, zp1) -> XCOMPILERRequantize -> Q(s2, zp2) -> ...
//   The Requantize op carries the explicit (s1,zp1) -> (s2,zp2) parameters
//   so that when `quant-types` runs afterwards, both DQ and Q are converted
//   to quant.scast ops and the Requantize remains as a proper op in the IR.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include <optional>
#include <utility>

using namespace mlir;

namespace {

/// Extract per-tensor / per-axis scale (float) and zero_point (int64) vectors
/// from ONNX constant operands. Returns nullopt if either value is not a
/// constant, the element kind is unsupported, or the vector lengths mismatch.
static std::optional<std::pair<SmallVector<float>, SmallVector<int64_t>>>
getConstantScaleAndZp(Value scaleVal, Value zpVal) {
  auto getScaleFloats = [](Value v) -> std::optional<SmallVector<float>> {
    auto *def = v.getDefiningOp();
    if (!def)
      return std::nullopt;
    auto constOp = dyn_cast<ONNXConstantOp>(def);
    if (!constOp || !constOp.getValueAttr())
      return std::nullopt;
    auto elements = dyn_cast<ElementsAttr>(constOp.getValueAttr());
    if (!elements)
      return std::nullopt;
    SmallVector<float> out;
    for (auto apFloat : elements.getValues<APFloat>())
      out.push_back(apFloat.convertToFloat());
    return out;
  };
  auto getZpInt64s = [](Value v) -> std::optional<SmallVector<int64_t>> {
    auto *def = v.getDefiningOp();
    if (!def)
      return std::nullopt;
    auto constOp = dyn_cast<ONNXConstantOp>(def);
    if (!constOp || !constOp.getValueAttr())
      return std::nullopt;
    auto elements = dyn_cast<ElementsAttr>(constOp.getValueAttr());
    if (!elements)
      return std::nullopt;
    SmallVector<int64_t> out;
    bool isUnsigned = elements.getElementType().isUnsignedInteger() ||
                      elements.getElementType().isInteger(1);
    for (auto apInt : elements.getValues<APInt>())
      out.push_back(
          isUnsigned ? (int64_t)apInt.getZExtValue() : apInt.getSExtValue());
    return out;
  };
  auto scales = getScaleFloats(scaleVal);
  auto zps = getZpInt64s(zpVal);
  if (!scales || !zps || scales->size() != zps->size())
    return std::nullopt;
  return std::pair{*scales, *zps};
}

static ArrayAttr buildScaleAttr(
    PatternRewriter &rewriter, ArrayRef<float> scales) {
  SmallVector<Attribute> attrs;
  for (float s : scales)
    attrs.push_back(rewriter.getF32FloatAttr(s));
  return rewriter.getArrayAttr(attrs);
}

static ArrayAttr buildZeroPointAttr(
    PatternRewriter &rewriter, ArrayRef<int64_t> zps) {
  return rewriter.getI64ArrayAttr(SmallVector<int64_t>(zps.begin(), zps.end()));
}

/// Pattern A: Fold Q(DQ(x)) when scale, zero-point, and storage type all
/// match. Mirrors FoldQDQPattern in QDQCanonicalize.cpp.
struct FoldEqualQDQPattern : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {
    auto dqOp = qOp.getX().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return failure();
    if (!onnx_mlir::isDequantQuantSame(dqOp, qOp))
      return failure();
    rewriter.replaceOp(qOp, dqOp.getX());
    return success();
  }
};

/// Pattern B: When Q(DQ(x)) has any mismatch in scale, zero-point, or
/// storage type, insert XCOMPILERRequantize on the f32 edge between DQ and Q.
struct InsertRequantizeBetweenQDQPattern
    : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {
    auto dqOp = qOp.getX().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return failure();

    // Pattern A handles the equal-parameters case; bail out here.
    if (onnx_mlir::isDequantQuantSame(dqOp, qOp))
      return failure();

    auto dqParams =
        getConstantScaleAndZp(dqOp.getXScale(), dqOp.getXZeroPoint());
    auto qParams = getConstantScaleAndZp(qOp.getYScale(), qOp.getYZeroPoint());
    if (!dqParams || !qParams)
      return failure();

    // Verifier requires both inputs/outputs to be per-tensor or per-channel.
    bool dqPerChannel = dqParams->first.size() > 1;
    bool qPerChannel = qParams->first.size() > 1;
    if (dqPerChannel != qPerChannel)
      return failure();

    ArrayAttr aScale = buildScaleAttr(rewriter, dqParams->first);
    ArrayAttr aZp = buildZeroPointAttr(rewriter, dqParams->second);
    ArrayAttr yScale = buildScaleAttr(rewriter, qParams->first);
    ArrayAttr yZp = buildZeroPointAttr(rewriter, qParams->second);

    // DQ's f32 output feeds the Requantize, which produces an identically
    // shaped f32 tensor that becomes Q's new input.
    Value dqResult = dqOp.getY();
    Type f32TensorType = dqResult.getType();

    auto requantize = rewriter.create<XCOMPILERRequantizeOp>(
        qOp.getLoc(), f32TensorType, dqResult, aScale, aZp, yScale, yZp);

    // ResultNames is not populated for RequantizeOp here.
    // In QuantTypes pass, it will get the ResultName of QuantizeLinear
    // following it.
    rewriter.modifyOpInPlace(
        qOp, [&]() { qOp->setOperand(0, requantize.getResult()); });

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ConvertQDQToRequantizePass
    : public PassWrapper<ConvertQDQToRequantizePass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "convert-qdq-to-requantize"; }
  StringRef getDescription() const override {
    return "Pre-quant-types pass: fold equal Q/DQ pairs and insert "
           "XCOMPILERRequantize between DQ -> Q pairs whose scale, "
           "zero-point or storage type differ.";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FoldEqualQDQPattern>(context);
    patterns.add<InsertRequantizeBetweenQDQPattern>(context);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createConvertQDQToRequantizePass() {
  return std::make_unique<ConvertQDQToRequantizePass>();
}

} // namespace onnx_mlir
