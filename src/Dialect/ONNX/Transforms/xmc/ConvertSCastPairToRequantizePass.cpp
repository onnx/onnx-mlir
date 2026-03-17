// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
// ConvertSCastPairToRequantizePass
//
// Converts back-to-back quant.scast pairs into XCOMPILERRequantize ops.
// Also converts:
//   1. Q node -> Scast node (with diff scale/zp) -> Requantize
//   2. Scast node -> Dequantize node (with diff scale/zp) -> Requantize + DQ
//
// Pattern 1:
//   %a = quant.scast %x : tensor<...x!quant.uniform<T1>> to tensor<...xT>
//   %b = quant.scast %a : tensor<...xT> to tensor<...x!quant.uniform<T2>>
//
// Pattern 2: ONNXQuantizeLinear(scale1,zp1) -> quant.scast(to scale2,zp2)
// Pattern 3: quant.scast(scale1,zp1) -> ONNXDequantizeLinear(scale2,zp2)
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/// Extract constant scale (float) and zero_point (int64) from ONNX constant
/// operands. Returns nullopt if either value is not a constant or lengths
/// don't match. Supports per-tensor (single element) and per-axis.
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
                      (elements.getElementType().isInteger(1));
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

/// Build F32ArrayAttr from float vector (e.g. from ONNX constant operands)
static ArrayAttr buildScaleAttr(
    PatternRewriter &rewriter, ArrayRef<float> scales) {
  SmallVector<Attribute> attrs;
  for (float s : scales)
    attrs.push_back(rewriter.getF32FloatAttr(s));
  return rewriter.getArrayAttr(attrs);
}

/// Build I64ArrayAttr from int64 vector (e.g. from ONNX constant operands)
static ArrayAttr buildZeroPointAttr(
    PatternRewriter &rewriter, ArrayRef<int64_t> zps) {
  return rewriter.getI64ArrayAttr(SmallVector<int64_t>(zps.begin(), zps.end()));
}

/// Build F32ArrayAttr from a UniformQuantizedType's scale
ArrayAttr buildScaleAttr(
    PatternRewriter &rewriter, quant::UniformQuantizedType qType) {
  return rewriter.getArrayAttr(
      {rewriter.getF32FloatAttr(static_cast<float>(qType.getScale()))});
}

/// Build I64ArrayAttr from a UniformQuantizedType's zero point
ArrayAttr buildZeroPointAttr(
    PatternRewriter &rewriter, quant::UniformQuantizedType qType) {
  return rewriter.getI64ArrayAttr({qType.getZeroPoint()});
}

/// Build F32ArrayAttr from a UniformQuantizedPerAxisType's scales
ArrayAttr buildScaleAttr(
    PatternRewriter &rewriter, quant::UniformQuantizedPerAxisType qType) {
  SmallVector<Attribute> attrs;
  for (double s : qType.getScales())
    attrs.push_back(rewriter.getF32FloatAttr(static_cast<float>(s)));
  return rewriter.getArrayAttr(attrs);
}

/// Build I64ArrayAttr from a UniformQuantizedPerAxisType's zero points
ArrayAttr buildZeroPointAttr(
    PatternRewriter &rewriter, quant::UniformQuantizedPerAxisType qType) {
  SmallVector<int64_t> zps(
      qType.getZeroPoints().begin(), qType.getZeroPoints().end());
  return rewriter.getI64ArrayAttr(zps);
}

/// Pattern: Match back-to-back quant.scast ops and convert to
/// XCOMPILERRequantize
struct ConvertSCastPairToRequantizePattern
    : public OpRewritePattern<quant::StorageCastOp> {
  using OpRewritePattern<quant::StorageCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(quant::StorageCastOp secondSCast,
      PatternRewriter &rewriter) const override {
    // The second scast: storage type → quantized type (requantize direction)
    // Result should be quantized
    auto resultType = dyn_cast<RankedTensorType>(secondSCast.getType());
    if (!resultType)
      return failure();

    Type resultElemType = resultType.getElementType();
    if (!isa<quant::QuantizedType>(resultElemType))
      return failure();

    // Input should be storage type (integer)
    Value midValue = secondSCast.getOperand();
    auto midType = dyn_cast<RankedTensorType>(midValue.getType());
    if (!midType)
      return failure();

    // Mid type should NOT be quantized (should be storage type like i8, u8)
    if (isa<quant::QuantizedType>(midType.getElementType()))
      return failure();

    // The first scast: quantized type → storage type (dequantize direction)
    auto firstSCast = midValue.getDefiningOp<quant::StorageCastOp>();
    if (!firstSCast)
      return failure();

    // First scast's input should be quantized
    auto inputType =
        dyn_cast<RankedTensorType>(firstSCast.getOperand().getType());
    if (!inputType)
      return failure();

    Type inputElemType = inputType.getElementType();
    if (!isa<quant::QuantizedType>(inputElemType))
      return failure();

    // Extract input quantization parameters
    auto inputQType = dyn_cast<quant::UniformQuantizedType>(inputElemType);
    auto inputQPerAxis =
        dyn_cast<quant::UniformQuantizedPerAxisType>(inputElemType);

    // Extract output quantization parameters
    auto outputQType = dyn_cast<quant::UniformQuantizedType>(resultElemType);
    auto outputQPerAxis =
        dyn_cast<quant::UniformQuantizedPerAxisType>(resultElemType);

    // Both must be same kind (per-tensor or per-channel)
    if ((inputQType != nullptr) != (outputQType != nullptr))
      return failure();
    if ((inputQPerAxis != nullptr) != (outputQPerAxis != nullptr))
      return failure();

    ArrayAttr aScaleAttr, aZpAttr, yScaleAttr, yZpAttr;

    if (inputQType && outputQType) {
      // Per-tensor quantization
      // Skip if params are identical (not a requantization)
      if (std::abs(inputQType.getScale() - outputQType.getScale()) < 1e-9 &&
          inputQType.getZeroPoint() == outputQType.getZeroPoint())
        return failure();

      aScaleAttr = buildScaleAttr(rewriter, inputQType);
      aZpAttr = buildZeroPointAttr(rewriter, inputQType);
      yScaleAttr = buildScaleAttr(rewriter, outputQType);
      yZpAttr = buildZeroPointAttr(rewriter, outputQType);
    } else if (inputQPerAxis && outputQPerAxis) {
      // Per-channel quantization
      aScaleAttr = buildScaleAttr(rewriter, inputQPerAxis);
      aZpAttr = buildZeroPointAttr(rewriter, inputQPerAxis);
      yScaleAttr = buildScaleAttr(rewriter, outputQPerAxis);
      yZpAttr = buildZeroPointAttr(rewriter, outputQPerAxis);
    } else {
      return failure();
    }

    // Create XCOMPILERRequantize op
    auto requantizeOp =
        rewriter.create<XCOMPILERRequantizeOp>(secondSCast.getLoc(), resultType,
            firstSCast.getOperand(), aScaleAttr, aZpAttr, yScaleAttr, yZpAttr);

    rewriter.replaceOp(secondSCast, requantizeOp.getResult());

    // If first scast has no other uses, remove it
    if (firstSCast->use_empty())
      rewriter.eraseOp(firstSCast);

    return success();
  }
};

/// Pattern 2: Q node -> Scast node with diff scale and zp -> Requantize
/// ONNXQuantizeLinear(x, scale_q, zp_q) -> quant.scast -> tensor<...x!quant.T2>
/// Replace with XCOMPILERRequantize(Q.result, scale_q, zp_q, scale_s, zp_s).
struct ConvertQAndScastToRequantizePattern
    : public OpRewritePattern<quant::StorageCastOp> {
  using OpRewritePattern<quant::StorageCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      quant::StorageCastOp scast, PatternRewriter &rewriter) const override {
    // Scast result must be quantized type
    auto resultType = dyn_cast<RankedTensorType>(scast.getType());
    if (!resultType)
      return failure();
    Type resultElemType = resultType.getElementType();
    auto outputQType = dyn_cast<quant::UniformQuantizedType>(resultElemType);
    auto outputQPerAxis =
        dyn_cast<quant::UniformQuantizedPerAxisType>(resultElemType);
    if (!outputQType && !outputQPerAxis)
      return failure();

    // Input to Scast must be produced by ONNX QuantizeLinear
    Value scastInput = scast.getOperand();
    auto qOp = scastInput.getDefiningOp<ONNXQuantizeLinearOp>();
    if (!qOp)
      return failure();

    // Get scale/zp from Q (operands 1 and 2)
    Value qScaleVal = qOp.getYScale();
    Value qZpVal = qOp.getYZeroPoint();
    auto qParams = getConstantScaleAndZp(qScaleVal, qZpVal);
    if (!qParams)
      return failure();

    ArrayAttr aScaleAttr = buildScaleAttr(rewriter, qParams->first);
    ArrayAttr aZpAttr = buildZeroPointAttr(rewriter, qParams->second);

    ArrayAttr yScaleAttr, yZpAttr;
    if (outputQType) {
      if (qParams->first.size() != 1)
        return failure();
      if (std::abs(qParams->first[0] - outputQType.getScale()) < 1e-9 &&
          qParams->second[0] == outputQType.getZeroPoint())
        return failure();

      yScaleAttr = buildScaleAttr(rewriter, outputQType);
      yZpAttr = buildZeroPointAttr(rewriter, outputQType);
    } else {
      if (qParams->first.size() != (size_t)outputQPerAxis.getScales().size())
        return failure();
      bool same = true;
      for (size_t i = 0; i < qParams->first.size(); ++i) {
        if (std::abs(qParams->first[i] - outputQPerAxis.getScales()[i]) >=
                1e-9 ||
            qParams->second[i] != outputQPerAxis.getZeroPoints()[i]) {
          same = false;
          break;
        }
      }
      if (same)
        return failure();
      yScaleAttr = buildScaleAttr(rewriter, outputQPerAxis);
      yZpAttr = buildZeroPointAttr(rewriter, outputQPerAxis);
    }

    auto requantizeOp = rewriter.create<XCOMPILERRequantizeOp>(scast.getLoc(),
        resultType, qOp.getResult(), aScaleAttr, aZpAttr, yScaleAttr, yZpAttr);
    rewriter.replaceOp(scast, requantizeOp.getResult());
    return success();
  }
};

/// Pattern 3: Scast node -> Dequantize node with diff scale and zp
/// quant.scast(quant_type_1) -> ONNXDequantizeLinear(scale_dq, zp_dq)
/// Replace with XCOMPILERRequantize(...) ->
/// ONNXDequantizeLinear(scale_dq,zp_dq).
struct ConvertScastAndDQToRequantizePattern
    : public OpRewritePattern<ONNXDequantizeLinearOp> {
  using OpRewritePattern<ONNXDequantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDequantizeLinearOp dqOp, PatternRewriter &rewriter) const override {
    Value dqInput = dqOp.getX();
    auto scast = dqInput.getDefiningOp<quant::StorageCastOp>();
    if (!scast)
      return failure();

    Value scastInput = scast.getOperand();
    auto inputType = dyn_cast<RankedTensorType>(scastInput.getType());
    if (!inputType)
      return failure();
    Type inputElemType = inputType.getElementType();
    auto inputQType = dyn_cast<quant::UniformQuantizedType>(inputElemType);
    auto inputQPerAxis =
        dyn_cast<quant::UniformQuantizedPerAxisType>(inputElemType);
    if (!inputQType && !inputQPerAxis)
      return failure();

    auto dqParams =
        getConstantScaleAndZp(dqOp.getXScale(), dqOp.getXZeroPoint());
    if (!dqParams)
      return failure();

    ArrayAttr aScaleAttr, aZpAttr, yScaleAttr, yZpAttr;
    RankedTensorType resultType;

    if (inputQType && dqParams->first.size() == 1) {
      if (std::abs(inputQType.getScale() - dqParams->first[0]) < 1e-9 &&
          inputQType.getZeroPoint() == dqParams->second[0])
        return failure();
      aScaleAttr = buildScaleAttr(rewriter, inputQType);
      aZpAttr = buildZeroPointAttr(rewriter, inputQType);
      yScaleAttr = buildScaleAttr(rewriter, dqParams->first);
      yZpAttr = buildZeroPointAttr(rewriter, dqParams->second);
      auto outQType = quant::UniformQuantizedType::get(inputQType.getFlags(),
          inputQType.getStorageType(), inputQType.getExpressedType(),
          dqParams->first[0], dqParams->second[0],
          inputQType.getStorageTypeMin(), inputQType.getStorageTypeMax());
      resultType = RankedTensorType::get(inputType.getShape(), outQType);
    } else if (inputQPerAxis && (size_t)dqParams->first.size() ==
                                    inputQPerAxis.getScales().size()) {
      bool same = true;
      for (size_t i = 0; i < dqParams->first.size(); ++i) {
        if (std::abs(inputQPerAxis.getScales()[i] - dqParams->first[i]) >=
                1e-9 ||
            inputQPerAxis.getZeroPoints()[i] != dqParams->second[i]) {
          same = false;
          break;
        }
      }
      if (same)
        return failure();
      aScaleAttr = buildScaleAttr(rewriter, inputQPerAxis);
      aZpAttr = buildZeroPointAttr(rewriter, inputQPerAxis);
      yScaleAttr = buildScaleAttr(rewriter, dqParams->first);
      yZpAttr = buildZeroPointAttr(rewriter, dqParams->second);
      SmallVector<double> scalesDouble(
          dqParams->first.begin(), dqParams->first.end());
      auto outQPerAxis = quant::UniformQuantizedPerAxisType::get(
          inputQPerAxis.getFlags(), inputQPerAxis.getStorageType(),
          inputQPerAxis.getExpressedType(), scalesDouble, dqParams->second,
          inputQPerAxis.getQuantizedDimension(),
          inputQPerAxis.getStorageTypeMin(), inputQPerAxis.getStorageTypeMax());
      resultType = RankedTensorType::get(inputType.getShape(), outQPerAxis);
    } else {
      return failure();
    }

    auto requantizeOp = rewriter.create<XCOMPILERRequantizeOp>(dqOp.getLoc(),
        resultType, scastInput, aScaleAttr, aZpAttr, yScaleAttr, yZpAttr);

    // Keep DQ to produce f32: DQ(requantize_result, scale_dq, zp_dq)
    auto dqResultType = cast<RankedTensorType>(dqOp.getResult().getType());
    auto newDQOp = rewriter.create<ONNXDequantizeLinearOp>(dqOp.getLoc(),
        dqResultType, requantizeOp.getResult(), dqOp.getXScale(),
        dqOp.getXZeroPoint(), dqOp.getAxisAttr(), dqOp.getBlockSizeAttr());

    rewriter.replaceOp(dqOp, newDQOp.getResult());
    if (scast->use_empty())
      rewriter.eraseOp(scast);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ConvertSCastPairToRequantizePass
    : public PassWrapper<ConvertSCastPairToRequantizePass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "convert-scast-pair-to-requantize";
  }
  StringRef getDescription() const override {
    return "Convert back-to-back quant.scast pairs with different quantization "
           "parameters into XCOMPILERRequantize operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertSCastPairToRequantizePattern>(context);
    patterns.add<ConvertQAndScastToRequantizePattern>(context);
    patterns.add<ConvertScastAndDQToRequantizePattern>(context);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertSCastPairToRequantizePass() {
  return std::make_unique<ConvertSCastPairToRequantizePass>();
}

} // namespace onnx_mlir
