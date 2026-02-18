// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
// ConvertSCastPairToRequantizePass
//
// Converts back-to-back quant.scast pairs into XCOMPILERRequantize ops.
//
// Pattern:
//   %a = quant.scast %x : tensor<...x!quant.uniform<T1>> to tensor<...xT>
//   %b = quant.scast %a : tensor<...xT> to tensor<...x!quant.uniform<T2>>
//
// Where T1 and T2 have different quantization parameters (scale/zero_point).
// The first scast dequantizes (quant → storage), the second requantizes
// (storage → quant with different params).
//
// Result:
//   %b = onnx.XCOMPILERRequantize(%x) {a_scale, a_zero_point, y_scale,
//         y_zero_point} : tensor<...x!quant.uniform<T1>>
//         -> tensor<...x!quant.uniform<T2>>
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

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

    // Check that the intermediate value has only one use (the second scast)
    if (!midValue.hasOneUse())
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

    GreedyRewriteConfig config;
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
