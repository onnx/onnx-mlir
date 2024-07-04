/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Reduce.cpp - ReduceMax Op --------------------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX reduce operators to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Expect the reduction op to have following configuration:
//   Inputs: Data, Axes
//   Attrs: KeepDims, noop_with_emty_axes
template <typename ONNXReduceOp>
DenseIntElementsAttr getAxesLatestsVersionAttr(ONNXReduceOp op) {
  typename ONNXReduceOp::Adaptor adaptor(op);

  Value input = adaptor.getData();
  Value axesValue = adaptor.getAxes();
  int64_t noOpIfAxesEmpty = adaptor.getNoopWithEmptyAxes();

  // axes is mandatory for tosa
  SmallVector<int64_t> targetAxes;
  if (isNoneValue(axesValue)) {
    if (noOpIfAxesEmpty == 0) {
      // Default behaviour when "axes" is none and "noop_with_empty_axes" is
      // set to false, it is to reduce all dims
      const int64_t numberOfAxes = input.getType().cast<ShapedType>().getRank();
      auto iotaRange =
          llvm::iota_range<int64_t>(0, numberOfAxes, /*Inclusive=*/false);
      targetAxes = SmallVector<int64_t>(iotaRange.begin(), iotaRange.end());
    } else {
      assert(noOpIfAxesEmpty == 1 &&
             "noop_with_empty_axes can only be either 0 or 1");
      // If "axes" is none and "noop_with_empty_axes" is true, then this
      // behaves as an identity operator, no reduction is performed and shape
      // is the same as the input. This is handed by later function just return
      // an empty axis array
    }
  } else if (axesValue.getDefiningOp<ONNXConstantOp>() ||
             axesValue.getDefiningOp<mlir::tosa::ConstOp>()) {
    // "axes" are specified, retrieve
    auto axesValues =
        tosa::getElementsAttrFromConst(axesValue).getValues<int64_t>();
    targetAxes = SmallVector<int64_t>(axesValues.begin(), axesValues.end());
  } else {
    return {};
  }

  const int64_t numTargetAxes = targetAxes.size();
  auto i64Ty =
      IntegerType::get(input.getContext(), /*width=*/64, IntegerType::Signless);
  return DenseIntElementsAttr::get(
      RankedTensorType::get({numTargetAxes}, i64Ty), targetAxes);
}

// Expect the reduction op to have following configuration:
//   Inputs: Data
//   Attrs: KeepDims, axes
template <typename ONNXReduceOp>
DenseIntElementsAttr getAxesLegacyVersionAttr(ONNXReduceOp op) {
  typename ONNXReduceOp::Adaptor adaptor(op);

  Value input = adaptor.getData();
  auto axes = adaptor.getAxes();

  // axes is mandatory for tosa
  SmallVector<int64_t> targetAxes;
  if (!axes) {
    // if not present all axes are reduced
    const int64_t numberOfAxes = input.getType().cast<ShapedType>().getRank();
    auto iotaRange =
        llvm::iota_range<int64_t>(0, numberOfAxes, /*Inclusive=*/false);
    targetAxes = SmallVector<int64_t>(iotaRange.begin(), iotaRange.end());
  } else {
    targetAxes = extractFromIntegerArrayAttr<int64_t>(axes.value());
  }

  const int64_t numTargetAxes = targetAxes.size();
  auto i64Ty =
      IntegerType::get(input.getContext(), /*width=*/64, IntegerType::Signless);
  return DenseIntElementsAttr::get(
      RankedTensorType::get({numTargetAxes}, i64Ty), targetAxes);
}

template <typename ONNXReduceOp,
    LogicalResult (*lowerFn)(ONNXReduceOp op, RankedTensorType inputType,
        RankedTensorType outputType, ConversionPatternRewriter &rewriter)>
class ONNXReduceOpLoweringToTOSA : public OpConversionPattern<ONNXReduceOp> {
public:
  using OpConversionPattern<ONNXReduceOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXReduceOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto inputType =
        adaptor.getData().getType().template dyn_cast<RankedTensorType>();
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input type not a ranked tensor.");

    auto outputType = this->getTypeConverter()
                          ->convertType(op.getResult().getType())
                          .template cast<RankedTensorType>();

    return (*lowerFn)(op, inputType, outputType, rewriter);
  }
};

template <typename ONNXOp_t, typename TosaOp_t>
LogicalResult reduceLatestVersionLowering(ONNXOp_t op,
    RankedTensorType inputType, RankedTensorType outputType,
    ConversionPatternRewriter &rewriter) {
  typename ONNXOp_t::Adaptor adaptor(op);

  Value val = onnx_mlir::tosa::convertReduceOpCommon<TosaOp_t>(rewriter, op,
      outputType, adaptor.getData(), inputType, getAxesLatestsVersionAttr(op),
      adaptor.getKeepdims());

  // Shape inference is handled by the helper functions
  rewriter.replaceOp(op, {val});
  return success();
}

template <typename ONNXOp_t, typename TosaOp_t>
LogicalResult reduceLegacyVersionsLowering(ONNXOp_t op,
    RankedTensorType inputType, RankedTensorType outputType,
    ConversionPatternRewriter &rewriter) {
  typename ONNXOp_t::Adaptor adaptor(op);
  Value val = onnx_mlir::tosa::convertReduceOpCommon<TosaOp_t>(rewriter, op,
      outputType, adaptor.getData(), inputType, getAxesLegacyVersionAttr(op),
      adaptor.getKeepdims());

  // Shape inference is handled by the helper functions
  rewriter.replaceOp(op, {val});
  return success();
}

LogicalResult reduceMeanLowering(ONNXReduceMeanOp op,
    RankedTensorType inputType, RankedTensorType outputType,
    ConversionPatternRewriter &rewriter) {
  typename ONNXReduceMeanOp::Adaptor adaptor(op);
  auto newAxesAttr = getAxesLatestsVersionAttr(op);
  if (!newAxesAttr) {
    return rewriter.notifyMatchFailure(op, "cannot convert with dynamic axis");
  }
  // reduce_mean is lowered as followed:
  // op1 = reduce_sum(input)
  // op2 = mul(op1, 1.0 / num_elements_on_reduced_axis)
  auto keepDims = adaptor.getKeepdims();
  int64_t inputRank = inputType.getRank();
  int64_t numElemsOnReducedAxis = 1;
  for (int i = 0; i < newAxesAttr.getNumElements(); i++) {
    int64_t axisVal = newAxesAttr.getValues<mlir::IntegerAttr>()[i].getInt();
    if (axisVal < 0)
      axisVal += inputRank;
    numElemsOnReducedAxis *= inputType.getShape()[axisVal];
  }
  double divScale = 1.0 / static_cast<double>(numElemsOnReducedAxis);

  Value val =
      onnx_mlir::tosa::convertReduceOpCommon<mlir::tosa::ReduceSumOp>(rewriter,
          op, outputType, adaptor.getData(), inputType, newAxesAttr, keepDims);

  TosaBuilder tosaBuilder(rewriter, op->getLoc());
  Value divConst = tosaBuilder.getSplattedConst(
      divScale, outputType.getElementType(), outputType.getShape());
  auto output = tosaBuilder.mul(val, divConst);

  if (!output) {
    return rewriter.notifyMatchFailure(op, "could not be converted");
  }
  // Shape inference is handled by the helper functions
  rewriter.replaceOp(op, {output});
  return success();
}

LogicalResult reduceMeanV13Lowering(ONNXReduceMeanV13Op op,
    RankedTensorType /*inputType*/, RankedTensorType outputType,
    ConversionPatternRewriter &rewriter) {
  typename ONNXReduceMeanV13Op::Adaptor adaptor(op);
  auto newAxesAttr = getAxesLegacyVersionAttr(op);

  auto keepDims = adaptor.getKeepdims();
  TosaBuilder tosaBuilder(rewriter, op->getLoc());
  auto output = tosa::convertReduceMeanOp(rewriter, op, tosaBuilder, outputType,
      adaptor.getData(), newAxesAttr, keepDims);

  if (!output) {
    return rewriter.notifyMatchFailure(op, "Could not be converted");
  }
  // Shape inference is handled by the helper functions
  rewriter.replaceOp(op, {output.value()});
  return success();
}

} // namespace

#define DECLARE_ONE_TO_ONE_LOWERING(ONNXOp, TOSAOp)                            \
  using ONNXOp##LoweringToTOSA = ONNXReduceOpLoweringToTOSA<ONNXOp,            \
      reduceLatestVersionLowering<ONNXOp, TOSAOp>>
// Covers versions 20(latests)-18
DECLARE_ONE_TO_ONE_LOWERING(ONNXReduceMinOp, mlir::tosa::ReduceMinOp);
// Covers versions 20(latests)-18
DECLARE_ONE_TO_ONE_LOWERING(ONNXReduceMaxOp, mlir::tosa::ReduceMaxOp);
// Covers versions 13 (latests)
DECLARE_ONE_TO_ONE_LOWERING(ONNXReduceProdOp, mlir::tosa::ReduceProdOp);
// Covers versions 18 (latests)
DECLARE_ONE_TO_ONE_LOWERING(ONNXReduceSumOp, mlir::tosa::ReduceSumOp);
// Covers versions 18 (latests)
using ONNXReduceMeanOpLoweringToTOSA =
    ONNXReduceOpLoweringToTOSA<ONNXReduceMeanOp, reduceMeanLowering>;
#undef DECLARE_ONE_TO_ONE_LOWERING

#define DECLARE_ONE_TO_ONE_LEGACY_LOWERING(ONNXOp, TOSAOp)                     \
  using ONNXOp##LegacyLoweringToTOSA = ONNXReduceOpLoweringToTOSA<ONNXOp,      \
      reduceLegacyVersionsLowering<ONNXOp, TOSAOp>>
// Covers versions 13-12-11
DECLARE_ONE_TO_ONE_LEGACY_LOWERING(ONNXReduceMinV13Op, mlir::tosa::ReduceMinOp);
// Covers versions 13-12-11
DECLARE_ONE_TO_ONE_LEGACY_LOWERING(ONNXReduceMaxV13Op, mlir::tosa::ReduceMaxOp);
// Covers version 11
DECLARE_ONE_TO_ONE_LEGACY_LOWERING(ONNXReduceSumV11Op, mlir::tosa::ReduceSumOp);
// Covers version 13-11
DECLARE_ONE_TO_ONE_LEGACY_LOWERING(
    ONNXReduceProdV13Op, mlir::tosa::ReduceProdOp);
// Covers version 13-11
using ONNXReduceMeanV13LegacyLoweringToTOSA =
    ONNXReduceOpLoweringToTOSA<ONNXReduceMeanV13Op, reduceMeanV13Lowering>;
#undef DECLARE_ONE_TO_ONE_LEGACY_LOWERING

void populateLoweringONNXReduceOpsToTOSAPattern(ConversionTarget & /*target*/,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXReduceMaxOpLoweringToTOSA>(typeConverter, ctx);
  patterns.insert<ONNXReduceMeanOpLoweringToTOSA>(typeConverter, ctx);
  patterns.insert<ONNXReduceMinOpLoweringToTOSA>(typeConverter, ctx);
  patterns.insert<ONNXReduceProdOpLoweringToTOSA>(typeConverter, ctx);
  patterns.insert<ONNXReduceSumOpLoweringToTOSA>(typeConverter, ctx);

  patterns.insert<ONNXReduceMaxV13OpLegacyLoweringToTOSA>(typeConverter, ctx);
  patterns.insert<ONNXReduceMeanV13LegacyLoweringToTOSA>(typeConverter, ctx);
  patterns.insert<ONNXReduceMinV13OpLegacyLoweringToTOSA>(typeConverter, ctx);
  patterns.insert<ONNXReduceProdV13OpLegacyLoweringToTOSA>(typeConverter, ctx);
  patterns.insert<ONNXReduceSumV11OpLegacyLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
