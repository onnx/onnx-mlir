/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Resize.cpp - Resize Op-------------------------------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers the ONNX Resize operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"

#include <cstdint>
#include <numeric>

using namespace mlir;

namespace onnx_mlir {

namespace {
struct ScaleHelper {
  ScaleHelper(
      int64_t numerator, int64_t denominator, int64_t offset, int64_t border)
      : numerator(numerator), denominator(denominator), offset(offset),
        border(border){};
  int64_t numerator, denominator, offset, border;
};

// Adapted from TFL to TOSA.
ScaleHelper normalize(int64_t output, int64_t input, bool pytorchHalfPixel,
    bool alignCorners, bool halfPixel, bool isNearest,
    bool isNearestModeFloor) {
  int64_t numerator, denominator, offset, border;
  // Test if pytorch_half_pixel needs special handling
  if (pytorchHalfPixel && output == 1) {
    numerator = 1;
    denominator = 1;
    offset = -1;
    border = denominator * (output - 1) - numerator * (input - 1) + offset;
    return ScaleHelper(numerator, denominator, offset, border);
  }

  // Apply if aligned and capable to be aligned.
  bool applyAligned = alignCorners && (output > 1);
  numerator = applyAligned ? (output - 1) : output;
  denominator = applyAligned ? (input - 1) : input;

  // Simplify the scalers, make sure they are even values.
  int gcd = std::gcd(numerator, denominator);
  numerator = 2 * numerator / gcd;
  denominator = 2 * denominator / gcd;

  // If half pixel centers we need to sample half a pixel inward.
  offset = halfPixel || pytorchHalfPixel ? (denominator - numerator) / 2 : 0;

  // If round_half_up we need to adjust the offset
  if (isNearest && isNearestModeFloor) {
    offset -= numerator / 2;
  }

  // We can compute this directly based on previous values.
  border = denominator * (output - 1) - numerator * (input - 1) + offset;
  return ScaleHelper(numerator, denominator, offset, border);
};

void valuesFromAxis(ArrayAttr *axis, llvm::SmallVectorImpl<int64_t> &axisVec) {
  auto axisRange = axis->getAsRange<IntegerAttr>();
  llvm::transform(axisRange, std::back_inserter(axisVec),
      [](IntegerAttr attr) { return getAxisInRange(attr.getInt(), 4, true); });
}

LogicalResult getScaleValue(ConversionPatternRewriter &rewriter, Operation *op,
    llvm::SmallVectorImpl<int64_t> &axisVec,
    llvm::SmallVectorImpl<float> &scaleVec, Value scaleValue) {
  mlir::ElementsAttr elementsAttr =
      getElementAttributeFromONNXValue(scaleValue);
  if (!elementsAttr)
    return rewriter.notifyMatchFailure(
        op, "Scale cannot come from a block argument.");

  // The axis attribute might permute and/or reduce the number of elements
  // in scaleValue. This reorders the scales to account for that.
  for (auto [index, value] : llvm::enumerate(axisVec))
    scaleVec[value] =
        elementsAttr.getValues<FloatAttr>()[index].getValueAsDouble();

  // Even if the shapes are identical, a scale value other than 1 is
  // possible. TOSA does not allow that for non-spatial dimensions.
  for (int64_t nonSpatialDimensions : {0, 1}) {
    if (scaleVec[nonSpatialDimensions] != 1)
      return rewriter.notifyMatchFailure(
          op, "Axis Attr if present must contain both output dimensions.");
  }
  return success();
}

class ONNXResizeOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXResizeOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXResizeOp::getOperationName(), 1, ctx) {}
  using OpAdaptor = typename ONNXResizeOp::Adaptor;

  struct FractionNumber {
    FractionNumber(float number) {
      double integral = std::floor(number);
      double frac = number - integral;

      const long precision = 1000000000; // This is the accuracy.

      long gcd_ = std::gcd((int)round(frac * precision), precision);

      long denominator = precision / gcd_;
      long numerator = round(frac * precision) / gcd_;

      this->numerator = numerator + denominator * integral;
      this->denominator = denominator;
    }
    int64_t numerator;
    int64_t denominator;
  };

  /// ## coordinateTransformationMode ##
  /// TOSA uses the formula ix = (ox * scale_x_d + offset_x) / scale_x_n
  /// to find the input coordinates. In order to lower ONNX one needs to
  /// express the modes in this context. Border is used to ensure that the shape
  /// calculation is exactly defined with an integer ratio. In order to fulfill
  /// the conditions, border is always border = d * (output - 1) - n *(input -
  /// 1) + offset
  ///
  /// ### half_pixel ###
  /// ONNX formula: ix = (ox + 0.5) / scale - 0.5
  /// gcd = greatest common divisor
  /// To meet TOSA requirements:
  /// - scale_x_d = input_size * 2 / gcd
  /// - scale_x_n = output_size * 2 / gcd
  /// - offset_x =  (scale_x_d - scale_x_n) / 2
  ///
  /// ### pytorch_half_pixel ###
  /// Same as half_pixel, but if output == 1:
  /// - scale_x_d = scale_x_n = 1
  /// - offset = -1
  ///
  /// ### half_pixel_symmetric ###
  /// NOT SUPPORTED BY TOSA, BECAUSE OF INT OFFSET
  /// Same as half_pixel, but with another offset
  /// - offset_x += symmetric_offset_x
  ///
  /// ### align_corners ###
  /// - scale_x_d = (input_size - 1)
  /// - scale_x_n = (output_size - 1)
  /// - offset = 0
  ///
  /// ### asymmetric ###
  /// - scale_x_d = input_size
  /// - scale_x_n = output_size
  /// - offset = 0
  ///
  /// ## nearest_mode ##
  /// If mode == nearest, then ONNX can set the nearest_mode attribute.
  /// The standard case for TOSA of round_half_up. Support for floor can
  /// be achieved with:
  /// offset_x -= n/2

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto resizeOp = llvm::cast<ONNXResizeOp>(op);
    Location loc = op->getLoc();
    OpAdaptor adaptor(operands, op->getAttrDictionary());

    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getX();
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());

    auto resultType =
        mlir::dyn_cast<RankedTensorType>(resizeOp.getResult().getType());

    StringRef mode = adaptor.getMode();
    StringRef nearestMode = adaptor.getNearestMode();
    StringRef coordinateTransformationMode =
        adaptor.getCoordinateTransformationMode();
    std::optional<ArrayAttr> axis = adaptor.getAxes();
    int64_t antialias = adaptor.getAntialias();

    if (!inputType || inputType.getRank() != 4) {
      return rewriter.notifyMatchFailure(
          resizeOp, "TOSA only support 4D tensors as input of resize.");
    }

    // With only static dimensions, scales and sizes as inputs are not relevant
    // anymore.
    if (inputType.isDynamicDim(2) || inputType.isDynamicDim(3)) {
      return rewriter.notifyMatchFailure(
          resizeOp, "Only static sized tensors are supported.");
    }

    auto elementType = inputType.getElementType();
    if (!(isTOSAFloat(elementType) || isTOSASignedInt(elementType))) {
      return rewriter.notifyMatchFailure(
          resizeOp, "Element type is not supported by TOSA.");
    }

    if (mode == "cubic") {
      return rewriter.notifyMatchFailure(
          resizeOp, "TOSA does not support cubic interpolation.");
    }

    if (mode == "nearest" &&
        (nearestMode == "ceil" || nearestMode == "round_prefer_floor")) {
      return rewriter.notifyMatchFailure(resizeOp,
          "TOSA does not support ceil and round_prefer_floor as nearestMode.");
    }

    // This also makes roi as an input irrelevant.
    if (coordinateTransformationMode == "tf_crop_and_resize") {
      return rewriter.notifyMatchFailure(
          resizeOp, "TOSA does not support tf_crop_and_resize.");
    }

    if (antialias != 0)
      return rewriter.notifyMatchFailure(
          op, "TOSA does not support antialiasing.");

    auto inputShape = inputType.getShape();
    auto outputShape = resultType.getShape();

    if (inputShape[0] != outputShape[0] || inputShape[1] != outputShape[1])
      return rewriter.notifyMatchFailure(
          op, "Cannot resize non spatial dimensions.");

    // Get axis values if set. Default is all axis in normal order.
    llvm::SmallVector<int64_t> axisVec;
    if (axis.has_value()) {
      valuesFromAxis(&axis.value(), axisVec);
    } else {
      axisVec.append({0, 1, 2, 3});
    }

    // Set these explicitly just out of convenience.
    int64_t inputHeight = inputShape[2];
    int64_t inputWidth = inputShape[3];
    int64_t outputHeight = outputShape[2];
    int64_t outputWidth = outputShape[3];

    // Check if scales are set. We need to get those float values, because they
    // make a difference in linear interpolation.
    Value scaleValue = resizeOp.getScales();
    llvm::SmallVector<float, 4> scales{1, 1, 1, 1};
    if (!isNoneValue(scaleValue)) {
      if (getScaleValue(rewriter, op, axisVec, scales, scaleValue).failed())
        return rewriter.notifyMatchFailure(
            op, "Could not retrieve scale values.");

      // In TOSA the scale is a fraction of two integer numbers.
      FractionNumber height(scales[2]);
      FractionNumber width(scales[3]);
      outputHeight = height.numerator;
      inputHeight = height.denominator;
      outputWidth = width.numerator;
      inputWidth = width.denominator;
    }

    bool alignCorners = coordinateTransformationMode == "align_corners";
    bool halfPixel = coordinateTransformationMode == "half_pixel";
    bool pytorchHalfPixel =
        coordinateTransformationMode == "pytorch_half_pixel";
    bool halfPixelSymmetric =
        coordinateTransformationMode == "half_pixel_symmetric";
    bool isBilinear = mode == "linear";
    bool isNearest = mode == "nearest";
    bool isNearestModeFloor = nearestMode == "floor";
    StringRef resizeMode = isBilinear ? "BILINEAR" : "NEAREST_NEIGHBOR";

    if (halfPixelSymmetric)
      return rewriter.notifyMatchFailure(op,
          "TOSA does not support float offsets which are required "
          "for symmetric mode.");

    ScaleHelper yDimension =
        normalize(outputHeight, inputHeight, pytorchHalfPixel, alignCorners,
            halfPixel, isNearest, isNearestModeFloor);
    ScaleHelper xDimension =
        normalize(outputWidth, inputWidth, pytorchHalfPixel, alignCorners,
            halfPixel, isNearest, isNearestModeFloor);

    // Convert input [N,IC,IH,IW] -> [N,IH,IW,IC]
    Value newInput = tosaBuilder.transpose(input, {0, 2, 3, 1});

    // Create resizeOp
    auto scale = rewriter.getDenseI64ArrayAttr({yDimension.numerator,
        yDimension.denominator, xDimension.numerator, xDimension.denominator});
    auto offset =
        rewriter.getDenseI64ArrayAttr({yDimension.offset, xDimension.offset});
    auto border =
        rewriter.getDenseI64ArrayAttr({yDimension.border, xDimension.border});
    auto resizeModeAttr = rewriter.getStringAttr(resizeMode);
    Type newOutputType =
        RankedTensorType::get(llvm::SmallVector<int64_t, 4>(
                                  inputType.getRank(), ShapedType::kDynamic),
            mlir::cast<ShapedType>(resultType).getElementType());

    Value resize = tosa::CreateOpAndInfer<mlir::tosa::ResizeOp>(rewriter, loc,
        newOutputType, newInput, scale, offset, border, resizeModeAttr);

    // Convert output [N,OH,OW,OC] -> [N,OC,OH,OW]
    Value newOutput = tosaBuilder.transpose(resize, {0, 3, 1, 2});

    rewriter.replaceOp(resizeOp, newOutput);

    return success();
  }
};

} // namespace

void populateLoweringONNXResizeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXResizeOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir