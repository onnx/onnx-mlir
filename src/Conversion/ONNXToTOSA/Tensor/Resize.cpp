/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Resize.cpp - Resize Op-------------------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
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

class ONNXResizeOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXResizeOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXResizeOp::getOperationName(), 1, ctx) {}
  using OpAdaptor = typename ONNXResizeOp::Adaptor;

  /// ## coordinateTransformationMode ##
  /// TOSA uses the formula ix = (ox * scale_x_d + offset_x) / scale_x_n
  /// to find the input coordinates. In order to lower ONNX one needs to
  /// express the modes in this context. Border is only used to check if certain
  /// conditions in the dimensions are met, but has no actual use in calculating
  /// something. It is probably an error in the TOSA specification. In order to
  /// fulfill the conditions, border is always
  /// border = d * (output - 1) - n *(input - 1) + offset
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
    auto inputType = input.getType().dyn_cast<RankedTensorType>();

    auto resultType =
        resizeOp.getResult().getType().dyn_cast<RankedTensorType>();

    StringRef mode = adaptor.getMode();
    StringRef nearestMode = adaptor.getNearestMode();
    StringRef coordinateTransformationMode =
        adaptor.getCoordinateTransformationMode();

    if (inputType.getRank() != 4) {
      return rewriter.notifyMatchFailure(
          resizeOp, "TOSA only support 4D tensors as input of resize.");
    }

    // With only static dimensions, scales and sizes as inputs are not relevant
    // anymore.
    if (inputType.isDynamicDim(2) || inputType.isDynamicDim(3)) {
      return rewriter.notifyMatchFailure(
          resizeOp, "Only static sized tensors are supported.");
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

    // Convert input [N,IC,IH,IW] -> [N,IH,IW,IC]
    Value newInput = tosaBuilder.transpose(input, {0, 2, 3, 1});

    bool alignCorners = coordinateTransformationMode == "align_corners";
    bool halfPixel = coordinateTransformationMode == "half_pixel";
    bool pytorchHalfPixel =
        coordinateTransformationMode == "pytorch_half_pixel";
    bool isBilinear = mode == "linear";
    bool isNearest = mode == "nearest";
    bool isNearestModeFloor = nearestMode == "floor";
    StringRef resizeMode = isBilinear ? "BILINEAR" : "NEAREST_NEIGHBOR";

    auto inputShape = inputType.getShape();
    auto outputShape = resultType.getShape();

    int64_t inputHeight = inputShape[2];
    int64_t inputWidth = inputShape[3];
    int64_t outputHeight = outputShape[2];
    int64_t outputWidth = outputShape[3];

    // Adapted from TFL to TOSA.
    // Align corners sets the scaling ratio to (OH - 1)/(IH - 1)
    // rather than OH / IH. Similarly for width.

    auto normalize = [=](int64_t input, int64_t output) {
      // Dimension is length 1, we are just sampling from one value.
      int64_t n, d, offset, border;
      if (input == 1) {
        n = output;
        d = 1;
        offset = 0;
        border = output - 1;
        return std::make_tuple(n, d, offset, border);
      }

      // Test if pytorch_half_pixel needs special handling
      if (pytorchHalfPixel && output == 1) {
        n = 1;
        d = 1;
        offset = -1;
        border = d * (output - 1) - n * (input - 1) + offset;
        return std::make_tuple(n, d, offset, border);
      }

      // Apply if aligned and capable to be aligned.
      bool applyAligned = alignCorners && (output > 1);
      n = applyAligned ? (output - 1) : output;
      d = applyAligned ? (input - 1) : input;

      // Simplify the scalers, make sure they are even values.
      int gcd = std::gcd(n, d);
      n = 2 * n / gcd;
      d = 2 * d / gcd;

      // If half pixel centers we need to sample half a pixel inward.
      offset = halfPixel || pytorchHalfPixel ? (d - n) / 2 : 0;
      // If round_half_up we need to adjust the offset
      if (isNearest && isNearestModeFloor) {
        offset -= n / 2;
      }

      // We can compute this directly based on previous values.
      border = d * (output - 1) - n * (input - 1) + offset;
      return std::make_tuple(n, d, offset, border);
    };

    auto [scale_y_n, scale_y_d, offset_y, border_y] =
        normalize(inputHeight, outputHeight);
    auto [scale_x_n, scale_x_d, offset_x, border_x] =
        normalize(inputWidth, outputWidth);

    auto scale = rewriter.getDenseI64ArrayAttr(
        {scale_y_n, scale_y_d, scale_x_n, scale_x_d});
    auto offset = rewriter.getDenseI64ArrayAttr({offset_y, offset_x});
    auto border = rewriter.getDenseI64ArrayAttr({border_y, border_x});
    auto resizeModeAttr = rewriter.getStringAttr(resizeMode);
    Type newOutputType =
        RankedTensorType::get(llvm::SmallVector<int64_t, 4>(
                                  inputType.getRank(), ShapedType::kDynamic),
            resultType.cast<ShapedType>().getElementType());

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