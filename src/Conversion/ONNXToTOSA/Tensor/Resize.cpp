/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Resize.cpp - Resize Op-----------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX Resize operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <numeric>
#include <src/Dialect/Mlir/IndexExpr.hpp>
#include <src/Dialect/ONNX/ONNXOps/OpHelper.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXResizeOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXResizeOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXResizeOp::getOperationName(), 1, ctx) {}
  using OpAdaptor = typename ONNXResizeOp::Adaptor;

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto resizeOp = llvm::cast<ONNXResizeOp>(op);
    Location loc = op->getLoc();
    OpAdaptor adaptor(operands, op->getAttrDictionary());

    Value input = adaptor.X();
    auto inputType = input.getType().dyn_cast<RankedTensorType>();

    auto resultType =
        resizeOp.getResult().getType().dyn_cast<RankedTensorType>();

    StringRef mode = adaptor.mode();
    StringRef nearestMode = adaptor.nearest_mode();
    StringRef coordinateTransformationMode =
        adaptor.coordinate_transformation_mode();
    int64_t excludeOutside = adaptor.exclude_outside();

    if (inputType.getRank() != 4) {
      return rewriter.notifyMatchFailure(
          resizeOp, "TOSA only support 4D tensors as input of resize.");
    }

    if (mode == "cubic") {
      return rewriter.notifyMatchFailure(
          resizeOp, "TOSA does not support cubic interpolation.");
    }

    if (nearestMode == "ceil" || nearestMode == "round_prefer_floor") {
      return rewriter.notifyMatchFailure(resizeOp,
          "TOSA does not support ceil and round_prefer_floor as nearestMode.");
    }

    // TODO: Maybe it does support it with border? Investigate on this.
    if (excludeOutside) {
      return rewriter.notifyMatchFailure(
          resizeOp, "TOSA does not support excludeOutside.");
    }

    // TODO: special handling for pytorch_half_pixel
    if (coordinateTransformationMode == "tf_crop_and_resize" ||
        coordinateTransformationMode == "pytorch_half_pixel") {
      return rewriter.notifyMatchFailure(resizeOp,
          "TOSA does not support tf_crop_and_resize or pytorch_half_pixel.");
    }

    // Convert input [N,IC,IH,IW] -> [N,IH,IW,IC]
    Value newInput = tosa::createTosaTransposedTensor(
        rewriter, resizeOp, input, {0, 2, 3, 1});

    bool alignCorners = coordinateTransformationMode == "align_corners";
    bool halfPixel = coordinateTransformationMode == "half_pixel";
    bool isBilinear = mode == "linear";
    bool isNearest = mode == "nearest";
    bool floor = nearestMode == "floor";
    StringRef resizeMode = isBilinear ? "BILINEAR" : "NEAREST_NEIGHBOR";

    auto inputShape = inputType.getShape();
    auto outputShape = resultType.getShape();

    int64_t inputHeight = inputShape[2];
    int64_t inputWidth = inputShape[3];
    int64_t outputHeight = outputShape[2];
    int64_t outputWidth = outputShape[3];

    // Align corners sets the scaling ratio to (OH - 1)/(IH - 1)
    // rather than OH / IH. Similarly for width.
    auto normalize = [&](int64_t input, int64_t output, int64_t &n, int64_t &d,
                         int64_t &offset, int64_t &border) {
      // Dimension is length 1, we are just sampling from one value.
      if (input == 1) {
        n = 1;
        d = 1;
        offset = 0;
        border = output - 1;
        return;
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
      offset = halfPixel ? (d - n) / 2 : 0;
      // If round_half_up we need to adjust the offset
      if (floor) {
        offset -= n / 2;
      }

      // If nearest neighbours we need to guarantee we round up.
      if (isNearest && alignCorners) {
        offset += n / 2;
      }

      if (isBilinear && halfPixel) {
        offset -= n / 2;
      }

      // We can compute this directly based on previous values.
      border = d * (output - 1) - n * (input - 1) + offset;
    };

    int64_t scale_y_n, scale_y_d, offset_y, border_y;
    int64_t scale_x_n, scale_x_d, offset_x, border_x;
    normalize(
        inputHeight, outputHeight, scale_y_n, scale_y_d, offset_y, border_y);
    normalize(
        inputWidth, outputWidth, scale_x_n, scale_x_d, offset_x, border_x);

    auto scale =
        rewriter.getI64ArrayAttr({scale_y_n, scale_y_d, scale_x_n, scale_x_d});
    auto offset = rewriter.getI64ArrayAttr({offset_y, offset_x});
    auto border = rewriter.getI64ArrayAttr({border_y, border_x});
    auto resizeModeAttr = rewriter.getStringAttr(resizeMode);
    Type newOutputType = RankedTensorType::get(
        {-1, -1, -1, -1}, resultType.cast<ShapedType>().getElementType());

    Value resize = tosa::CreateOpAndInfer<mlir::tosa::ResizeOp>(rewriter, loc,
        newOutputType, newInput, scale, offset, border, resizeModeAttr);

    // Convert output [N,OH,OW,OC] -> [N,OC,OH,OW]
    Value newOutput = tosa::createTosaTransposedTensor(
        rewriter, resizeOp, resize, {0, 3, 1, 2});

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