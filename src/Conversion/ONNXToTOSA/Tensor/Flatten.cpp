/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Flatten.cpp - Flatten Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX ReshapeOp to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXFlattenLoweringToTOSA : public OpConversionPattern<ONNXFlattenOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXFlattenOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXFlattenOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getInput();

    // tosa.reshape does not allow a dynamic entry in the new_shape attribute
    if (!hasStaticShape(input.getType()))
      return rewriter.notifyMatchFailure(
          op, "only static shapes are supported");

    auto loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, loc);

    int64_t axis = adaptor.getAxis();
    auto inputType = cast<ShapedType>(input.getType());

    // onnx allows values beetween [-r, r] where r is the rank.
    if (axis == inputType.getRank()) {
      // axis == rank is valid for Flatten
    } else {
      // check if the axis is in range [-r, r-1] where r is the rank
      axis = tosa::convertNegativeAxis(axis, inputType.getRank());
    }

    llvm::SmallVector<int64_t, 4> newShape;
    auto inputShape = inputType.getShape();
    if (axis == 0) {
      newShape.push_back(1);
      int64_t lastShape = 1;
      for (const int64_t axis : inputShape) {
        lastShape *= axis;
      }
      newShape.push_back(lastShape);
    } else {
      int64_t firstShape = 1;
      for (int i = 0; i < axis; i++) {
        firstShape *= inputShape[i];
      }
      newShape.push_back(firstShape);
      int64_t secondShape = 1;
      for (int i = axis; i < inputType.getRank(); i++) {
        secondShape *= inputShape[i];
      }
      newShape.push_back(secondShape);
    }
    Value reshapeOp = tosaBuilder.reshape(input, newShape);

    rewriter.replaceOp(op, reshapeOp);

    return success();
  }
};

} // namespace

void populateLoweringONNXFlattenOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXFlattenLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
