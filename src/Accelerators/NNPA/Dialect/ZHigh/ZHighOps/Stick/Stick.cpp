/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Stick.cpp - ZHigh Operations ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/Stick/ONNXZHighStick.inc"
} // end anonymous namespace

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Custom builders
//===----------------------------------------------------------------------===//

void ZHighStickOp::build(
    OpBuilder &builder, OperationState &state, Value input, StringAttr layout) {
  Type resType = builder.getNoneType();
  if (!input.getType().isa<NoneType>()) {
    ShapedType inputType = input.getType().cast<ShapedType>();
    int64_t rank = -1;
    if (inputType.hasRank()) {
      rank = inputType.getRank();
      ZTensorEncodingAttr::DataLayout dataLayout;
      if (layout)
        dataLayout = convertStringAttrToZTensorDataLayout(layout);
      else {
        dataLayout = getZTensorDataLayoutByRank(rank);
        // Create a layout attribute.
        layout = convertZTensorDataLayoutToStringAttr(builder, dataLayout);
      }
      // Compute shape.
      ArrayRef<int64_t> inputShape = inputType.getShape();
      SmallVector<int64_t, 4> resShape(inputShape.begin(), inputShape.end());
      // Direct stickify from NCHW to NHWC.
      if (isNHWCLayout(layout)) {
        assert((inputShape.size() == 4) && "Input must have rank 4");
        // NCHW -> NHWC
        resShape[0] = inputShape[0];
        resShape[1] = inputShape[2];
        resShape[2] = inputShape[3];
        resShape[3] = inputShape[1];
      }
      resType = RankedTensorType::get(resShape, inputType.getElementType(),
          ZTensorEncodingAttr::get(builder.getContext(), dataLayout));
    } else {
      resType = UnrankedTensorType::get(inputType.getElementType());
    }
  }
  build(builder, state, resType, input, layout);
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(In()))
    return success();

  OpBuilder builder(this->getContext());
  ShapedType inputType = In().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  ZHighStickOpAdaptor operandAdaptor(*this);
  ZHighStickOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh Stick parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);

  StringAttr layout = layoutAttr();
  ZTensorEncodingAttr::DataLayout dataLayout;
  if (layout)
    dataLayout = convertStringAttrToZTensorDataLayout(layout);
  else
    dataLayout = getZTensorDataLayoutByRank(inputShape.size());

  updateType(getResult(), outputDims, inputType.getElementType(),
      ZTensorEncodingAttr::get(this->getContext(), dataLayout));
  return success();
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

void ZHighStickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<NoneTypeStickRemovalPattern>(context);
  results.insert<ReplaceONNXLeakyReluPattern>(context);
  results.insert<StickUnstickSameLayoutRemovalPattern>(context);
  results.insert<StickUnstickDiffLayoutRemovalPattern>(context);
}

} // namespace zhigh
} // namespace onnx_mlir
