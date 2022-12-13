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
// ShapeHelper
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickOpShapeHelper::computeShape() {
  auto stickOp = llvm::dyn_cast<ZHighStickOp>(op);
  ZHighStickOp::Adaptor operandAdaptor(operands);

  // Output dims of result.
  DimsExpr outputDims;

  // Get operands and bounds.
  Value input = operandAdaptor.In();
  MemRefBoundsIndexCapture inBounds(input);
  int64_t rank = inBounds.getRank();

  for (int64_t i = 0; i < rank; ++i)
    outputDims.emplace_back(inBounds.getDim(i));

  // Direct stickify from NCHW to NHWC.
  if (isNHWCLayout(stickOp.layoutAttr())) {
    assert((rank == 4) && "Stickify input must have rank 4");
    // NCHW -> NHWC
    outputDims[0] = inBounds.getDim(0);
    outputDims[1] = inBounds.getDim(2);
    outputDims[2] = inBounds.getDim(3);
    outputDims[3] = inBounds.getDim(1);
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Value input = In();
  if (!hasRankedType(input))
    return success();

  auto inputType = input.getType().cast<RankedTensorType>();
  StringAttr layout = layoutAttr();
  Type elementType = inputType.getElementType();
  int64_t rank = inputType.getRank();

  ZTensorEncodingAttr::DataLayout dataLayout;
  if (layout)
    dataLayout = convertStringAttrToZTensorDataLayout(layout);
  else
    dataLayout = getZTensorDataLayoutByRank(rank);
  auto encoding = ZTensorEncodingAttr::get(this->getContext(), dataLayout);

  ZHighStickOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType, encoding);
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
