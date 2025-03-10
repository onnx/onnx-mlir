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
#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

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

void ZHighStickOp::build(OpBuilder &builder, OperationState &state, Value input,
    StringAttr layout, IntegerAttr saturation) {
  Type resType = builder.getNoneType();
  Type resElementType = builder.getF16Type();
  if (!mlir::isa<NoneType>(input.getType())) {
    ShapedType inputType = mlir::cast<ShapedType>(input.getType());
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
      resType = RankedTensorType::get(resShape, resElementType,
          ZTensorEncodingAttr::get(builder.getContext(), dataLayout));
    } else {
      resType = UnrankedTensorType::get(resElementType);
    }
  }
  build(builder, state, resType, input, layout, saturation);
}

//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickOpShapeHelper::computeShape() {
  auto stickOp = llvm::dyn_cast<ZHighStickOp>(op);
  ZHighStickOp::Adaptor operandAdaptor(operands);
  Value input = operandAdaptor.getIn();

  // Output dims of result.
  DimsExpr outputDims;

  // Get operands and bounds.
  SmallVector<IndexExpr, 4> inputDims;
  createIE->getShapeAsDims(input, inputDims);
  int64_t rank = inputDims.size();

  for (int64_t i = 0; i < rank; ++i)
    outputDims.emplace_back(inputDims[i]);

  // Direct stickify from NCHW to NHWC.
  if (isNHWCLayout(stickOp.getLayoutAttr())) {
    assert((rank == 4) && "Stickify input must have rank 4");
    // NCHW -> NHWC
    outputDims[0] = inputDims[0];
    outputDims[1] = inputDims[2];
    outputDims[2] = inputDims[3];
    outputDims[3] = inputDims[1];
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Value input = getIn();
  if (isa<NoneType>(input.getType()) || !hasRankedType(input))
    return success();

  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  StringAttr layout = getLayoutAttr();
  int64_t rank = inputType.getRank();

  ZTensorEncodingAttr::DataLayout dataLayout;
  if (layout)
    dataLayout = convertStringAttrToZTensorDataLayout(layout);
  else
    dataLayout = getZTensorDataLayoutByRank(rank);
  auto encoding = ZTensorEncodingAttr::get(this->getContext(), dataLayout);

  ZHighStickOpShapeHelper shapeHelper(getOperation());
  Type elementType =
      mlir::cast<ShapedType>(getResult().getType()).getElementType();
  return shapeHelper.computeShapeAndUpdateType(elementType, encoding);
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

void ZHighStickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<NoneTypeStickRemovalPattern>(context);
  results.insert<StickUnstickSameLayoutRemovalPattern>(context);
  results.insert<StickUnstickDiffLayoutRemovalPattern>(context);
  results.insert<Stick3DSSqueezeUnstick4DSPattern>(context);
  results.insert<ReplaceONNXLeakyReluPattern>(context);
  results.insert<ReplaceONNXSoftplusPattern>(context);
  results.insert<ReplaceONNXReciprocalSqrtPattern>(context);
  results.insert<ReshapeTransposeReshape2DTo3DSPattern>(context);
  results.insert<ReshapeTransposeReshape3DSTo2DPattern>(context);
  results.insert<ReshapeTransposeReshapeRoberta3DSWPattern1>(context);
  results.insert<ReshapeTransposeReshapeRoberta3DSWPattern2>(context);
}

} // namespace zhigh
} // namespace onnx_mlir
