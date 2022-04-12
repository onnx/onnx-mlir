/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ Squeeze.cpp - Shape Inference for Squeeze Op ------------===//
//
// This file implements shape inference for the ONNX Squeeze Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <typename ShapeHelper, typename OperandAdaptor>
LogicalResult ONNXSqueezeOpShapeHelperCommon(ShapeHelper *shapeHelper,
    OperandAdaptor operandAdaptor, ArrayRef<IndexExpr> indexExprArray) {
  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t dataRank = data.getType().cast<ShapedType>().getShape().size();

  // Get axis values. They are expected to be normalized before so that there
  // is no negative values.
  SmallVector<int64_t, 4> axes;
  for (auto axisAttr : indexExprArray) {
    int64_t axis = axisAttr.getLiteral();
    assert(axis >= 0 && "Invalid axis");
    axes.emplace_back(axis);
  }

  for (int i = 0; i < dataRank; ++i)
    if (std::find(axes.begin(), axes.end(), i) == axes.end())
      outputDims.emplace_back(dataBounds.getDim(i));

  // Save the final result.
  shapeHelper->dimsForOutput() = outputDims;

  return success();
}

LogicalResult ONNXSqueezeOpShapeHelper::computeShape(
    ONNXSqueezeOpAdaptor operandAdaptor) {
  auto axes = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (auto axesConstOp = getONNXConstantOp(axes)) {
    ArrayValueIndexCapture axesCapture(axes, fGetDenseVal, fLoadVal);
    axesCapture.getSymbolList(indexExprArray);
  } else if (!axes.getType().template isa<NoneType>()) {
    llvm_unreachable("dynamic axes not yet supported");
  }

  return ONNXSqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

LogicalResult ONNXSqueezeV11OpShapeHelper::computeShape(
    ONNXSqueezeV11OpAdaptor operandAdaptor) {
  auto axesAttr = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (axesAttr.hasValue()) {
    ArrayAttributeIndexCapture axesCapture(axesAttr.getValue());
    auto axesRank = axesCapture.size();
    for (unsigned i = 0; i < axesRank; ++i) {
      indexExprArray.emplace_back(axesCapture.getLiteral(i));
    }
  }
  return ONNXSqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

} // namespace onnx_mlir
