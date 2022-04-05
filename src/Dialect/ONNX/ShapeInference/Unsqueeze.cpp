/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Unsqueeze.cpp - Shape Inference for Unsqueeze Op ----------===//
//
// This file implements shape inference for the ONNX Unsqueeze Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <typename ShapeHelper, typename OperandAdaptor>
LogicalResult ONNXUnsqueezeOpShapeHelperCommon(ShapeHelper *shapeHelper,
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

  int64_t outRank = dataRank + axes.size();
  for (int i = 0, j = 0; i < outRank || j < dataRank; ++i)
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      outputDims.emplace_back(LiteralIndexExpr(1));
    else
      outputDims.emplace_back(dataBounds.getDim(j++));

  // Save the final result.
  shapeHelper->dimsForOutput(0) = outputDims;

  return success();
}

ONNXUnsqueezeOpShapeHelper::ONNXUnsqueezeOpShapeHelper(ONNXUnsqueezeOp *newOp)
    : ONNXOpShapeHelper<ONNXUnsqueezeOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXUnsqueezeOpShapeHelper::ONNXUnsqueezeOpShapeHelper(ONNXUnsqueezeOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXUnsqueezeOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXUnsqueezeOpShapeHelper::computeShape(
    ONNXUnsqueezeOpAdaptor operandAdaptor) {
  auto axes = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (auto axesConstOp = getONNXConstantOp(axes)) {
    ArrayValueIndexCapture axesCapture(axes, fGetDenseVal, fLoadVal);
    axesCapture.getSymbolList(indexExprArray);
  } else if (!axes.getType().template isa<NoneType>()) {
    llvm_unreachable("dynamic axes not yet supported");
  }

  return ONNXUnsqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

ONNXUnsqueezeV11OpShapeHelper::ONNXUnsqueezeV11OpShapeHelper(
    ONNXUnsqueezeV11Op *newOp)
    : ONNXOpShapeHelper<ONNXUnsqueezeV11Op>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXUnsqueezeV11OpShapeHelper::ONNXUnsqueezeV11OpShapeHelper(
    ONNXUnsqueezeV11Op *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXUnsqueezeV11Op>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXUnsqueezeV11OpShapeHelper::computeShape(
    ONNXUnsqueezeV11OpAdaptor operandAdaptor) {
  auto axesAttr = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  ArrayAttributeIndexCapture axesCapture(axesAttr);
  auto axesRank = axesCapture.size();
  for (unsigned i = 0; i < axesRank; ++i) {
    indexExprArray.emplace_back(axesCapture.getLiteral(i));
  }
  return ONNXUnsqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

} // namespace onnx_mlir
