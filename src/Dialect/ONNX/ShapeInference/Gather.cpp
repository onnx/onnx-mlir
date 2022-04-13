/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ Gather.cpp - Shape Inference for Gather Op --------------===//
//
// This file implements shape inference for the ONNX Gather Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXGatherOpShapeHelper::ONNXGatherOpShapeHelper(
    ONNXGatherOp *newOp, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXGatherOp>(
          newOp, newOp->getOperation()->getNumResults(), inScope),
      dataDims(), indicesDims(), positiveConstantIndices(false) {}

ONNXGatherOpShapeHelper::ONNXGatherOpShapeHelper(ONNXGatherOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXGatherOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal, inScope),
      dataDims(), indicesDims(), positiveConstantIndices(false) {}

LogicalResult ONNXGatherOpShapeHelper::computeShape(
    ONNXGatherOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Read data and indices shapes as dim indices.
  MemRefBoundsIndexCapture dataBounds(operandAdaptor.data());
  MemRefBoundsIndexCapture indicesBounds(operandAdaptor.indices());
  dataBounds.getDimList(dataDims);
  indicesBounds.getDimList(indicesDims);

  // Read constant 'axis' attribute and normalize when negative.
  int64_t axisIndex = op->axis();
  // The 'axis' value must be in [-rank, rank-1].
  int dataRank = dataDims.size();
  if (axisIndex < -dataRank || axisIndex >= dataRank)
    return op->emitError("Gather axis value out of bound");
  // Convert a negative axis to a positive axis.
  if (axisIndex < 0) {
    axisIndex += dataRank;
    auto builder = mlir::Builder(op->getContext());
    op->axisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisIndex, /*isSigned=*/true)));
  }

  // If 'indices' is a constant tensor, check whether its values are valid.
  if (dataDims[axisIndex].isLiteral()) {
    auto valueAttribute = fGetDenseVal(operandAdaptor.indices());
    if (valueAttribute) {
      int64_t dataDimAtAxis = dataDims[axisIndex].getLiteral();
      positiveConstantIndices = true;
      for (auto value : valueAttribute.getValues<IntegerAttr>()) {
        auto index = value.cast<IntegerAttr>().getInt();
        if (index < -dataDimAtAxis || index >= dataDimAtAxis)
          return op->emitError("Indices tensor contains an out-of-bound index");
        if (index < 0)
          // TODO: make the negative consant number positive.
          positiveConstantIndices = false;
      }
    }
  }

  // Output has rank of 'indicesRank + (dataRank - 1).
  // Output shape is constructed from 'input' by:
  //    replacing the dimension at 'axis' in 'input' by the shape of
  //    'indices'.
  for (int i = 0; i < dataRank; ++i) {
    if (i == axisIndex)
      for (IndexExpr d : indicesDims)
        dimsForOutput().emplace_back(d);
    else
      dimsForOutput().emplace_back(dataDims[i]);
  }

  return success();
}

} // namespace onnx_mlir
