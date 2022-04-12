/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AveragePool.cpp - Shape Inference for AveragePool Op ---------===//
//
// This file implements shape inference for the ONNX AveragePool Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXAveragePoolOpShapeHelper::ONNXAveragePoolOpShapeHelper(
    ONNXAveragePoolOp *newOp, IndexExprScope *inScope)
    : ONNXGenericPoolShapeHelper<ONNXAveragePoolOp, ONNXAveragePoolOpAdaptor>(
          newOp, false /*hasFilter*/, newOp->ceil_mode(), inScope) {}

ONNXAveragePoolOpShapeHelper::ONNXAveragePoolOpShapeHelper(
    ONNXAveragePoolOp *newOp, mlir::OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXGenericPoolShapeHelper<ONNXAveragePoolOp, ONNXAveragePoolOpAdaptor>(
          newOp, false /*hasFilter*/, newOp->ceil_mode(), rewriter,
          fGetDenseVal, fLoadVal, inScope) {}

LogicalResult ONNXAveragePoolOpShapeHelper::computeShape(
    ONNXAveragePoolOpAdaptor operandAdaptor) {
  return ONNXGenericPoolShapeHelper<ONNXAveragePoolOp,
      ONNXAveragePoolOpAdaptor>::computeShape(operandAdaptor, nullptr,
      op->kernel_shape(), op->pads(), op->strides(), None);
}

} // namespace onnx_mlir
