/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Conv.cpp - Shape Inference for Conv Op ---------------===//
//
// This file implements shape inference for the ONNX Conv Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXConvOpShapeHelper::ONNXConvOpShapeHelper(
    ONNXConvOp *newOp, IndexExprScope *inScope)
    : ONNXGenericPoolShapeHelper<ONNXConvOp, ONNXConvOpAdaptor>(
          newOp, true /*hasFilter*/, false /*hasCeil*/, inScope) {}

ONNXConvOpShapeHelper::ONNXConvOpShapeHelper(ONNXConvOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXGenericPoolShapeHelper<ONNXConvOp, ONNXConvOpAdaptor>(newOp,
          true /*hasFilter*/, false /*hasCeil*/, rewriter, fGetDenseVal,
          fLoadVal, inScope) {}

LogicalResult ONNXConvOpShapeHelper::computeShape(
    ONNXConvOpAdaptor operandAdaptor) {
  return ONNXGenericPoolShapeHelper<ONNXConvOp,
      ONNXConvOpAdaptor>::computeShape(operandAdaptor, operandAdaptor.W(),
      op->kernel_shape(), op->pads(), op->strides(), op->dilations());
}

} // namespace onnx_mlir
