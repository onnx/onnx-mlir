/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Transpose.cpp - Shape Inference for Transpose Op ----------===//
//
// This file implements shape inference for the ONNX Transpose Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXTransposeOpShapeHelper::computeShape(
    ONNXTransposeOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Basic information.
  auto rank = operandAdaptor.data().getType().cast<ShapedType>().getRank();

  // Transposition which handles the default case of
  // reversing the shape of the tensor (similar to numpy.transpose).
  ArrayAttr permAttr = op->permAttr();
  if (!permAttr) {
    // Generate reverse order for default transpose operation.
    SmallVector<int64_t, 4> defaultVals;
    auto builder = mlir::Builder(op->getContext());
    for (int i = rank - 1; i >= 0; --i)
      defaultVals.emplace_back(i);
    // Set default attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    op->permAttr(builder.getI64ArrayAttr(defaultRefs));
    permAttr = op->permAttr();
  }

  // Perform transposition according to perm attribute.
  DimsExpr transposedDims;
  MemRefBoundsIndexCapture dataBounds(operandAdaptor.data());
  for (decltype(rank) i = 0; i < rank; ++i) {
    int64_t inputIndex = ArrayAttrIntVal(permAttr, i);
    transposedDims.emplace_back(dataBounds.getDim(inputIndex));
  }

  // Set type for the first output.
  dimsForOutput() = transposedDims;
  return success();
}

} // namespace onnx_mlir
