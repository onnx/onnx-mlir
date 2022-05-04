/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ Reshape.cpp - Shape Inference for Reshape Op ------------===//
//
// This file implements shape inference for the ONNX Reshape Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXReshapeOpShapeHelper::computeShape(
    ONNXReshapeOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t dataRank = data.getType().cast<ShapedType>().getShape().size();

  // Get info about shape operand.
  Value shape = operandAdaptor.shape();
  ArrayValueIndexCapture shapeCapture(shape, fGetDenseVal, fLoadVal);
  int64_t outputRank = shape.getType().cast<ShapedType>().getShape()[0];
  assert(outputRank != -1 && "Shape tensor must have constant shape");

  // Initialize context and results.
  outputDims.resize(outputRank);

  // Shape values can be 0, -1, or N (N > 0).
  //   - 0: the output dim is setting to the input dim at the same index.
  //   Thus, it must happen at the index < dataRank.
  //   - -1: the output dim is calculated from the other output dims. No more
  //   than one dim in the output has value -1.

  // Compute the total number of elements using the input data operand.
  IndexExpr numOfElements = LiteralIndexExpr(1);
  for (unsigned i = 0; i < dataRank; ++i)
    numOfElements = numOfElements * dataBounds.getDim(i);

  // Compute the total number of elements from the shape values.
  IndexExpr numOfElementsFromShape = LiteralIndexExpr(1);
  for (unsigned i = 0; i < outputRank; ++i) {
    SymbolIndexExpr dimShape(shapeCapture.getSymbol(i));
    if (dimShape.isUndefined())
      return op->emitError("shape input parameter could not be processed");
    IndexExpr dim;
    if (i < dataRank)
      // dimShape == 0: use dim from the input.
      dim = dimShape.selectOrSelf(dimShape == 0, dataBounds.getDim(i));
    else
      dim = dimShape;

    // Just store the dim as it is. Real value for -1 will be computed later.
    outputDims[i] = dim;

    // dimShape == -1: use 1 to compute the number of elements to avoid
    // negative value.
    dim = dim.selectOrSelf(dim == -1, LiteralIndexExpr(1));
    numOfElementsFromShape = numOfElementsFromShape * dim;
  }

  // All the output dims except the one with -1 are computed. Thus, only
  // update the dim with -1 here.
  for (unsigned i = 0; i < outputRank; ++i)
    outputDims[i] = outputDims[i].selectOrSelf(
        outputDims[i] == -1, numOfElements.floorDiv(numOfElementsFromShape));

  // Save the final result.
  dimsForOutput() = outputDims;

  return success();
}

} // namespace onnx_mlir
