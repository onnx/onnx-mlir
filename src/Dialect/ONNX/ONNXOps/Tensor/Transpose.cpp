/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Transpose.cpp - ONNX Operations -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Transpose operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

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
  setOutputDims(transposedDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Fold
//===----------------------------------------------------------------------===//

namespace {
// TODO: move to OpHelper, it's duplicated in ConstProp.cpp
template <typename T>
SmallVector<T, 4> createIntVectorFromArrayAttr(ArrayAttr a) {
  SmallVector<T, 4> vec;
  for (auto val : a.getValue())
    vec.push_back(val.cast<IntegerAttr>().getInt());
  return vec;
}
} // namespace

OpFoldResult ONNXTransposeOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "ONNXTransposeOp has 1 operand");
  if (!(operands.front())) // Null means operand is not a constant.
    return nullptr;        // No change when operand is not constant.
  assert(operands.front().isa<ElementsAttr>() &&
         "ONNXTransposeOp operand is tensor");
  ElementsAttr tensor = operands.front().cast<ElementsAttr>();
  ArrayAttr permAttr = (*this)->getAttr("perm").cast<ArrayAttr>();
  SmallVector<uint64_t, 4> perm =
      createIntVectorFromArrayAttr<uint64_t>(permAttr);
  ElementsAttrBuilder elementsBuilder(getContext());
  return elementsBuilder.transpose(
      elementsBuilder.fromElementsAttr(tensor), perm);
}

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXTransposeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>())
    return success();

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXTransposeOpShapeHelper, ONNXTransposeOp,
      ONNXTransposeOpAdaptor>(*this, elementType);
}
