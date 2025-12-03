/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Reshape.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Reshape operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXReshapeOpShapeHelper::computeShape() {
  ONNXReshapeOpAdaptor operandAdaptor(operands);
  DimsExpr outputDims;

  // Get info about input data operand.
  Value data = operandAdaptor.getData();
  int64_t dataRank = mlir::cast<ShapedType>(data.getType()).getShape().size();

  // Get info about shape operand.
  Value shape = operandAdaptor.getShape();
  int64_t outputRank = createIE->getShape(shape, 0);
  assert(outputRank != ShapedType::kDynamic &&
         "Shape tensor must have constant shape");

  // Initialize context and results.
  outputDims.resize(outputRank);

  // Shape values can be 0, -1, or N (N > 0).
  //   - 0: the output dim is setting to the input dim at the same index.
  //   Thus, it must happen at the index < dataRank.
  //   - -1: the output dim is calculated from the other output dims. No more
  //   than one dim in the output has value -1.

  // Shape inference can be simplified if there is a bijection between a set of
  // unknown dimensions in data and unknown dimensions in shape. In such a case,
  // there is no need to include these unknown dimensions in computing the
  // dimension at position of -1, which increases the chance that the dim value
  // at position of -1 can be a static value.
  //
  // For example,
  //  - data is tensor<1x?x2048xf32>,
  //  - shape is tensor<4xi64> of [1, dim_1_of_data, -1, 64]
  // In this case, the 2nd dimension of data is unknown but it is similar to the
  // 2nd value in shape. So to compute the output dim at position of -1, we just
  // do 2048/64, that is 32. Without this simplification, the output dim at
  // position of -1 would be unknown at compile time.
  std::set<int64_t> dataIgnoredDims, outputIgnoredDims;
  SmallVector<Value> shapeDimVals;
  if (areDimsFromConcat(shape)) {
    getDims(shape, shapeDimVals);
    Value refData = data;

    // Get the input A of MatMul that is the producer of "data" if applicable.
    // Special case to handle a pattern in the IBM granite-3.1-2b-instruct
    // model. This pattern is found in the IBM granite-3.1-2b-instruct model.
    // clang-format off
    // %0 = onnx.Constant dense<1.000000e+00> : tensor<2048x2048xf32>
    // %1 = onnx.Constant dense<64> : tensor<1xi64>
    // %2 = onnx.Constant dense<-1> : tensor<1xi64>
    // %3 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x?x2048xf32>) -> tensor<1xi64>
    // %4 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<?x?x2048xf32>) -> tensor<1xi64>
    // %5 = "onnx.MatMul"(%arg0, %0) : (tensor<?x?x2048xf32>, tensor<2048x2048xf32>) -> tensor<?x?x2048xf32>
    // %6 = "onnx.Concat"(%3, %4, %2, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    // %7 = "onnx.Reshape"(%5, %6) {allowzero = 0 : si64} : (tensor<?x?x2048xf32>, tensor<4xi64>) -> tensor<?x?x?x64xf32>
    // clang-format on
    // This is a special handling which is not encouraged to be used widely.
    // Since there is no good mechanism to handle this situation in a systematic
    // way (e.g. using dynamic dimension analysis), so we handle it here.
    ONNXMatMulOp mmOp = data.getDefiningOp<ONNXMatMulOp>();
    bool fromMatMul = false;
    if (mmOp && isRankedShapedType(mmOp.getB().getType()) &&
        getRank(mmOp.getB().getType()) == 2) {
      refData = mmOp.getA();
      fromMatMul = true;
    }

    // Find the bijective mapping.
    // We do not compute the actual mapping, just storing the source and target
    // sets is enough if the map exists.
    bool isBijective = true;
    for (int64_t i = 0; i < outputRank; ++i) {
      Value dim = shapeDimVals[i];
      if (auto dimOp = dim.getDefiningOp<ONNXDimOp>()) {
        if (dimOp.getData() != refData)
          continue;
        int64_t axis = dimOp.getAxis();
        if (auto search = dataIgnoredDims.find(axis);
            search != dataIgnoredDims.end())
          isBijective = false;
        if (fromMatMul && axis == getRank(refData.getType()) - 1)
          isBijective = false;
        outputIgnoredDims.insert(i);
        dataIgnoredDims.insert(axis);
      }
    }
    if (!isBijective) {
      outputIgnoredDims.clear();
      dataIgnoredDims.clear();
    }
  }

  // Compute the total number of elements using the input data operand.
  // dataRank will be 0 if Data is unranked tensor.
  // The number of element will not be computed
  IndexExpr numOfElements = LitIE(1);
  for (unsigned i = 0; i < dataRank; ++i) {
    if (auto search = dataIgnoredDims.find(i); search != dataIgnoredDims.end())
      continue;
    numOfElements = numOfElements * createIE->getShapeAsDim(data, i);
  }

  // Compute the total number of elements from the shape values.
  IndexExpr numOfElementsFromShape = LitIE(1);
  for (unsigned i = 0; i < outputRank; ++i) {
    IndexExpr dimShape = createIE->getIntFromArrayAsSymbol(shape, i);
    if (dimShape.isUndefined())
      return op->emitError("shape input parameter could not be processed");
    IndexExpr dim;
    if (i < dataRank)
      // dimShape == 0: use dim from the input.
      dim = dimShape.selectOrSelf(
          dimShape == 0, createIE->getShapeAsDim(data, i));
    else
      dim = dimShape;

    // Just store the dim as it is. Real value for -1 will be computed later.
    outputDims[i] = dim;

    // dimShape == -1: use 1 to compute the number of elements to avoid
    // negative value.
    if (auto search = outputIgnoredDims.find(i);
        search != outputIgnoredDims.end())
      continue;
    dim = dim.selectOrSelf(dim == -1, LitIE(1));
    numOfElementsFromShape = numOfElementsFromShape * dim;
  }

  // When data is ranked tensor, all the output dims except the one with -1
  // are computed. Thus, only update the dim with -1 here.
  // When data is unranked tensor, output dims with -1 or 0 (allowzero == 0)
  // should be -1 (represented as QuestionmarkIndexExpr)
  for (unsigned i = 0; i < outputRank; ++i) {
    if (hasShapeAndRank(data)) {
      IndexExpr dimShape = createIE->getIntFromArrayAsSymbol(shape, i);
      outputDims[i] = outputDims[i].selectOrSelf(
          dimShape == -1, numOfElements.floorDiv(numOfElementsFromShape));
    } else {
      // ToFix: can not check getAllowzero because the operandAdaptor is
      // constructed without attributes
      // Anyway the question mark is a conservative but correct result.
      outputDims[i] = outputDims[i].selectOrSelf(
          outputDims[i] == 0, QuestionmarkIndexExpr(false));
      outputDims[i] = outputDims[i].selectOrSelf(
          outputDims[i] == -1, QuestionmarkIndexExpr(false));
    }
  }

  // Save the final result.
  setOutputDims(outputDims);

  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXReshapeOp::verify() {
  // Cannot verify if shape has unknown rank.
  if (!hasShapeAndRank(getShape()))
    return success();

  // Only rank 1 shape tensors are supported.
  auto shapeTy = cast<ShapedType>(getShape().getType());
  if (shapeTy.getRank() != 1)
    return emitOpError("Shape tensor must have rank one");

  // TODO: Check that any -1 dim is used correctly.
  // TODO: Check that any 0 dim is used correctly with allowzero.
  // TODO: Check that data can reshape to shape if data's shape is known.

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXReshapeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape without data rank and static shape of shape.
  // If shape is constant shape, the rank of the output known.
  // This step may be helpful to reach the fix point.
  // TODO: Infer shape without data rank if shape is a constant
  //       without -1 and without 0 and allowzero.
  if (!hasShapeAndRank(getData()) && !hasStaticShape(getShape().getType()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getData().getType()).getElementType();
  ONNXReshapeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXReshapeOp>;
} // namespace onnx_mlir
