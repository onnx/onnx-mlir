/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ DepthToSpace.cpp - ONNX Operations ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect DepthToSpace operation.
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
LogicalResult ONNXDepthToSpaceOpShapeHelper::computeShape() {
  // Get info about input data operand and blocksize.
  ONNXDepthToSpaceOp depthOp = llvm::cast<ONNXDepthToSpaceOp>(op);
  ONNXDepthToSpaceOpAdaptor operandAdaptor(operands);
  Value input = operandAdaptor.getInput();
  if (!hasShapeAndRank(input)) {
    return failure();
  }
  int64_t inputRank = createIE->getShapedTypeRank(input);
  assert(inputRank == 4 && "Unexpected input tensor rank");
  int64_t blocksize = depthOp.getBlocksize();
  assert(blocksize > 0 && "blocksize should be strictly positive");

  // Compute outputDims.
  // The input tensor has format [N,C,H,W], where N is the batch axis, C is the
  // channel or depth, H is the height and W is the width. The output tensor has
  // shape [N, C / (blocksize * blocksize), H * blocksize, W * blocksize].
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  DimIndexExpr N(createIE->getShapeAsDim(input, 0));
  DimIndexExpr C(createIE->getShapeAsDim(input, 1));
  DimIndexExpr H(createIE->getShapeAsDim(input, 2));
  DimIndexExpr W(createIE->getShapeAsDim(input, 3));

  outputDims[0] = N;
  outputDims[1] = C.floorDiv(blocksize * blocksize);
  outputDims[2] = H * blocksize;
  outputDims[3] = W * blocksize;

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXDepthToSpaceOp::verify() {
  ONNXDepthToSpaceOpAdaptor operandAdaptor(*this);

  // Check input.
  Value input = operandAdaptor.getInput();
  if (!hasShapeAndRank(input)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  auto inputType = mlir::cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();
  if (inputShape.size() != 4)
    return emitOpError("Input should have a rank of four");

  // Check blocksize.
  int64_t blocksize = operandAdaptor.getBlocksize();
  if (blocksize < 0)
    return emitOpError("Blocksize should be non negative");

  int64_t C = inputShape[1];
  if (!ShapedType::isDynamic(C) && C % (blocksize * blocksize) != 0)
    return emitOpError("The input tensor depth must be divisible by the "
                       "(blocksize * blocksize)");

  // Check mode.
  StringRef mode = operandAdaptor.getMode();
  if (mode != "DCR" && mode != "CRD")
    return emitOpError("Mode must be DCR or CRD");

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXDepthToSpaceOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no input shape exists.
  if (!hasShapeAndRank(getInput()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getInput().getType()).getElementType();
  ONNXDepthToSpaceOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXDepthToSpaceOp>;
} // namespace onnx_mlir
