/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ RandomNormal.cpp - ONNX Operations ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect RandomNormal operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXRandomNormalOpShapeHelper::computeShape() {
  ONNXRandomNormalOp randomOp = llvm::cast<ONNXRandomNormalOp>(op);

  DimsExpr outputDims;
  createIE->getIntFromArrayAsLiterals(randomOp.getShape(), outputDims);
  if (!IndexExpr::isNonNegativeLiteral(outputDims))
    return op->emitError("Random normal tensor has dynamic dimension.");
  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

namespace {
Type getRandomNormalElementType(ONNXRandomNormalOp op) {
  if (op.getDtypeAttr()) {
    const auto elementTypeID =
        static_cast<onnx::TensorProto_DataType>(op.getDtype());
    if (elementTypeID ==
        onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
      return FloatType::getF16(op.getContext());
    } else if (elementTypeID ==
               onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
      return FloatType::getF32(op.getContext());
    } else if (elementTypeID ==
               onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE) {
      return FloatType::getF64(op.getContext());
    } else if (elementTypeID ==
               onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16) {
      return FloatType::getBF16(op.getContext());
    } else {
      llvm_unreachable("dtype not supported for RandomNormal");
    }
  }
  return FloatType::getF32(op.getContext());
}
} // namespace

std::vector<Type> ONNXRandomNormalOp::resultTypeInference() {
  return {UnrankedTensorType::get(getRandomNormalElementType(*this))};
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomNormalOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ONNXRandomNormalOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      getRandomNormalElementType(*this));
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXRandomNormalOp>;
} // namespace onnx_mlir
