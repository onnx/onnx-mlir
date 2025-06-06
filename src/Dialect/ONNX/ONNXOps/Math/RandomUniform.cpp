/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ RandomUniform.cpp - ONNX Operations//------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect RandomUniform operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXRandomUniformOpShapeHelper::computeShape() {
  ONNXRandomUniformOp randomOp = llvm::cast<ONNXRandomUniformOp>(op);

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
Type getRandomUniformElementType(ONNXRandomUniformOp op) {
  if (op.getDtypeAttr()) {
    const auto elementTypeID =
        static_cast<onnx::TensorProto_DataType>(op.getDtype());
    if (elementTypeID ==
        onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
      return Float16Type::get(op.getContext());
    } else if (elementTypeID ==
               onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
      return Float32Type::get(op.getContext());
    } else if (elementTypeID ==
               onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE) {
      return Float64Type::get(op.getContext());
    } else if (elementTypeID ==
               onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16) {
      return BFloat16Type::get(op.getContext());
    } else {
      llvm_unreachable("dtype not supported for RandomUniform");
    }
  }
  return Float32Type::get(op.getContext());
}
} // namespace

std::vector<Type> ONNXRandomUniformOp::resultTypeInference() {
  return {UnrankedTensorType::get(getRandomUniformElementType(*this))};
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomUniformOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ONNXRandomUniformOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      getRandomUniformElementType(*this));
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXRandomUniformOp>;
} // namespace onnx_mlir
