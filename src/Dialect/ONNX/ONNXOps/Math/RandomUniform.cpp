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

LogicalResult ONNXRandomUniformLikeOp::verify() {
  ONNXRandomUniformLikeOpAdaptor operandAdaptor(*this);
  Value input = operandAdaptor.getInput();
  if (!hasShapeAndRank(input))
    return success();
  Value output = this->getOutput();
  if (!hasShapeAndRank(output))
    return success();

  auto inputType =
      mlir::cast<RankedTensorType>(input.getType()).getElementType();
  auto outputType =
      mlir::cast<RankedTensorType>(output.getType()).getElementType();

  auto elementTypeIDDType = operandAdaptor.getDtype();
  if (elementTypeIDDType) {
    const auto elementTypeID =
        static_cast<onnx::TensorProto_DataType>(*elementTypeIDDType);
    if (elementTypeID !=
            onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16 &&
        elementTypeID !=
            onnx::TensorProto_DataType::TensorProto_DataType_FLOAT &&
        elementTypeID !=
            onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE &&
        elementTypeID !=
            onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16) {
      return emitOpError("dtype not float16, float, double or bfloat16");
    }
    if (elementTypeID ==
            onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16 &&
        outputType != Float16Type::get(getContext()))
      return emitOpError("output tensor does not match float16 dtype.");
    else if (elementTypeID ==
                 onnx::TensorProto_DataType::TensorProto_DataType_FLOAT &&
             outputType != Float32Type::get(getContext()))
      return emitOpError("output tensor does not match float dtype.");
    else if (elementTypeID ==
                 onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE &&
             outputType != Float64Type::get(getContext()))
      return emitOpError("output tensor does not match double dtype.");
    else if (elementTypeID ==
                 onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16 &&
             outputType != BFloat16Type::get(getContext()))
      return emitOpError("output tensor does not match bfloat16 dtype.");
  } else if (inputType != outputType) {
    return emitOpError("output and input element types do not match.");
  }

  return success();
}

static Type getRandomUniformLikeOutputElementType(ONNXRandomUniformLikeOp op) {
  auto inputType = mlir::cast<TensorType>(op.getInput().getType());
  Type elementType = inputType.getElementType();
  if (op.getDtypeAttr()) {
    auto builder = OpBuilder(op.getContext());
    elementType = convertONNXTypeToMLIRType(
        builder, static_cast<onnx::TensorProto_DataType>(
                     op.getDtypeAttr().getValue().getSExtValue()));
  }
  return elementType;
}

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

std::vector<Type> ONNXRandomUniformLikeOp::resultTypeInference() {
  Type elementType = getRandomUniformLikeOutputElementType(*this);
  std::vector<Type> resultTypes;
  if (auto rankedInputType =
          mlir::dyn_cast<RankedTensorType>(getInput().getType())) {
    resultTypes.push_back(rankedInputType.clone(elementType));
  } else {
    resultTypes.push_back(UnrankedTensorType::get(elementType));
  }
  return resultTypes;
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

LogicalResult ONNXRandomUniformLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();
  Type elementType = getRandomUniformLikeOutputElementType(*this);
  return inferShapeForUnaryOps(getOperation(), elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXRandomUniformOp>;
} // namespace onnx_mlir
