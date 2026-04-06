// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "src/Interface/TensorNameInference.hpp"

namespace mlir {

#include "src/Interface/TensorNameInference.cpp.inc"

}

#include "src/Dialect/ONNX/TensorName.hpp"

using namespace mlir;

namespace onnx_mlir {

std::unique_ptr<Transform>
ReshapeOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  return {};
}

std::unique_ptr<Transform>
TransposeOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  return {};
}

std::unique_ptr<Transform> SliceOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  return {};
}

std::unique_ptr<Transform> PadOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  return {};
}

} // namespace onnx_mlir
