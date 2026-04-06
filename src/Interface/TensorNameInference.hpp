// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "src/Dialect/ONNX/TensorName.hpp"
#include "src/Interface/TensorNameInference.hpp.inc"

namespace onnx_mlir {

class ReshapeOpTensorNameInference
    : public mlir::TensorNameInference::ExternalModel<
          ReshapeOpTensorNameInference, mlir::ONNXReshapeOp> {
public:
  std::unique_ptr<onnx_mlir::Transform> inferTensorNameTransform(
      mlir::Operation *op) const;
};

class TransposeOpTensorNameInference
    : public mlir::TensorNameInference::ExternalModel<
          TransposeOpTensorNameInference, mlir::ONNXTransposeOp> {
public:
  std::unique_ptr<onnx_mlir::Transform> inferTensorNameTransform(
      mlir::Operation *op) const;
};

class PadOpTensorNameInference
    : public mlir::TensorNameInference::ExternalModel<PadOpTensorNameInference,
          mlir::ONNXPadOp> {
public:
  std::unique_ptr<onnx_mlir::Transform> inferTensorNameTransform(
      mlir::Operation *op) const;
};

class SliceOpTensorNameInference
    : public mlir::TensorNameInference::ExternalModel<
          SliceOpTensorNameInference, mlir::ONNXSliceOp> {
public:
  std::unique_ptr<onnx_mlir::Transform> inferTensorNameTransform(
      mlir::Operation *op) const;
};

} // namespace onnx_mlir
