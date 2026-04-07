// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

include "mlir/IR/OpBase.td"

def TensorNameInference : OpInterface<"TensorNameInference"> {
  let methods = [
    InterfaceMethod<"Return Transform representing the Op",
                    "std::unique_ptr<onnx_mlir::Transform>",
                    "inferTensorNameTransform">
    ];
  let cppNamespace = "::mlir";
}
