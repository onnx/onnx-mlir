// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --remove-redundant-relu %s | FileCheck %s

func.func @test_relu_chain(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %1 = "onnx.Relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %2 = "onnx.Relu"(%1) : (tensor<1xf32>) -> tensor<1xf32>
  %3 = "onnx.Relu"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "onnx.Relu"(%3) : (tensor<1xf32>) -> tensor<1xf32>

  return %4 : tensor<1xf32>
}

// CHECK-LABEL: func.func @test_relu_chain
// CHECK: %[[R:.*]] = "onnx.Relu"(%arg0)
// CHECK-NOT: "onnx.Relu"(%[[R]])
// CHECK: return %[[R]]
