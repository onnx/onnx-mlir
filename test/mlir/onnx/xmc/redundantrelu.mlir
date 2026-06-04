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

// -----

// Inner Relu fans out: %0 feeds both the outer Relu (%1) and an independent
// consumer (Sigmoid). Because the inner Relu has more than one use, the pass
// skips it and leaves the chain untouched.
func.func @test_relu_inner_fanout(%arg0: tensor<1xf32>)
    -> (tensor<1xf32>, tensor<1xf32>) {
  %0 = "onnx.Relu"(%arg0) {ResultNames = ["inner"]} : (tensor<1xf32>) -> tensor<1xf32>
  %1 = "onnx.Relu"(%0) {ResultNames = ["outer"]} : (tensor<1xf32>) -> tensor<1xf32>
  %2 = "onnx.Sigmoid"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  return %1, %2 : tensor<1xf32>, tensor<1xf32>
}

// CHECK-LABEL: func.func @test_relu_inner_fanout
// CHECK: %[[INNER:.*]] = "onnx.Relu"(%arg0) {{.*}}ResultNames = ["inner"]
// CHECK: %[[OUTER:.*]] = "onnx.Relu"(%[[INNER]]) {{.*}}ResultNames = ["outer"]
// CHECK: %[[SIG:.*]] = "onnx.Sigmoid"(%[[INNER]])
// CHECK: return %[[OUTER]], %[[SIG]]
