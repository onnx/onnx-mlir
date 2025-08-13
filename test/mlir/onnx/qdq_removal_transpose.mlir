// RUN: onnx-mlir-opt --canonicalize --qdq-around-op-opt-onnx-to-onnx %s -split-input-file | FileCheck %s

  func.func @test_transpose_pattern(%arg0: tensor<*xui16>) -> tensor<*xui16> {
    %0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
    %1 = onnx.Constant dense<39664> : tensor<ui16>
    %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<*xui16>, tensor<f32>, tensor<ui16>) -> tensor<*xf32>
    %3 = "onnx.Transpose"(%2) {saturate = 1 : si64, to = f32} : (tensor<*xf32>) -> tensor<*xf32>
    %4 = "onnx.QuantizeLinear"(%3, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<*xf32>, tensor<f32>, tensor<ui16>) -> tensor<*xui16>
    return %4 : tensor<*xui16>
  }

  // CHECK-LABEL: func.func @test_transpose_pattern(%arg0: tensor<*xui16>) -> tensor<*xui16>
  // CHECK-NOT: onnx.DequantizeLinear
  // CHECK: onnx.Transpose
  // CHECK-NOT: onnx.QuantizeLinear
