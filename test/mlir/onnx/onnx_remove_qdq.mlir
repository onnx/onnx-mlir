// RUN: onnx-mlir-opt --qdq-opt-onnx-to-onnx %s -split-input-file | FileCheck %s

 func.func @test_qdq_pattern1(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
    %0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
    %1 = onnx.Constant dense<39664> : tensor<ui16>
    %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
    %3 = "onnx.QuantizeLinear"(%2, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
    return %3 : tensor<1x128x768xui16>

  }

  // CHECK-LABEL: func.func @test_qdq_pattern1(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16>
  // CHECK: return %arg0 : tensor<1x128x768xui16>
  // CHECK-NOT: onnx.DequantizeLinear
  // CHECK-NOT: onnx.QuantizeLinear

func.func @test_qdq_pattern2(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
    %0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
    %1 = onnx.Constant dense<39664> : tensor<ui16>
    %2 = onnx.Constant dense<6.57987776E-5> : tensor<f32>
    %3 = onnx.Constant dense<45664> : tensor<ui16>
    %4 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
    %5 = "onnx.QuantizeLinear"(%4, %2, %3) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
    return %5 : tensor<1x128x768xui16>
  }

  // CHECK-LABEL: func.func @test_qdq_pattern2(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16>
  // CHECK: onnx.DequantizeLinear
  // CHECK: onnx.QuantizeLinear