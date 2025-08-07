// RUN: onnx-mlir-opt --shape-inference --qdq-opt-onnx-to-onnx -split-input-file | FileCheck %s
 
  func.func @test_slice_pattern1(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
    %0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
    %1 = onnx.Constant dense<39664> : tensor<ui16>
    %2 = onnx.Constant dense<0> : tensor<3xi64>
    %3 = onnx.Constant dense<[1, 128, 768]> : tensor<3xi64>
    %4 = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
    %5 = onnx.Constant dense<1> : tensor<3xi64>
    %6 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
    %7 = "onnx.Slice"(%6, %2, %3, %4, %5) : (tensor<1x128x768xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x128x768xf32>
    %8 = "onnx.QuantizeLinear"(%7, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
    return %8 : tensor<1x128x768xui16>
  }
 
  // CHECK-LABEL: func.func @test_slice_pattern1(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16>
  // CHECK-NOT: onnx.DequantizeLinear
  // CHECK-NOT: onnx.Slice
  // CHECK-NOT: onnx.QuantizeLinear
  // CHECK: return %arg0 : tensor<1x128x768xui16>

  func.func @test_slice_pattern2(%arg0: tensor<1x128x768xui16>) -> tensor<1x64x384xui16> {
    %0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
    %1 = onnx.Constant dense<39664> : tensor<ui16>
    %2 = onnx.Constant dense<0> : tensor<3xi64>
    %3 = onnx.Constant dense<[1, 128, 768]> : tensor<3xi64>
    %4 = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
    %5 = onnx.Constant dense<2> : tensor<3xi64>
    %6 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
    %7 = "onnx.Slice"(%6, %2, %3, %4, %5) : (tensor<1x128x768xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x64x384xf32>
    %8 = "onnx.QuantizeLinear"(%7, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x64x384xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x64x384xui16>
    return %8 : tensor<1x64x384xui16>
  }
 
  // CHECK-LABEL: func.func @test_slice_pattern2(%arg0: tensor<1x128x768xui16>) -> tensor<1x64x384xui16>
  // CHECK: onnx.DequantizeLinear
  // CHECK: onnx.Slice
  // CHECK: onnx.QuantizeLinear