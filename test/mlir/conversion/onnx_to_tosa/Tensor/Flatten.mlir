// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_static_flatten(%arg0: tensor<32x512x1x1xf32>) -> tensor<32x512xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<32x512x1x1xf32>) -> tensor<32x512xf32>
  return %0 : tensor<32x512xf32>
  //CHECK: {{%.+}} =  "tosa.reshape"(%arg0) {new_shape = array<i64: 32, 512>} : (tensor<32x512x1x1xf32>) -> tensor<32x512xf32>
}

// -----
func.func @test_static_flatten_mult(%arg0: tensor<32x51x5x3xf32>) -> tensor<1632x15xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<32x51x5x3xf32>) -> tensor<1632x15xf32>
  return %0 : tensor<1632x15xf32>
  //CHECK:  {{%.+}} = "tosa.reshape"(%arg0) {new_shape = array<i64: 1632, 15>} : (tensor<32x51x5x3xf32>) -> tensor<1632x15xf32>
}

// -----
func.func @test_flatten_axes_0(%arg0: tensor<32x51x1x1xf32>) -> tensor<1x1632xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 0 : si64} : (tensor<32x51x1x1xf32>) -> tensor<1x1632xf32>
  return %0 : tensor<1x1632xf32>
  //CHECK:  {{%.+}} = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 1632>} : (tensor<32x51x1x1xf32>) -> tensor<1x1632xf32>
}

// -----
func.func @test_flatten_axes_last(%arg0: tensor<32x51x1x3xf32>) -> tensor<1632x3xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 3 : si64} : (tensor<32x51x1x3xf32>) -> tensor<1632x3xf32>
  return %0 : tensor<1632x3xf32>
  //CHECK:  {{%.+}} = "tosa.reshape"(%arg0) {new_shape = array<i64: 1632, 3>} : (tensor<32x51x1x3xf32>) -> tensor<1632x3xf32>
}