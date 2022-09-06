// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func private @test_relu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func private @test_relu(
// CHECK-SAME:  [[INPUT:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK:  [[OUTPUT:%.+]] = "tosa.clamp"([[INPUT]]) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK:  return [[OUTPUT]] : tensor<?x10xf32>
}
