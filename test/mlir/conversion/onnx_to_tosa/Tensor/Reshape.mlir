// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_reshape(%arg0 : tensor<128x1024xf32>) -> tensor<1x128x16x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 128, 16, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<128x1024xf32>, tensor<4xi64>) -> tensor<1x128x16x64xf32>
  "func.return"(%1) : (tensor<1x128x16x64xf32>) -> ()
// CHECK-LABEL:  func @test_reshape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x1024xf32>) -> tensor<1x128x16x64xf32> {
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 128, 16, 64>} : (tensor<128x1024xf32>) -> tensor<1x128x16x64xf32>
// CHECK-NEXT:      return [[VAR_1_]] : tensor<1x128x16x64xf32>
}

func.func @test_reshape_allowzero(%arg0 : tensor<12x128x1024xf32>) -> tensor<12x128x16x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 0, 16, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<12x128x1024xf32>, tensor<4xi64>) -> tensor<12x128x16x64xf32>
  "func.return"(%1) : (tensor<12x128x16x64xf32>) -> ()
// CHECK-LABEL:  func @test_reshape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<12x128x1024xf32>) -> tensor<12x128x16x64xf32> {
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 12, 128, 16, 64>} : (tensor<12x128x1024xf32>) -> tensor<12x128x16x64xf32>
// CHECK-NEXT:      return [[VAR_1_]] : tensor<12x128x16x64xf32>
}
