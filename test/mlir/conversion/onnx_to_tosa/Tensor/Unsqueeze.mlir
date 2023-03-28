// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_unsqueeze(%arg0 : tensor<10x10xf32>) -> tensor<1x10x10x1xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<1x10x10x1xf32>
  func.return %1 : tensor<1x10x10x1xf32>
// CHECK-LABEL:   func.func @test_unsqueeze(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<10x10xf32>) -> tensor<1x10x10x1xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.reshape"(%[[VAL_0]]) {new_shape = array<i64: 1, 10, 10, 1>} : (tensor<10x10xf32>) -> tensor<1x10x10x1xf32>
// CHECK:           return %[[VAL_1]] : tensor<1x10x10x1xf32>
// CHECK:         }
}

func.func @test_unsqueeze_negative_axis(%arg0 : tensor<16x32x64xf32>) -> tensor<16x32x1x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[-2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<1xi64>) -> tensor<16x32x1x64xf32>
  func.return %1 : tensor<16x32x1x64xf32>
// CHECK-LABEL:   func.func @test_unsqueeze_negative_axis(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<16x32x64xf32>) -> tensor<16x32x1x64xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.reshape"(%[[VAL_0]]) {new_shape = array<i64: 16, 32, 1, 64>} : (tensor<16x32x64xf32>) -> tensor<16x32x1x64xf32>
// CHECK:           return %[[VAL_1]] : tensor<16x32x1x64xf32>
// CHECK:         }
}

func.func @test_unsqueeze_mix(%arg0 : tensor<16x32x64xf32>) -> tensor<16x1x32x1x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<2xi64>) -> tensor<16x1x32x1x64xf32>
  func.return %1 : tensor<16x1x32x1x64xf32>
// CHECK-LABEL:   func.func @test_unsqueeze_mix(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<16x32x64xf32>) -> tensor<16x1x32x1x64xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.reshape"(%[[VAL_0]]) {new_shape = array<i64: 16, 1, 32, 1, 64>} : (tensor<16x32x64xf32>) -> tensor<16x1x32x1x64xf32>
// CHECK:           return %[[VAL_1]] : tensor<16x1x32x1x64xf32>
// CHECK:         }
}