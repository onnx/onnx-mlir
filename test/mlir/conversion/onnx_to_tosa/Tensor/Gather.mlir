// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_gather_axis0(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()
// CHECK-LABEL:   func.func @test_gather_axis0(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<3x2xf32>) -> tensor<2x2x2xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() {value = dense<{{\[\[}}0, 1], [1, 2]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK:           %[[VAL_3:.*]] = "tosa.transpose"(%[[VAL_0]], %[[VAL_2]]) : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<3x2xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_3]]) {new_shape = [1, 3, 2]} : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.reshape"(%[[VAL_1]]) {new_shape = [1, 4]} : (tensor<2x2xi32>) -> tensor<1x4xi32>
// CHECK:           %[[VAL_6:.*]] = "tosa.gather"(%[[VAL_4]], %[[VAL_5]]) : (tensor<1x3x2xf32>, tensor<1x4xi32>) -> tensor<1x4x2xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.reshape"(%[[VAL_6]]) {new_shape = [2, 2, 2]} : (tensor<1x4x2xf32>) -> tensor<2x2x2xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK:           %[[VAL_9:.*]] = "tosa.transpose"(%[[VAL_7]], %[[VAL_8]]) : (tensor<2x2x2xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
// CHECK:           return %[[VAL_9]] : tensor<2x2x2xf32>
}

// -----

// Test negative indices.
func.func @test_gather_axis0_neg_idx(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()
// CHECK-LABEL:   func.func @test_gather_axis0_neg_idx(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<3x2xf32>) -> tensor<2x2x2xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() {value = dense<{{\[\[}}0, 2], [1, 2]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK:           %[[VAL_3:.*]] = "tosa.transpose"(%[[VAL_0]], %[[VAL_2]]) : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<3x2xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_3]]) {new_shape = [1, 3, 2]} : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.reshape"(%[[VAL_1]]) {new_shape = [1, 4]} : (tensor<2x2xi32>) -> tensor<1x4xi32>
// CHECK:           %[[VAL_6:.*]] = "tosa.gather"(%[[VAL_4]], %[[VAL_5]]) : (tensor<1x3x2xf32>, tensor<1x4xi32>) -> tensor<1x4x2xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.reshape"(%[[VAL_6]]) {new_shape = [2, 2, 2]} : (tensor<1x4x2xf32>) -> tensor<2x2x2xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK:           %[[VAL_9:.*]] = "tosa.transpose"(%[[VAL_7]], %[[VAL_8]]) : (tensor<2x2x2xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
// CHECK:           return %[[VAL_9]] : tensor<2x2x2xf32>
}

// -----

// Test along axis 1. Transpose should be different.
func.func @test_gather_axis1(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 2]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()
// CHECK-LABEL:   func.func @test_gather_axis1(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<3x3xf32>) -> tensor<3x1x2xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() {value = dense<{{\[\[}}0, 2]]> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK:           %[[VAL_3:.*]] = "tosa.transpose"(%[[VAL_0]], %[[VAL_2]]) : (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_3]]) {new_shape = [1, 3, 3]} : (tensor<3x3xf32>) -> tensor<1x3x3xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.reshape"(%[[VAL_1]]) {new_shape = [1, 2]} : (tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_6:.*]] = "tosa.gather"(%[[VAL_4]], %[[VAL_5]]) : (tensor<1x3x3xf32>, tensor<1x2xi32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.reshape"(%[[VAL_6]]) {new_shape = [1, 2, 3]} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK:           %[[VAL_9:.*]] = "tosa.transpose"(%[[VAL_7]], %[[VAL_8]]) : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x1x2xf32>
// CHECK:           return %[[VAL_9]] : tensor<3x1x2xf32>
}