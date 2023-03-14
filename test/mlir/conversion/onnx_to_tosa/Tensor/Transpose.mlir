// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_default_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
  "func.return"(%0) : (tensor<32x1x5x5xf32>) -> ()
// CHECK-LABEL:   func.func @test_default_transpose(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() {value = dense<[3, 2, 1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:           %[[VAL_2:.*]] = "tosa.transpose"(%[[VAL_0]], %[[VAL_1]]) : (tensor<5x5x1x32xf32>, tensor<4xi32>) -> tensor<32x1x5x5xf32>
// CHECK:           return %[[VAL_2]] : tensor<32x1x5x5xf32>
}

func.func @test_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<5x1x32x5xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<5x5x1x32xf32>) -> tensor<5x1x32x5xf32>
  "func.return"(%0) : (tensor<5x1x32x5xf32>) -> ()
// CHECK-LABEL:   func.func @test_transpose(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<5x5x1x32xf32>) -> tensor<5x1x32x5xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:           %[[VAL_2:.*]] = "tosa.transpose"(%[[VAL_0]], %[[VAL_1]]) : (tensor<5x5x1x32xf32>, tensor<4xi32>) -> tensor<5x1x32x5xf32>
// CHECK:           return %[[VAL_2]] : tensor<5x1x32x5xf32>
}