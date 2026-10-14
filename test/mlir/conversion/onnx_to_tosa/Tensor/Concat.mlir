// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_concat(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x4xf32>) -> tensor<2x7xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  "func.return"(%0) : (tensor<2x7xf32>) -> ()
// CHECK-LABEL:  func @test_concat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3xf32>, [[PARAM_1_:%.+]]: tensor<2x4xf32>) -> tensor<2x7xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.concat [[PARAM_0_]], [[PARAM_1_]] {axis = 1 : i32} : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
// CHECK:           return [[VAR_0_]] : tensor<2x7xf32>
}

// -----

func.func @test_concat_negative_axis(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x4xf32>) -> tensor<2x7xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  "func.return"(%0) : (tensor<2x7xf32>) -> ()
// CHECK-LABEL:  func @test_concat_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3xf32>, [[PARAM_1_:%.+]]: tensor<2x4xf32>) -> tensor<2x7xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.concat [[PARAM_0_]], [[PARAM_1_]] {axis = 1 : i32} : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
// CHECK:           return [[VAR_0_]] : tensor<2x7xf32>
}

// -----

func.func @test_concat_three_inputs(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2xf32>, %arg2 : tensor<3x2xf32>) -> tensor<6x2xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 0 : si64} : (tensor<1x2xf32>, tensor<2x2xf32>, tensor<3x2xf32>) -> tensor<6x2xf32>
  "func.return"(%0) : (tensor<6x2xf32>) -> ()
// CHECK-LABEL:  func @test_concat_three_inputs
// CHECK:           [[VAR_0_:%.+]] = tosa.concat [[PARAM_0_:%.+]], [[PARAM_1_:%.+]], [[PARAM_2_:%.+]] {axis = 0 : i32} : (tensor<1x2xf32>, tensor<2x2xf32>, tensor<3x2xf32>) -> tensor<6x2xf32>
// CHECK:           return [[VAR_0_]] : tensor<6x2xf32>
}
