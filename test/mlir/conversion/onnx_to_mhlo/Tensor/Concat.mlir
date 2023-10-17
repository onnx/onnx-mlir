// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo %s -split-input-file | FileCheck %s

// Test when output shape is unknown
func.func @test_concat_dynamic_shape(%arg0 : tensor<5x5x?x32xf32>, %arg1 : tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) { axis = 2 : si64} : (tensor<5x5x?x32xf32>, tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32>
  "func.return"(%0) : (tensor<5x5x?x32xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_concat_dynamic_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x?x32xf32>, [[PARAM_1_:%.+]]: tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.concatenate [[PARAM_0_]], [[PARAM_1_]], dim = 2 : (tensor<5x5x?x32xf32>, tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32>
// CHECK:           return [[VAR_0_]] : tensor<5x5x?x32xf32>
// CHECK:         }

// -----

// Test when axis is negative
func.func @test_concat_negative_axis(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) { axis = -2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32>
  "func.return"(%0) : (tensor<5x5x4x32xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_concat_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>, [[PARAM_1_:%.+]]: tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.concatenate [[PARAM_0_]], [[PARAM_1_]], dim = 2 : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32>
// CHECK:           return [[VAR_0_]] : tensor<5x5x4x32xf32>
// CHECK:         }
