// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s -split-input-file | FileCheck %s

// Test when output shape is unkown
func.func @test_concat_dynamic_shape(%arg0 : tensor<5x5x?x32xf32>, %arg1 : tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) { axis = 2 : si64} : (tensor<5x5x?x32xf32>, tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32>
  "func.return"(%0) : (tensor<5x5x?x32xf32>) -> ()
// CHECK-LABEL:  func @test_concat_dynamic_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x?x32xf32>, [[PARAM_1_:%.+]]: tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32> {
// CHECK-NEXT:    [[VAR_0_:%.+]] = "mhlo.concatenate"(%arg0, %arg1) {dimension = 2 : i64} : (tensor<5x5x?x32xf32>, tensor<5x5x?x32xf32>) -> tensor<5x5x?x32xf32>
// CHECK-NEXT:   return [[VAR_0_]] : tensor<5x5x?x32xf32>
// CHECK-NEXT:   }
}

// -----

// Test when axis is negative
func.func @test_concat_negative_axis(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) { axis = -2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32>
  "func.return"(%0) : (tensor<5x5x4x32xf32>) -> ()
// CHECK-LABEL:  func @test_concat_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>, [[PARAM_1_:%.+]]: tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32> {
// CHECK-NEXT:    [[VAR_0_:%.+]] = "mhlo.concatenate"(%arg0, %arg1) {dimension = 2 : i64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>) -> tensor<5x5x4x32xf32>
// CHECK-NEXT:   return [[VAR_0_]] : tensor<5x5x4x32xf32>
// CHECK-NEXT:   }
}