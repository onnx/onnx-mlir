// RUN: onnx-mlir-opt --canonicalize --convert-onnx-to-stablehlo %s -split-input-file | FileCheck %s

func.func @test_scatternd_1(%arg0 : tensor<8xf32>, %arg1 : tensor<4x1xi64>, %arg2 : tensor<4xf32>) -> tensor<8xf32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) : (tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
// CHECK-LABEL:  func.func @test_scatternd_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8xf32>, [[PARAM_1_:%.+]]: tensor<4x1xi64>, [[PARAM_2_:%.+]]: tensor<4xf32>) -> tensor<8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "stablehlo.scatter"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) ({
// CHECK:           ^bb0([[arg3_:%.+]]: tensor<f32>, [[arg4_:%.+]]: tensor<f32>):
// CHECK:             stablehlo.return [[arg4_]] : tensor<f32>
// CHECK:           }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>) -> tensor<8xf32>
// CHECK:           return [[VAR_0_]] : tensor<8xf32>
// CHECK:         }
}

// -----

func.func @test_scatternd_2(%arg0 : tensor<4x4x4xi32>, %arg1 : tensor<2x1xi64>, %arg2 : tensor<2x4x4xi32>) -> tensor<4x4x4xi32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) : (tensor<4x4x4xi32>, tensor<2x1xi64>, tensor<2x4x4xi32>) -> tensor<4x4x4xi32>
  return %0 : tensor<4x4x4xi32>
// CHECK-LABEL:  func.func @test_scatternd_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x4x4xi32>, [[PARAM_1_:%.+]]: tensor<2x1xi64>, [[PARAM_2_:%.+]]: tensor<2x4x4xi32>) -> tensor<4x4x4xi32> {
// CHECK:           [[VAR_0_:%.+]] = "stablehlo.scatter"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) ({
// CHECK:           ^bb0([[arg3_:%.+]]: tensor<i32>, [[arg4_:%.+]]: tensor<i32>):
// CHECK:             stablehlo.return [[arg4_]] : tensor<i32>
// CHECK:           }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<4x4x4xi32>, tensor<2x1xi64>, tensor<2x4x4xi32>) -> tensor<4x4x4xi32>
// CHECK:           return [[VAR_0_]] : tensor<4x4x4xi32>
// CHECK:         }
}
