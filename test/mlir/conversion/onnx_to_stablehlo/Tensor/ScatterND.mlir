// RUN: onnx-mlir-opt --canonicalize --convert-onnx-to-stablehlo %s -split-input-file | FileCheck %s

func.func @test_scatternd_1(%arg0 : tensor<8xf32>, %arg1 : tensor<4x1xi64>, %arg2 : tensor<4xf32>) -> tensor<8xf32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) : (tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
// CHECK-LABEL:  func.func @test_scatternd_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8xf32>, [[PARAM_1_:%.+]]: tensor<4x1xi64>, [[PARAM_2_:%.+]]: tensor<4xf32>) -> tensor<8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "stablehlo.scatter"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:           ^bb0([[arg3_:%.+]]: tensor<f32>, [[arg4_:%.+]]: tensor<f32>):
// CHECK:             stablehlo.return [[arg4_]] : tensor<f32>
// CHECK:           }) : (tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>) -> tensor<8xf32>
// CHECK:           return [[VAR_0_]] : tensor<8xf32>
// CHECK:         }
}

// -----

func.func @test_scatternd_2(%arg0 : tensor<4x4x4xi32>, %arg1 : tensor<2x1xi64>, %arg2 : tensor<2x4x4xi32>) -> tensor<4x4x4xi32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) : (tensor<4x4x4xi32>, tensor<2x1xi64>, tensor<2x4x4xi32>) -> tensor<4x4x4xi32>
  return %0 : tensor<4x4x4xi32>
// CHECK-LABEL:  func.func @test_scatternd_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x4x4xi32>, [[PARAM_1_:%.+]]: tensor<2x1xi64>, [[PARAM_2_:%.+]]: tensor<2x4x4xi32>) -> tensor<4x4x4xi32> {
// CHECK:           [[VAR_0_:%.+]] = "stablehlo.scatter"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:           ^bb0([[arg3_:%.+]]: tensor<i32>, [[arg4_:%.+]]: tensor<i32>):
// CHECK:             stablehlo.return [[arg4_]] : tensor<i32>
// CHECK:           }) : (tensor<4x4x4xi32>, tensor<2x1xi64>, tensor<2x4x4xi32>) -> tensor<4x4x4xi32>
// CHECK:           return [[VAR_0_]] : tensor<4x4x4xi32>
// CHECK:         }
}

// -----

func.func @test_scatternd_dynamic(%arg0 : tensor<1x?x32x128xf32>, %arg1 : tensor<?x?x32x64x4xi64>, %arg2 : tensor<?x?x?x?xf32>) -> tensor<1x?x32x128xf32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) : (tensor<1x?x32x128xf32>, tensor<?x?x32x64x4xi64>, tensor<?x?x?x?xf32>) -> tensor<1x?x32x128xf32>
  return %0 : tensor<1x?x32x128xf32>
// CHECK-LABEL:  func.func @test_scatternd_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x32x128xf32>, [[PARAM_1_:%.+]]: tensor<?x?x32x64x4xi64>, [[PARAM_2_:%.+]]: tensor<?x?x?x?xf32>) -> tensor<1x?x32x128xf32> {
// CHECK:           [[VAR_0_:%.+]] = "stablehlo.scatter"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 4>, unique_indices = false}> ({
// CHECK:           ^bb0([[arg3_:%.+]]: tensor<f32>, [[arg4_:%.+]]: tensor<f32>):
// CHECK:             stablehlo.return [[arg4_]] : tensor<f32>
// CHECK:           }) : (tensor<1x?x32x128xf32>, tensor<?x?x32x64x4xi64>, tensor<?x?x?x?xf32>) -> tensor<1x?x32x128xf32>
// CHECK:           return [[VAR_0_]] : tensor<1x?x32x128xf32>
// CHECK:         }
}
