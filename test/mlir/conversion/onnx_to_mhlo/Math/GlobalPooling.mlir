// RUN: onnx-mlir-opt --canonicalize --convert-onnx-to-mhlo --canonicalize %s -split-input-file | FileCheck %s

// COM: Test rewriting GlobalAveragePool into ReduceMean
func.func @test_global_average_pool(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
  %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  return %0 : tensor<1x3x1x1xf32>
  // CHECK-LABEL: test_global_average_pool
  // CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
  // CHECK: [[VAR_2_:%.+]] = mhlo.reduce([[PARAM_0_]] init: [[VAR_1_:%.+]]) applies mhlo.add across dimensions = [2, 3] : (tensor<1x3x5x5xf32>, tensor<f32>) -> tensor<1x3xf32>
  // CHECK: [[VAR_3_:%.+]] = "mhlo.reshape"([[VAR_2_]]) : (tensor<1x3xf32>) -> tensor<1x3x1x1xf32>
}

// -----

// COM: Test rewriting GlobalAveragePool into ReduceMean with dynamic dimensions
func.func @test_global_average_pool_dyn_dims(%arg0: tensor<1x?x?x5xf32>) -> tensor<1x?x1x1xf32> {
  %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x?x?x5xf32>) -> tensor<1x?x1x1xf32>
  return %0 : tensor<1x?x1x1xf32>
  // CHECK-LABEL: test_global_average_pool_dyn_dims
  // CHECK: [[VAR_3_:%.+]] = mhlo.reduce([[PARAM_0_:%.+]] init: [[VAR_2_:%.+]]) applies mhlo.add across dimensions = [2, 3] : (tensor<1x?x?x5xf32>, tensor<f32>) -> tensor<1x?xf32>
  // CHECK: [[VAR_4_:%.+]] = "mhlo.dynamic_reshape"([[VAR_3_]], [[VAR_0_:%.+]]) : (tensor<1x?xf32>, tensor<4xi64>) -> tensor<1x?x1x1xf32>
  // CHECK: [[VAR_5_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<1x?x?x5xf32> -> tensor<4xindex>
  // CHECK: [[VAR_6_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_1_:%.+]], [[VAR_5_]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<4xindex>) -> tensor<1x?x?x5xf32>
  // CHECK: [[VAR_7_:%.+]] = mhlo.reduce([[VAR_6_]] init: [[VAR_2_]]) applies mhlo.add across dimensions = [2, 3] : (tensor<1x?x?x5xf32>, tensor<f32>) -> tensor<1x?xf32>
  // CHECK: [[VAR_8_:%.+]] = "mhlo.dynamic_reshape"([[VAR_7_]], [[VAR_0_]]) : (tensor<1x?xf32>, tensor<4xi64>) -> tensor<1x?x1x1xf32>
}

// -----

// COM: Test rewriting GlobalMaxPool into ReduceMax
func.func @test_global_max_pool(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
  %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  return %0 : tensor<1x3x1x1xf32>
  // CHECK-LABEL: test_global_max_pool
  // CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
  // CHECK: [[VAR_1_:%.+]] = mhlo.reduce([[PARAM_0_]] init: [[VAR_0_:%.+]]) applies mhlo.maximum across dimensions = [2, 3] : (tensor<1x3x5x5xf32>, tensor<f32>) -> tensor<1x3xf32>
  // CHECK: [[VAR_2_:%.+]] = "mhlo.reshape"([[VAR_1_]]) : (tensor<1x3xf32>) -> tensor<1x3x1x1xf32>
}

// -----

// COM: Test rewriting GlobalMaxPool into ReduceMax with dynamic dimensions
func.func @test_global_max_pool_dyn_dims(%arg0: tensor<1x?x?x5xf32>) -> tensor<1x?x1x1xf32> {
  %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x?x?x5xf32>) -> tensor<1x?x1x1xf32>
  return %0 : tensor<1x?x1x1xf32>
  // CHECK-LABEL: test_global_max_pool_dyn_dims
  // CHECK:  [[VAR_2_:%.+]] = mhlo.reduce([[PARAM_0_:%.+]] init: [[VAR_1_:%.+]]) applies mhlo.maximum across dimensions = [2, 3] : (tensor<1x?x?x5xf32>, tensor<f32>) -> tensor<1x?xf32>
  // CHECK:  [[VAR_3_:%.+]] = "mhlo.dynamic_reshape"([[VAR_2_]], [[VAR_0_:%.+]]) : (tensor<1x?xf32>, tensor<4xi64>) -> tensor<1x?x1x1xf32>
}
