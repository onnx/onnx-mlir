// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s

// Test normal case
func.func @test_flatten(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_flatten
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<25x32xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [25, 32] : tensor<2xindex>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x1x32xf32>, tensor<2xindex>) -> tensor<25x32xf32>
// CHECK:           return [[VAR_1_]] : tensor<25x32xf32>
// CHECK:         }

// -----

// Test when axis is negative
func.func @test_flatten_negative_axis(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = -2 : si64} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_flatten_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<25x32xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [25, 32] : tensor<2xindex>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x1x32xf32>, tensor<2xindex>) -> tensor<25x32xf32>
// CHECK:           return [[VAR_1_]] : tensor<25x32xf32>
// CHECK:         }

// -----

// Test when axis is not set
func.func @test_flatten_with_default_axis(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Flatten"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_flatten_with_default_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<5x160xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [5, 160] : tensor<2xindex>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x1x32xf32>, tensor<2xindex>) -> tensor<5x160xf32>
// CHECK:           return [[VAR_1_]] : tensor<5x160xf32>
// CHECK:         }

// -----

func.func @test_flatten1(%arg0 : tensor<2x?x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x?x4xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_flatten1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x?x4xf32>) -> tensor<?x4xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<2x?x4xf32> -> tensor<3xindex>
// CHECK:           [[VAR_1_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_0_]] : tensor<3xindex>, index -> index
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.mul [[VAR_1_]], [[CST_1_]] : index, index -> index
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_1_]] : tensor<3xindex>, index -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = shape.mul [[VAR_2_]], [[VAR_3_]] : index, index -> index
// CHECK-DAG:       [[VAR_5_:%.+]] = shape.get_extent [[VAR_0_]], [[CST_2_]] : tensor<3xindex>, index -> index
// CHECK:           [[VAR_6_:%.+]] = shape.mul [[VAR_5_]], [[CST_1_]] : index, index -> index
// CHECK:           [[VAR_7_:%.+]] = shape.from_extents [[VAR_4_]], [[VAR_6_]] : index, index
// CHECK:           [[VAR_8_:%.+]] = shape.to_extent_tensor [[VAR_7_]] : !shape.shape -> tensor<2xindex>
// CHECK:           [[VAR_9_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_8_]] : (tensor<2x?x4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
// CHECK:           return [[VAR_9_]] : tensor<?x4xf32>
// CHECK:         }
