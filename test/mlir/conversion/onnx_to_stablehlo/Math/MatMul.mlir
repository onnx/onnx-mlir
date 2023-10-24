// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_onnx_to_matmul2d(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_onnx_to_matmul2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_0_]], dims = [0, 1] : (tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_1_]], dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot [[VAR_0_]], [[VAR_1_]] : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
// CHECK:           return [[VAR_2_]] : tensor<4x16xf32>
// CHECK:         }

// -----

func.func @test_onnx_to_matmul3d(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_onnx_to_matmul3d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<100x8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_0_]], dims = [0, 1, 2] : (tensor<100x4x8xf32>) -> tensor<100x4x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_1_]], dims = [0, 1, 2] : (tensor<100x8x16xf32>) -> tensor<100x8x16xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot_general [[VAR_0_]], [[VAR_1_]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_2_]] : tensor<100x4x16xf32>
// CHECK:         }

// -----

func.func @test_onnx_to_matmul3dbcast(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_onnx_to_matmul3dbcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_0_]], dims = [0, 1, 2] : (tensor<100x4x8xf32>) -> tensor<100x4x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_1_]], dims = [1, 2] : (tensor<8x16xf32>) -> tensor<100x8x16xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot_general [[VAR_0_]], [[VAR_1_]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_2_]] : tensor<100x4x16xf32>
// CHECK:         }

// -----

func.func @test_onnx_1d(%arg0 : tensor<6xf32>, %arg1 : tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_onnx_1d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<6xf32>, [[PARAM_1_:%.+]]: tensor<6xf32>) -> tensor<f32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_0_]], dims = [0] : (tensor<6xf32>) -> tensor<6xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_1_]], dims = [0] : (tensor<6xf32>) -> tensor<6xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot [[VAR_0_]], [[VAR_1_]] : (tensor<6xf32>, tensor<6xf32>) -> tensor<f32>
// CHECK:           return [[VAR_2_]] : tensor<f32>
// CHECK:         }

// -----

func.func @test_onnx_12d(%arg0 : tensor<6xf32>, %arg1 : tensor<6x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<6xf32>, tensor<6x2xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_onnx_12d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<6xf32>, [[PARAM_1_:%.+]]: tensor<6x2xf32>) -> tensor<2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_0_]], dims = [0] : (tensor<6xf32>) -> tensor<6xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_1_]], dims = [0, 1] : (tensor<6x2xf32>) -> tensor<6x2xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot [[VAR_0_]], [[VAR_1_]] : (tensor<6xf32>, tensor<6x2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAR_2_]] : tensor<2xf32>
// CHECK:         }

// -----

func.func @test_onnx_21d(%arg0 : tensor<2x6xf32>, %arg1 : tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x6xf32>, tensor<6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_onnx_21d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x6xf32>, [[PARAM_1_:%.+]]: tensor<6xf32>) -> tensor<2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_0_]], dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_1_]], dims = [0] : (tensor<6xf32>) -> tensor<6xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot [[VAR_0_]], [[VAR_1_]] : (tensor<2x6xf32>, tensor<6xf32>) -> tensor<2xf32>
// CHECK:           return [[VAR_2_]] : tensor<2xf32>
// CHECK:         }
