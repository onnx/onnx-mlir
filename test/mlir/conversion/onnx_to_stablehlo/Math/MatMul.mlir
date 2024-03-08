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

func.func @test_onnx_to_matmul2d_dynM(%arg0 : tensor<?x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_onnx_to_matmul2d_dynM
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<?x16xf32> {
// CHECK-DAG:       [[SHAPE_A_INDEX_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x8xf32> -> tensor<2xindex>
// CHECK-DAG:       [[SHAPE_A_:%.+]] = arith.index_cast [[SHAPE_A_INDEX_]] : tensor<2xindex> to tensor<2xi64>
// CHECK-DAG:       [[SHAPE_A_BCAST_:%.+]] = stablehlo.concatenate [[SHAPE_A_]], dim = 0 : (tensor<2xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[SHAPE_A_BCAST_]], dims = [0, 1] : (tensor<?x8xf32>, tensor<2xi64>) -> tensor<?x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_1_]], dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot [[VAR_0_]], [[VAR_1_]] : (tensor<?x8xf32>, tensor<8x16xf32>) -> tensor<?x16xf32>
// CHECK:           return [[VAR_2_]] : tensor<?x16xf32>
// CHECK:         }

// -----

func.func @test_onnx_to_matmul2d_dynN(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}


// CHECK-LABEL:  func.func @test_onnx_to_matmul2d_dynN
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x?xf32>) -> tensor<4x?xf32> {
// CHECK-DAG:       [[SHAPE_B_INDEX_:%.+]] = shape.shape_of [[PARAM_1_]] : tensor<8x?xf32> -> tensor<2xindex>
// CHECK-DAG:       [[SHAPE_B_:%.+]] = arith.index_cast [[SHAPE_B_INDEX_]] : tensor<2xindex> to tensor<2xi64>
// CHECK-DAG:       [[SHAPE_B_BCAST_:%.+]] = stablehlo.concatenate [[SHAPE_B_]], dim = 0 : (tensor<2xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_0_]], dims = [0, 1] : (tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[SHAPE_B_BCAST_]], dims = [0, 1] : (tensor<8x?xf32>, tensor<2xi64>) -> tensor<8x?xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot [[VAR_0_]], [[VAR_1_]] : (tensor<4x8xf32>, tensor<8x?xf32>) -> tensor<4x?xf32>
// CHECK:           return [[VAR_2_]] : tensor<4x?xf32>
// CHECK:         }

// -----

func.func @test_onnx_to_matmul2d_dynK(%arg0 : tensor<4x?xf32>, %arg1 : tensor<?x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x?xf32>, tensor<?x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}


// CHECK-LABEL:  func.func @test_onnx_to_matmul2d_dynK
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x?xf32>, [[PARAM_1_:%.+]]: tensor<?x16xf32>) -> tensor<4x16xf32> {
// CHECK-DAG:       [[SHAPE_A_INDEX_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<4x?xf32> -> tensor<2xindex>
// CHECK-DAG:       [[SHAPE_A_:%.+]] = arith.index_cast [[SHAPE_A_INDEX_]] : tensor<2xindex> to tensor<2xi64>
// CHECK-DAG:       [[SHAPE_A_BCAST_:%.+]] = stablehlo.concatenate [[SHAPE_A_]], dim = 0 : (tensor<2xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[SHAPE_B_INDEX_:%.+]] = shape.shape_of [[PARAM_1_]] : tensor<?x16xf32> -> tensor<2xindex>
// CHECK-DAG:       [[SHAPE_B_:%.+]] = arith.index_cast [[SHAPE_B_INDEX_]] : tensor<2xindex> to tensor<2xi64>
// CHECK-DAG:       [[SHAPE_B_BCAST_:%.+]] = stablehlo.concatenate [[SHAPE_B_]], dim = 0 : (tensor<2xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[SHAPE_A_BCAST_]], dims = [0, 1] : (tensor<4x?xf32>, tensor<2xi64>) -> tensor<4x?xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[SHAPE_B_BCAST_]], dims = [0, 1] : (tensor<?x16xf32>, tensor<2xi64>) -> tensor<?x16xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot [[VAR_0_]], [[VAR_1_]] : (tensor<4x?xf32>, tensor<?x16xf32>) -> tensor<4x16xf32>
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

func.func @test_onnx_to_matmul3dbcast_dynMN(%arg0 : tensor<100x?x8xf32>, %arg1 : tensor<8x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x?x8xf32>, tensor<8x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_onnx_to_matmul3dbcast_dynMN
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x?x8xf32>, [[PARAM_1_:%.+]]: tensor<8x?xf32>) -> tensor<100x?x?xf32> {
// CHECK-DAG:       [[BDIM:%.+]] = arith.constant dense<100> : tensor<1xi64>
// CHECK-DAG:       [[SHAPE_A_INDEX_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<100x?x8xf32> -> tensor<3xindex>
// CHECK-DAG:       [[SHAPE_A_:%.+]] = arith.index_cast [[SHAPE_A_INDEX_]] : tensor<3xindex> to tensor<3xi64>
// CHECK-DAG:       [[SHAPE_A_BCAST_:%.+]] = stablehlo.concatenate [[SHAPE_A_]], dim = 0 : (tensor<3xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[SHAPE_B_INDEX_:%.+]] = shape.shape_of [[PARAM_1_]] : tensor<8x?xf32> -> tensor<2xindex>
// CHECK-DAG:       [[SHAPE_B_:%.+]] = arith.index_cast [[SHAPE_B_INDEX_]] : tensor<2xindex> to tensor<2xi64>
// CHECK-DAG:       [[SHAPE_B_BCAST_:%.+]] = stablehlo.concatenate [[BDIM]], [[SHAPE_B_]], dim = 0 : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[SHAPE_A_BCAST_]], dims = [0, 1, 2] : (tensor<100x?x8xf32>, tensor<3xi64>) -> tensor<100x?x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[SHAPE_B_BCAST_]], dims = [1, 2] : (tensor<8x?xf32>, tensor<3xi64>) -> tensor<100x8x?xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot_general [[VAR_0_]], [[VAR_1_]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<100x?x8xf32>, tensor<100x8x?xf32>) -> tensor<100x?x?xf32>
// CHECK:           return [[VAR_2_]] : tensor<100x?x?xf32>
// CHECK:         }

// -----

func.func @test_onnx_to_matmul3dbcast_dynBatch(%arg0 : tensor<4x8xf32>, %arg1 : tensor<?x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<?x8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_onnx_to_matmul3dbcast_dynBatch
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<?x8x16xf32>) -> tensor<?x4x16xf32> {
// CHECK-DAG:       [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[BATCH_DIM_INDEX_:%.+]] = tensor.dim [[PARAM_1_]], [[C0]] : tensor<?x8x16xf32>
// CHECK-DAG:       [[BATCH_DIM_:%.+]] = arith.index_cast [[BATCH_DIM_INDEX_]] : index to i64
// CHECK-DAG:       [[BATCH_DIM_TENSOR_:%.+]] = tensor.from_elements [[BATCH_DIM_]] : tensor<1xi64>
// CHECK-DAG:       [[SHAPE_A_:%.+]] = arith.constant dense<[4, 8]> : tensor<2xi64>
// CHECK-DAG:       [[SHAPE_A_BCAST_:%.+]] = stablehlo.concatenate [[BATCH_DIM_TENSOR_]], [[SHAPE_A_]], dim = 0 : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[SHAPE_B_INDEX_:%.+]] = shape.shape_of [[PARAM_1_]] : tensor<?x8x16xf32> -> tensor<3xindex>
// CHECK-DAG:       [[SHAPE_B_:%.+]] = arith.index_cast [[SHAPE_B_INDEX_]] : tensor<3xindex> to tensor<3xi64>
// CHECK-DAG:       [[SHAPE_B_BCAST_:%.+]] = stablehlo.concatenate [[SHAPE_B_]], dim = 0 : (tensor<3xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[SHAPE_A_BCAST_]], dims = [1, 2] : (tensor<4x8xf32>, tensor<3xi64>) -> tensor<?x4x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[SHAPE_B_BCAST_]], dims = [0, 1, 2] : (tensor<?x8x16xf32>, tensor<3xi64>) -> tensor<?x8x16xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dot_general [[VAR_0_]], [[VAR_1_]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<?x4x8xf32>, tensor<?x8x16xf32>) -> tensor<?x4x16xf32>
// CHECK:           return [[VAR_2_]] : tensor<?x4x16xf32>
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
