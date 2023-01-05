// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_onnx_to_matmul2d(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_to_matmul2d(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32>
  // CHECK:   %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  // CHECK:   return %0 : tensor<4x16xf32>
}

func.func @test_onnx_to_matmul3d(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_to_matmul3d(%arg0: tensor<100x4x8xf32>, %arg1: tensor<100x8x16xf32>) -> tensor<100x4x16xf32> {
  // CHECK:   %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<100x4x16xf32>
  // CHECK:   return %0 : tensor<100x4x16xf32>  
}

func.func @test_onnx_to_matmul3dbcast(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_to_matmul3dbcast(%arg0: tensor<100x4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<100x4x16xf32> {
  // CHECK:   %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[1, 2]> : vector<2xi64>} : (tensor<8x16xf32>) -> tensor<100x8x16xf32>
  // CHECK:   %1 = "mhlo.dot_general"(%arg0, %0) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<100x4x16xf32>
  // CHECK:   return %1 : tensor<100x4x16xf32>
}

func.func @test_onnx_to_matmul2d_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK:   func.func @test_onnx_to_matmul2d_dyn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK:     %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK:     return %0 : tensor<?x?xf32>
}

func.func @test_onnx_to_matmul3d_dyn(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_to_matmul3d_dyn(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK:   %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK:   return %0 : tensor<?x?x?xf32>  
}

func.func @test_onnx_1d(%arg0 : tensor<6xf32>, %arg1 : tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_1d(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<f32> {
  // CHECK:   %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<f32>
  // CHECK:   return %0 : tensor<f32>
}

func.func @test_onnx_12d(%arg0 : tensor<6xf32>, %arg1 : tensor<6x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<6xf32>, tensor<6x2xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_12d(%arg0: tensor<6xf32>, %arg1: tensor<6x2xf32>) -> tensor<2xf32> {
  // CHECK:   %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<6xf32>, tensor<6x2xf32>) -> tensor<2xf32>
  // CHECK:   return %0 : tensor<2xf32>
}

func.func @test_onnx_21d(%arg0 : tensor<2x6xf32>, %arg1 : tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x6xf32>, tensor<6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_21d(%arg0: tensor<2x6xf32>, %arg1: tensor<6xf32>) -> tensor<2xf32> {
  // CHECK:   %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x6xf32>, tensor<6xf32>) -> tensor<2xf32>
  // CHECK:   return %0 : tensor<2xf32>
}