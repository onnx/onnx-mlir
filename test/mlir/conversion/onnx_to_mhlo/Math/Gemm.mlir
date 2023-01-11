// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s -split-input-file | FileCheck %s

func.func @test_gemm_bias_none(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>) -> tensor<10x10xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 ="onnx.Gemm"(%arg0, %arg1, %bias) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x5xf32>, tensor<5x10xf32>, none) -> tensor<10x10xf32>
 "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_gemm_bias_none
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>) -> tensor<10x10xf32> {
// CHECK:         [[VAR_0_:%.+]] = "mhlo.dot"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<10x10xf32>
}

func.func @test_gemm_bias_1d(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<10x10xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x5xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
 "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_gemm_bias_1d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = "mhlo.dot"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<10x10xf32>
// CHECK-DAG:    [[VAR_1_:%.+]] = "mhlo.broadcast_in_dim"([[PARAM_2_]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:   [[VAR_2_:%.+]] = mhlo.add [[VAR_0_]], [[VAR_1_]] : tensor<10x10xf32>
}

func.func @test_gemm_bias_2d(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 2.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x5xf32>, tensor<5x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
 "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_gemm_bias_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<2.000000e+00> : tensor<10x10xf32>
// CHECK-DAG:    [[VAR_1_:%.+]] = "mhlo.dot"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:   [[VAR_2_:%.+]] = mhlo.multiply [[VAR_1_]], [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:   [[VAR_3_:%.+]] = mhlo.add [[VAR_2_]], [[PARAM_2_]] : tensor<10x10xf32>
}

func.func @test_gemm_transA(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<10x10xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 2.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
 "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_gemm_transA
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x10xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<2.000000e+00> : tensor<10x10xf32>  
// CHECK-DAG:    [[VAR_1_:%.+]] = "mhlo.transpose"([[PARAM_0_]]) {permutation = dense<[1, 0]> : vector<2xi64>} : (tensor<5x10xf32>) -> tensor<10x5xf32>
// CHECK-DAG:    [[VAR_2_:%.+]] = "mhlo.dot"([[VAR_1_]], [[PARAM_1_]]) : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<10x10xf32>
// CHECK-DAG:    [[VAR_3_:%.+]] = "mhlo.broadcast_in_dim"([[PARAM_2_]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:   [[VAR_4_:%.+]] = mhlo.multiply [[VAR_3_]], [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:   [[VAR_5_:%.+]] = mhlo.add [[VAR_2_]], [[VAR_4_]] : tensor<10x10xf32>
}

func.func @test_gemm_transAB(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<5xf32>) -> tensor<5x5xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 2.0 : f32, beta = 3.0 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<10x5xf32>, tensor<5x10xf32>, tensor<5xf32>) -> tensor<5x5xf32>
 "func.return"(%0) : (tensor<5x5xf32>) -> ()
// CHECK-LABEL:  func @test_gemm_transAB
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<5xf32>) -> tensor<5x5xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<3.000000e+00> : tensor<5x5xf32>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<2.000000e+00> : tensor<5x5xf32>
// CHECK-DAG:    [[VAR_2_:%.+]] = "mhlo.transpose"([[PARAM_0_]]) {permutation = dense<[1, 0]> : vector<2xi64>} : (tensor<10x5xf32>) -> tensor<5x10xf32>
// CHECK-DAG:    [[VAR_3_:%.+]] = "mhlo.transpose"([[PARAM_1_]]) {permutation = dense<[1, 0]> : vector<2xi64>} : (tensor<5x10xf32>) -> tensor<10x5xf32>
// CHECK-DAG:    [[VAR_4_:%.+]] = "mhlo.dot"([[VAR_2_]], [[VAR_3_]]) : (tensor<5x10xf32>, tensor<10x5xf32>) -> tensor<5x5xf32>
// CHECK-DAG:    [[VAR_5_:%.+]] = mhlo.multiply [[VAR_4_]], [[VAR_1_]] : tensor<5x5xf32>
// CHECK-DAG:    [[VAR_6_:%.+]] = "mhlo.broadcast_in_dim"([[PARAM_2_]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<5x5xf32>
// CHECK-NEXT:   [[VAR_7_:%.+]] = mhlo.multiply [[VAR_6_]], [[VAR_0_]] : tensor<5x5xf32>
// CHECK-NEXT:   [[VAR_8_:%.+]] = mhlo.add [[VAR_5_]], [[VAR_7_]] : tensor<5x5xf32>
}

func.func @test_gemm_unknown_dims(%arg0: tensor<?x5xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<?x10xf32> {
  %0= "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<?x5xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_gemm_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<?x10xf32> {
// CHECK-NEXT:    [[VAR_0_:%.+]] = "mhlo.dot"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x5xf32>, tensor<5x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:    [[VAR_1_:%.+]] = shape.shape_of [[VAR_0_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK-NEXT:    [[VAR_2_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[PARAM_2_]], [[VAR_1_]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK-NEXT:    [[VAR_3_:%.+]] = mhlo.add [[VAR_0_]], [[VAR_2_]] : tensor<?x10xf32>
}

func.func @test_gemm_unknown_dims_not_lowered(%arg0: tensor<?x5xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<?xf32>) -> tensor<?x10xf32> {
  %0= "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<?x5xf32>, tensor<5x10xf32>, tensor<?xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_gemm_unknown_dims_not_lowered
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<?xf32>) -> tensor<?x10xf32> {
// CHECK-NEXT:    [[VAR_0_:%.+]] = "mhlo.dot"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x5xf32>, tensor<5x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:    [[VAR_1_:%.+]] = shape.shape_of [[VAR_0_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK-NEXT:    [[VAR_2_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[PARAM_2_]], [[VAR_1_]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK-NEXT:    [[VAR_3_:%.+]] = mhlo.add [[VAR_0_]], [[VAR_2_]] : tensor<?x10xf32>
}

func.func @test_gemm_not_lowered(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<10x10xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 2.0 : f32, beta = 1.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
 "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_gemm_not_lowered
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x10xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<2.000000e+00> : tensor<10x10xf32>
// CHECK-DAG:    [[VAR_1_:%.+]] = "mhlo.transpose"([[PARAM_0_]]) {permutation = dense<[1, 0]> : vector<2xi64>} : (tensor<5x10xf32>) -> tensor<10x5xf32>
// CHECK-DAG:    [[VAR_2_:%.+]] = "mhlo.dot"([[VAR_1_]], [[PARAM_1_]]) : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<10x10xf32>
// CHECK-DAG:    [[VAR_3_:%.+]] = mhlo.multiply [[VAR_2_]], [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    [[VAR_4_:%.+]] = "mhlo.broadcast_in_dim"([[PARAM_2_]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:    [[VAR_5_:%.+]] = mhlo.add [[VAR_3_]], [[VAR_4_]] : tensor<10x10xf32>
}

func.func @test_exceed_limit_gemm(%arg0 : tensor<32769x5xf32>, %arg1 : tensor<5x32769xf32>, %arg2: tensor<32769xf32>) -> tensor<32769x32769xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<32769x5xf32>, tensor<5x32769xf32>, tensor<32769xf32>) -> tensor<32769x32769xf32>
 "func.return"(%0) : (tensor<32769x32769xf32>) -> ()
// CHECK-LABEL:  func @test_exceed_limit_gemm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<32769x5xf32>, [[PARAM_1_:%.+]]: tensor<5x32769xf32>, [[PARAM_2_:%.+]]: tensor<32769xf32>) -> tensor<32769x32769xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = "mhlo.dot"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<32769x5xf32>, tensor<5x32769xf32>) -> tensor<32769x32769xf32>
// CHECK-DAG:    [[VAR_1_:%.+]] = "mhlo.broadcast_in_dim"([[PARAM_2_]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<32769xf32>) -> tensor<32769x32769xf32>
// CHECK-NEXT:    [[VAR_2_:%.+]] = mhlo.add [[VAR_0_]], [[VAR_1_]] : tensor<32769x32769xf32>
}