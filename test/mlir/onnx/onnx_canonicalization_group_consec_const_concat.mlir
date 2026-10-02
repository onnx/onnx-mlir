// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference --constprop-onnx --cse %s -split-input-file -verify-diagnostics | FileCheck %s

func.func @test_group_consecutive_const_concat(%arg0: tensor<1x3x4xf32>) -> tensor<1x9x4xf32> {
  %c0 = onnx.Constant dense<1.000000e+00> : tensor<1x3x4xf32>
  %0 = "onnx.Concat"(%arg0, %c0, %c0) {axis = 1 : si64} : (tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x9x4xf32>
  return %0 : tensor<1x9x4xf32>
// CHECK-LABEL:  func.func @test_group_consecutive_const_concat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x4xf32>) -> tensor<1x9x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1x6x4xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<1x3x4xf32>, tensor<1x6x4xf32>) -> tensor<1x9x4xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x9x4xf32>
// CHECK:         }
}

// -----

func.func @test_group_two_const_concat_runs(%arg0: tensor<1x3x4xf32>) -> tensor<1x15x4xf32> {
  %c1 = onnx.Constant dense<1.000000e+00> : tensor<1x3x4xf32>
  %c2 = onnx.Constant dense<2.000000e+00> : tensor<1x3x4xf32>
  %0 = "onnx.Concat"(%c1, %c1, %arg0, %c2, %c2) {axis = 1 : si64} : (tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x15x4xf32>
  return %0 : tensor<1x15x4xf32>
// CHECK-LABEL:  func.func @test_group_two_const_concat_runs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x4xf32>) -> tensor<1x15x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1x6x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x6x4xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_0_]], [[PARAM_0_]], [[VAR_1_]]) {axis = 1 : si64} : (tensor<1x6x4xf32>, tensor<1x3x4xf32>, tensor<1x6x4xf32>) -> tensor<1x15x4xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x15x4xf32>
// CHECK:         }
}

// -----

func.func @test_no_group_nonconsecutive_const_concat(%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x4xf32>) -> tensor<1x9x4xf32> {
  %c0 = onnx.Constant dense<1.000000e+00> : tensor<1x3x4xf32>
  %0 = "onnx.Concat"(%arg0, %c0, %arg1) {axis = 1 : si64} : (tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x9x4xf32>
  return %0 : tensor<1x9x4xf32>
// CHECK-LABEL:  func.func @test_no_group_nonconsecutive_const_concat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1x3x4xf32>) -> tensor<1x9x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1x3x4xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[VAR_0_]], [[PARAM_1_]]) {axis = 1 : si64} : (tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x9x4xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x9x4xf32>
}
