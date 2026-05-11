// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --shape-inference --constprop-onnx %s -split-input-file | FileCheck %s

func.func @test_add_onnx_plus_tosa_const() -> tensor<f32> {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = "tosa.const"() <{value = dense<2.0> : tensor<f32>}> : () -> tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "onnx.Return"(%2) : (tensor<f32>) -> ()
}
// CHECK-LABEL: @test_add_onnx_plus_tosa_const() -> tensor<f32>
// CHECK:       [[C:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<f32>
// CHECK-NOT:   onnx.Add

// -----

func.func @test_add_two_tosa_consts() -> tensor<f32> {
  %0 = "tosa.const"() <{value = dense<2.0> : tensor<f32>}> : () -> tensor<f32>
  %1 = "tosa.const"() <{value = dense<3.0> : tensor<f32>}> : () -> tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "onnx.Return"(%2) : (tensor<f32>) -> ()
}
// CHECK-LABEL: @test_add_two_tosa_consts() -> tensor<f32>
// CHECK:       [[C:%.+]] = onnx.Constant dense<5.000000e+00> : tensor<f32>
// CHECK-NOT:   onnx.Add

// -----

func.func @test_neg_tosa_const() -> tensor<2xf32> {
  %0 = "tosa.const"() <{value = dense<[1.0, 2.0]> : tensor<2xf32>}> : () -> tensor<2xf32>
  %1 = "onnx.Neg"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  "onnx.Return"(%1) : (tensor<2xf32>) -> ()
}
// CHECK-LABEL: @test_neg_tosa_const() -> tensor<2xf32>
// CHECK:       [[C:%.+]] = onnx.Constant dense<[-1.000000e+00, -2.000000e+00]> : tensor<2xf32>
// CHECK-NOT:   onnx.Neg

// -----

func.func @test_add_two_arith_consts() -> tensor<f32> {
  %0 = arith.constant dense<2.0> : tensor<f32>
  %1 = arith.constant dense<3.0> : tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "onnx.Return"(%2) : (tensor<f32>) -> ()
}
// CHECK-LABEL: @test_add_two_arith_consts() -> tensor<f32>
// CHECK:       [[C:%.+]] = onnx.Constant dense<5.000000e+00> : tensor<f32>
// CHECK-NOT:   onnx.Add
// CHECK-NOT:   arith.constant

// -----

func.func @test_add_onnx_tosa_arith_consts() -> tensor<f32> {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = "tosa.const"() <{value = dense<2.0> : tensor<f32>}> : () -> tensor<f32>
  %2 = arith.constant dense<3.0> : tensor<f32>
  %3 = "onnx.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %4 = "onnx.Add"(%3, %2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "onnx.Return"(%4) : (tensor<f32>) -> ()
}
// CHECK-LABEL: @test_add_onnx_tosa_arith_consts() -> tensor<f32>
// CHECK:       [[C:%.+]] = onnx.Constant dense<6.000000e+00> : tensor<f32>
// CHECK-NOT:   onnx.Add
// CHECK-NOT:   tosa.const
// CHECK-NOT:   arith.constant

// -----

func.func @test_add_dyn_plus_tosa_const(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "tosa.const"() <{value = dense<1.0> : tensor<f32>}> : () -> tensor<f32>
  %1 = "onnx.Add"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "onnx.Return"(%1) : (tensor<f32>) -> ()
}
// CHECK-LABEL: @test_add_dyn_plus_tosa_const
// CHECK:       onnx.Add
// CHECK-NOT:   onnx.Constant
