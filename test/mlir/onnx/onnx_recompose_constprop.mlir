// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --shape-inference --constprop-onnx --recompose-onnx --canonicalize %s -split-input-file | FileCheck %s

// Tests that the factored HardSigmoid decomposition clip(x + b/a, 0, 1/a) * a
// is first distributed by const-prop into clip(x * a + b, 0, 1) and then
// recomposed into onnx.HardSigmoid.

// clip(x + b/a, 0, 1/a) * a with a=1/6, b=0.5 in f32
func.func @test_hardsigmoid_factored_f32(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %b_over_a = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %inv_a = onnx.Constant dense<6.000000e+00> : tensor<f32>
  %0 = "onnx.Add"(%arg0, %b_over_a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Clip"(%0, %zero, %inv_a) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Mul"(%1, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %2 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardsigmoid_factored_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]])
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// clip(x + b/a, 0, 1/a) * a with a=1/6, b=0.5 in bf16
func.func @test_hardsigmoid_factored_bf16(%arg0 : tensor<?x?x3072xbf16>) -> tensor<?x?x3072xbf16> {
  %a = onnx.Constant dense<0.166992188> : tensor<bf16>
  %b_over_a = onnx.Constant dense<3.000000e+00> : tensor<bf16>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<bf16>
  %inv_a = onnx.Constant dense<6.000000e+00> : tensor<bf16>
  %0 = "onnx.Add"(%arg0, %b_over_a) : (tensor<?x?x3072xbf16>, tensor<bf16>) -> tensor<?x?x3072xbf16>
  %1 = "onnx.Clip"(%0, %zero, %inv_a) : (tensor<?x?x3072xbf16>, tensor<bf16>, tensor<bf16>) -> tensor<?x?x3072xbf16>
  %2 = "onnx.Mul"(%1, %a) : (tensor<?x?x3072xbf16>, tensor<bf16>) -> tensor<?x?x3072xbf16>
  return %2 : tensor<?x?x3072xbf16>

// CHECK-LABEL:  func.func @test_hardsigmoid_factored_bf16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xbf16>) -> tensor<?x?x3072xbf16> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]])
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xbf16>
// CHECK:         }
}

// -----

// clip(x + b/a, 0, 1/a) * a with a=1/6, b=0.5 in f16
func.func @test_hardsigmoid_factored_f16(%arg0 : tensor<?x?x3072xf16>) -> tensor<?x?x3072xf16> {
  %a = onnx.Constant dense<0.166625977> : tensor<f16>
  %b_over_a = onnx.Constant dense<3.000000e+00> : tensor<f16>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f16>
  %inv_a = onnx.Constant dense<6.000000e+00> : tensor<f16>
  %0 = "onnx.Add"(%arg0, %b_over_a) : (tensor<?x?x3072xf16>, tensor<f16>) -> tensor<?x?x3072xf16>
  %1 = "onnx.Clip"(%0, %zero, %inv_a) : (tensor<?x?x3072xf16>, tensor<f16>, tensor<f16>) -> tensor<?x?x3072xf16>
  %2 = "onnx.Mul"(%1, %a) : (tensor<?x?x3072xf16>, tensor<f16>) -> tensor<?x?x3072xf16>
  return %2 : tensor<?x?x3072xf16>

// CHECK-LABEL:  func.func @test_hardsigmoid_factored_f16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf16>) -> tensor<?x?x3072xf16> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]])
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf16>
// CHECK:         }
}

// -----

// clip(x + b/a, 0, 1/a) * a with a=0.2, b=0.5 in f32
func.func @test_hardsigmoid_factored_default_alpha_beta(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %a = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %b_over_a = onnx.Constant dense<2.500000e+00> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %inv_a = onnx.Constant dense<5.000000e+00> : tensor<f32>
  %0 = "onnx.Add"(%arg0, %b_over_a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Clip"(%0, %zero, %inv_a) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Mul"(%1, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %2 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardsigmoid_factored_default_alpha_beta
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]])
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// clip(x + b/a, 0, 1/a) * a with a=0.2, b=0.5 in f16
func.func @test_hardsigmoid_factored_default_alpha_beta_f16(%arg0 : tensor<?x?x3072xf16>) -> tensor<?x?x3072xf16> {
  %a = onnx.Constant dense<2.000000e-01> : tensor<f16>
  %b_over_a = onnx.Constant dense<2.500000e+00> : tensor<f16>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f16>
  %inv_a = onnx.Constant dense<5.000000e+00> : tensor<f16>
  %0 = "onnx.Add"(%arg0, %b_over_a) : (tensor<?x?x3072xf16>, tensor<f16>) -> tensor<?x?x3072xf16>
  %1 = "onnx.Clip"(%0, %zero, %inv_a) : (tensor<?x?x3072xf16>, tensor<f16>, tensor<f16>) -> tensor<?x?x3072xf16>
  %2 = "onnx.Mul"(%1, %a) : (tensor<?x?x3072xf16>, tensor<f16>) -> tensor<?x?x3072xf16>
  return %2 : tensor<?x?x3072xf16>

// CHECK-LABEL:  func.func @test_hardsigmoid_factored_default_alpha_beta_f16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf16>) -> tensor<?x?x3072xf16> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]])
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf16>
// CHECK:         }
}

// -----

// clip(x + b/a, 0, 1/a) * a with a=0.2, b=0.5 in bf16
func.func @test_hardsigmoid_factored_default_alpha_beta_bf16(%arg0 : tensor<?x?x3072xbf16>) -> tensor<?x?x3072xbf16> {
  %a = onnx.Constant dense<2.000000e-01> : tensor<bf16>
  %b_over_a = onnx.Constant dense<2.500000e+00> : tensor<bf16>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<bf16>
  %inv_a = onnx.Constant dense<5.000000e+00> : tensor<bf16>
  %0 = "onnx.Add"(%arg0, %b_over_a) : (tensor<?x?x3072xbf16>, tensor<bf16>) -> tensor<?x?x3072xbf16>
  %1 = "onnx.Clip"(%0, %zero, %inv_a) : (tensor<?x?x3072xbf16>, tensor<bf16>, tensor<bf16>) -> tensor<?x?x3072xbf16>
  %2 = "onnx.Mul"(%1, %a) : (tensor<?x?x3072xbf16>, tensor<bf16>) -> tensor<?x?x3072xbf16>
  return %2 : tensor<?x?x3072xbf16>

// CHECK-LABEL:  func.func @test_hardsigmoid_factored_default_alpha_beta_bf16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xbf16>) -> tensor<?x?x3072xbf16> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]])
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xbf16>
// CHECK:         }
}
