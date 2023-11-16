// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_clip(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> {
  %0 = "onnx.Clip"(%arg0, %arg1, %arg2) : (tensor<3xf32>, tensor<f32>, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func @test_clip
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>) -> tensor<3xf32> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = stablehlo.clamp [[PARAM_1_]], [[PARAM_0_]], [[PARAM_2_]] : (tensor<f32>, tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<3xf32>
// CHECK-NEXT:   }
}

// -----

// Test when min is none
func.func @test_clip_default_min_f32(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %cst, %arg2) : (tensor<3xf32>, none, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func @test_clip_default_min_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>) -> tensor<3xf32> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:     [[VAR_1_:%.+]] = stablehlo.clamp [[VAR_0_]], [[PARAM_0_]], [[PARAM_2_]] : (tensor<f32>, tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:     return [[VAR_1_]] : tensor<3xf32>
// CHECK-NEXT:   }
}

// Test when min is none
func.func @test_clip_default_min_f64(%arg0: tensor<3xf64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<3xf64> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %cst, %arg2) : (tensor<3xf64>, none, tensor<f64>) -> tensor<3xf64>
  return %0 : tensor<3xf64>
// CHECK-LABEL:  func @test_clip_default_min_f64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf64>, [[PARAM_1_:%.+]]: tensor<f64>, [[PARAM_2_:%.+]]: tensor<f64>) -> tensor<3xf64> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
// CHECK-NEXT:     [[VAR_1_:%.+]] = stablehlo.clamp [[VAR_0_]], [[PARAM_0_]], [[PARAM_2_]] : (tensor<f64>, tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT:     return [[VAR_1_]] : tensor<3xf64>
// CHECK-NEXT:   }
}

// Test when min is none
func.func @test_clip_default_min_i32(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<3xi32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %cst, %arg2) : (tensor<3xi32>, none, tensor<i32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
// CHECK-LABEL:  func @test_clip_default_min_i32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xi32>, [[PARAM_1_:%.+]]: tensor<i32>, [[PARAM_2_:%.+]]: tensor<i32>) -> tensor<3xi32> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = stablehlo.constant dense<-2147483648> : tensor<i32>
// CHECK-NEXT:     [[VAR_1_:%.+]] = stablehlo.clamp [[VAR_0_]], [[PARAM_0_]], [[PARAM_2_]] : (tensor<i32>, tensor<3xi32>, tensor<i32>) -> tensor<3xi32>
// CHECK-NEXT:     return [[VAR_1_]] : tensor<3xi32>
// CHECK-NEXT:   }
}

// -----

// Test when max is none
func.func @test_clip_default_max_f32(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %arg1, %cst) : (tensor<3xf32>, tensor<f32>, none) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func @test_clip_default_max_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>) -> tensor<3xf32> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK-NEXT:     [[VAR_1_:%.+]] = stablehlo.clamp [[PARAM_1_]], [[PARAM_0_]], [[VAR_0_]] : (tensor<f32>, tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:     return [[VAR_1_]] : tensor<3xf32>
// CHECK-NEXT:   }
}

// Test when max is none
func.func @test_clip_default_max_f64(%arg0: tensor<3xf64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<3xf64> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %arg1, %cst) : (tensor<3xf64>, tensor<f64>, none) -> tensor<3xf64>
  return %0 : tensor<3xf64>
// CHECK-LABEL:  func @test_clip_default_max_f64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf64>, [[PARAM_1_:%.+]]: tensor<f64>, [[PARAM_2_:%.+]]: tensor<f64>) -> tensor<3xf64> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK-NEXT:     [[VAR_1_:%.+]] = stablehlo.clamp [[PARAM_1_]], [[PARAM_0_]], [[VAR_0_]] : (tensor<f64>, tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
// CHECK-NEXT:     return [[VAR_1_]] : tensor<3xf64>
// CHECK-NEXT:   }
}

// Test when max is none
func.func @test_clip_default_max_i32(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<3xi32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %arg1, %cst) : (tensor<3xi32>, tensor<i32>, none) -> tensor<3xi32>
  return %0 : tensor<3xi32>
// CHECK-LABEL:  func @test_clip_default_max_i32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xi32>, [[PARAM_1_:%.+]]: tensor<i32>, [[PARAM_2_:%.+]]: tensor<i32>) -> tensor<3xi32> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = stablehlo.constant dense<2147483647> : tensor<i32>
// CHECK-NEXT:     [[VAR_1_:%.+]] = stablehlo.clamp [[PARAM_1_]], [[PARAM_0_]], [[VAR_0_]] : (tensor<i32>, tensor<3xi32>, tensor<i32>) -> tensor<3xi32>
// CHECK-NEXT:     return [[VAR_1_]] : tensor<3xi32>
// CHECK-NEXT:   }
}
