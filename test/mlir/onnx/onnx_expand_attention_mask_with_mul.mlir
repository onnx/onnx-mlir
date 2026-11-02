// RUN: onnx-mlir-opt --expand-attention-mask %s -split-input-file | FileCheck %s

// -----

// Test Case 1: Basic pattern with Mul (≥4 uses) - should expand both mask and scale
// CHECK-LABEL: func.func @test_matmul_mul_add_softmax_expand
func.func @test_matmul_mul_add_softmax_expand(
    %arg0: tensor<1x12x?x64xf32>,
    %arg1: tensor<1x12x64x?xf32>,
    %scale: tensor<f32>,
    %mask: tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32> {
  
  // Layer 1
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %1 = "onnx.Mul"(%0, %scale) : (tensor<1x12x?x?xf32>, tensor<f32>) -> tensor<1x12x?x?xf32>
  %2 = "onnx.Add"(%1, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %3 = "onnx.Softmax"(%2) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 2
  %4 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %5 = "onnx.Mul"(%4, %scale) : (tensor<1x12x?x?xf32>, tensor<f32>) -> tensor<1x12x?x?xf32>
  %6 = "onnx.Add"(%5, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %7 = "onnx.Softmax"(%6) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 3
  %8 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %9 = "onnx.Mul"(%8, %scale) : (tensor<1x12x?x?xf32>, tensor<f32>) -> tensor<1x12x?x?xf32>
  %10 = "onnx.Add"(%9, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %11 = "onnx.Softmax"(%10) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 4
  %12 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %13 = "onnx.Mul"(%12, %scale) : (tensor<1x12x?x?xf32>, tensor<f32>) -> tensor<1x12x?x?xf32>
  %14 = "onnx.Add"(%13, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %15 = "onnx.Softmax"(%14) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  return %15 : tensor<1x12x?x?xf32>
  
  // CHECK: [[MATMUL0:%.+]] = "onnx.MatMul"(%arg0, %arg1)
  // CHECK: [[CONST1_1:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: [[CONST12_1:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
  // CHECK: [[DIM2_1:%.+]] = "onnx.Dim"([[MATMUL0]]) <{axis = 2 : si64}>
  // CHECK: [[DIM3_1:%.+]] = "onnx.Dim"([[MATMUL0]]) <{axis = 3 : si64}>
  // CHECK: [[SHAPE1:%.+]] = "onnx.Concat"([[CONST1_1]], [[CONST12_1]], [[DIM2_1]], [[DIM3_1]]) <{axis = 0 : si64}>
  // CHECK: [[EXPANDED_SCALE:%.+]] = "onnx.Expand"(%arg2, [[SHAPE1]])
  // CHECK: [[CONST1_2:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: [[CONST12_2:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
  // CHECK: [[DIM2_2:%.+]] = "onnx.Dim"([[MATMUL0]]) <{axis = 2 : si64}>
  // CHECK: [[DIM3_2:%.+]] = "onnx.Dim"([[MATMUL0]]) <{axis = 3 : si64}>
  // CHECK: [[SHAPE2:%.+]] = "onnx.Concat"([[CONST1_2]], [[CONST12_2]], [[DIM2_2]], [[DIM3_2]]) <{axis = 0 : si64}>
  // CHECK: [[EXPANDED_MASK:%.+]] = "onnx.Expand"(%arg3, [[SHAPE2]])
  // CHECK: [[MUL1:%.+]] = "onnx.Mul"([[MATMUL0]], [[EXPANDED_SCALE]])
  // CHECK: [[ADD1:%.+]] = "onnx.Add"([[MUL1]], [[EXPANDED_MASK]])
  // CHECK: "onnx.Softmax"([[ADD1]])
}

// -----

// Test Case 2: Pattern without Mul (backward compatibility) - should still work
// CHECK-LABEL: func.func @test_matmul_add_softmax_no_mul
func.func @test_matmul_add_softmax_no_mul(
    %arg0: tensor<1x12x?x64xf32>,
    %arg1: tensor<1x12x64x?xf32>,
    %mask: tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32> {
  
  // Layer 1
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %1 = "onnx.Add"(%0, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %2 = "onnx.Softmax"(%1) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 2
  %3 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %4 = "onnx.Add"(%3, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %5 = "onnx.Softmax"(%4) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 3
  %6 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %7 = "onnx.Add"(%6, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %8 = "onnx.Softmax"(%7) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 4
  %9 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %10 = "onnx.Add"(%9, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %11 = "onnx.Softmax"(%10) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  return %11 : tensor<1x12x?x?xf32>
  
  // CHECK: [[MATMUL0:%.+]] = "onnx.MatMul"(%arg0, %arg1)
  // CHECK: [[EXPANDED_MASK:%.+]] = "onnx.Expand"(%arg2
  // CHECK: [[ADD1:%.+]] = "onnx.Add"([[MATMUL0]], [[EXPANDED_MASK]])
  // CHECK: "onnx.Softmax"([[ADD1]])
}

// -----

// Test Case 3: Below threshold (<4 uses) - should NOT expand
// CHECK-LABEL: func.func @test_below_threshold
func.func @test_below_threshold(
    %arg0: tensor<1x12x?x64xf32>,
    %arg1: tensor<1x12x64x?xf32>,
    %scale: tensor<f32>,
    %mask: tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32> {
  
  // Only 3 layers - below threshold
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %1 = "onnx.Mul"(%0, %scale) : (tensor<1x12x?x?xf32>, tensor<f32>) -> tensor<1x12x?x?xf32>
  %2 = "onnx.Add"(%1, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %3 = "onnx.Softmax"(%2) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  %4 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %5 = "onnx.Mul"(%4, %scale) : (tensor<1x12x?x?xf32>, tensor<f32>) -> tensor<1x12x?x?xf32>
  %6 = "onnx.Add"(%5, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %7 = "onnx.Softmax"(%6) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  %8 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %9 = "onnx.Mul"(%8, %scale) : (tensor<1x12x?x?xf32>, tensor<f32>) -> tensor<1x12x?x?xf32>
  %10 = "onnx.Add"(%9, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %11 = "onnx.Softmax"(%10) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  return %11 : tensor<1x12x?x?xf32>
  
  // CHECK-NOT: "onnx.Expand"
  // CHECK: "onnx.Mul"({{%.+}}, %arg2)
  // CHECK: "onnx.Add"({{%.+}}, %arg3)
}

// -----

// Test Case 4: Non-scale Mul (non-scalar operand) - should expand mask but NOT scale
// CHECK-LABEL: func.func @test_non_scale_mul
func.func @test_non_scale_mul(
    %arg0: tensor<1x12x?x64xf32>,
    %arg1: tensor<1x12x64x?xf32>,
    %scale: tensor<1x12x?x?xf32>,
    %mask: tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32> {

  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %1 = "onnx.Mul"(%0, %scale) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %2 = "onnx.Add"(%1, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %3 = "onnx.Softmax"(%2) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>

  %4 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %5 = "onnx.Mul"(%4, %scale) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %6 = "onnx.Add"(%5, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %7 = "onnx.Softmax"(%6) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>

  %8 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %9 = "onnx.Mul"(%8, %scale) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %10 = "onnx.Add"(%9, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %11 = "onnx.Softmax"(%10) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>

  %12 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %13 = "onnx.Mul"(%12, %scale) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %14 = "onnx.Add"(%13, %mask) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %15 = "onnx.Softmax"(%14) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>

  return %15 : tensor<1x12x?x?xf32>

  // Mask should be expanded but non-scalar scale should NOT be expanded.
  // CHECK: [[MATMUL0:%.+]] = "onnx.MatMul"(%arg0, %arg1)
  // CHECK: [[CONST1:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: [[CONST12:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
  // CHECK: [[DIM2:%.+]] = "onnx.Dim"([[MATMUL0]]) <{axis = 2 : si64}>
  // CHECK: [[DIM3:%.+]] = "onnx.Dim"([[MATMUL0]]) <{axis = 3 : si64}>
  // CHECK: [[SHAPE:%.+]] = "onnx.Concat"([[CONST1]], [[CONST12]], [[DIM2]], [[DIM3]]) <{axis = 0 : si64}>
  // CHECK: [[EXPANDED_MASK:%.+]] = "onnx.Expand"(%arg3, [[SHAPE]])
  // CHECK: [[MUL1:%.+]] = "onnx.Mul"([[MATMUL0]], %arg2)
  // CHECK: [[ADD1:%.+]] = "onnx.Add"([[MUL1]], [[EXPANDED_MASK]])
  // CHECK: "onnx.Softmax"([[ADD1]])
  // Verify subsequent layers reuse the expanded mask
  // CHECK: [[MATMUL1:%.+]] = "onnx.MatMul"(%arg0, %arg1)
  // CHECK: [[MUL2:%.+]] = "onnx.Mul"([[MATMUL1]], %arg2)
  // CHECK: [[ADD2:%.+]] = "onnx.Add"([[MUL2]], [[EXPANDED_MASK]])
}

// -----

// Test Case 5: Static dimensions - should use constants instead of Dim
// CHECK-LABEL: func.func @test_static_dimensions
func.func @test_static_dimensions(
    %arg0: tensor<1x12x128x64xf32>,
    %arg1: tensor<1x12x64x128xf32>,
    %scale: tensor<f32>,
    %mask: tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32> {
  
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %1 = "onnx.Mul"(%0, %scale) : (tensor<1x12x128x128xf32>, tensor<f32>) -> tensor<1x12x128x128xf32>
  %2 = "onnx.Add"(%1, %mask) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %3 = "onnx.Softmax"(%2) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  %4 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %5 = "onnx.Mul"(%4, %scale) : (tensor<1x12x128x128xf32>, tensor<f32>) -> tensor<1x12x128x128xf32>
  %6 = "onnx.Add"(%5, %mask) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %7 = "onnx.Softmax"(%6) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  %8 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %9 = "onnx.Mul"(%8, %scale) : (tensor<1x12x128x128xf32>, tensor<f32>) -> tensor<1x12x128x128xf32>
  %10 = "onnx.Add"(%9, %mask) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %11 = "onnx.Softmax"(%10) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  %12 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %13 = "onnx.Mul"(%12, %scale) : (tensor<1x12x128x128xf32>, tensor<f32>) -> tensor<1x12x128x128xf32>
  %14 = "onnx.Add"(%13, %mask) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %15 = "onnx.Softmax"(%14) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  return %15 : tensor<1x12x128x128xf32>
  
  // CHECK: onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: onnx.Constant dense<12> : tensor<1xi64>
  // CHECK: onnx.Constant dense<128> : tensor<1xi64>
  // CHECK: onnx.Constant dense<128> : tensor<1xi64>
  // CHECK-NOT: "onnx.Dim"
  // CHECK: [[EXPANDED_SCALE:%.+]] = "onnx.Expand"(%arg2
  // CHECK: [[EXPANDED_MASK:%.+]] = "onnx.Expand"(%arg3
}
