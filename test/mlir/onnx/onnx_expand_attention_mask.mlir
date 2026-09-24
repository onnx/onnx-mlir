// RUN: onnx-mlir-opt --expand-attention-mask %s -split-input-file | FileCheck %s

// -----

// COM: Test case 1: Single attention layer - no expansion (< 8 threshold)
// CHECK-LABEL: @test_single_layer_no_expand
func.func @test_single_layer_no_expand(%arg0: tensor<1x12x?x64xf32>, 
                                       %arg1: tensor<1x12x64x?xf32>,
                                       %arg2: tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32> {
  // CHECK-NOT: onnx.Expand
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %2 = "onnx.Softmax"(%1) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  return %2 : tensor<1x12x?x?xf32>
}

// -----

// COM: Test case 2: 8 attention layers with same shape - expansion applied
// CHECK-LABEL: @test_8_layers_expand
func.func @test_8_layers_expand(%arg0: tensor<1x12x?x64xf32>, 
                                %arg1: tensor<1x12x64x?xf32>,
                                %arg2: tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32> {
  // CHECK: [[MATMUL0:%.+]] = "onnx.MatMul"(%arg0, %arg1)
  // CHECK: [[CONST1:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: [[CONST12:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
  // CHECK: [[DIM2:%.+]] = "onnx.Dim"([[MATMUL0]]) <{axis = 2 : si64}> : (tensor<1x12x?x?xf32>) -> tensor<1xi64>
  // CHECK: [[DIM3:%.+]] = "onnx.Dim"([[MATMUL0]]) <{axis = 3 : si64}> : (tensor<1x12x?x?xf32>) -> tensor<1xi64>
  // CHECK: [[SHAPE:%.+]] = "onnx.Concat"([[CONST1]], [[CONST12]], [[DIM2]], [[DIM3]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  // CHECK: [[EXPANDED:%.+]] = "onnx.Expand"(%arg2, [[SHAPE]]) : (tensor<1x1x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
  
  // Layer 1
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %2 = "onnx.Softmax"(%1) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 2
  %3 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %4 = "onnx.Add"(%3, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %5 = "onnx.Softmax"(%4) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 3
  %6 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %7 = "onnx.Add"(%6, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %8 = "onnx.Softmax"(%7) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 4
  %9 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %10 = "onnx.Add"(%9, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %11 = "onnx.Softmax"(%10) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 5
  %12 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %13 = "onnx.Add"(%12, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %14 = "onnx.Softmax"(%13) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 6
  %15 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %16 = "onnx.Add"(%15, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %17 = "onnx.Softmax"(%16) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 7
  %18 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %19 = "onnx.Add"(%18, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %20 = "onnx.Softmax"(%19) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  // Layer 8
  %21 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  %22 = "onnx.Add"(%21, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %23 = "onnx.Softmax"(%22) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  return %23 : tensor<1x12x?x?xf32>
}

// -----

// COM: Test case 3: 7 layers - no expansion (< 8 threshold)
// CHECK-LABEL: @test_7_layers_no_expand
func.func @test_7_layers_no_expand(%arg0: tensor<1x12x?x64xf32>, 
                                   %arg1: tensor<1x12x64x?xf32>,
                                   %arg2: tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32> {
  // CHECK-NOT: onnx.Expand
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %2 = "onnx.Softmax"(%1) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  %3 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %4 = "onnx.Add"(%3, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %5 = "onnx.Softmax"(%4) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  %6 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %7 = "onnx.Add"(%6, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %8 = "onnx.Softmax"(%7) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  %9 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %10 = "onnx.Add"(%9, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %11 = "onnx.Softmax"(%10) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  %12 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %13 = "onnx.Add"(%12, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %14 = "onnx.Softmax"(%13) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  %15 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %16 = "onnx.Add"(%15, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %17 = "onnx.Softmax"(%16) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  %18 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %19 = "onnx.Add"(%18, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  %20 = "onnx.Softmax"(%19) {axis = -1 : si64} : (tensor<1x12x?x?xf32>) -> tensor<1x12x?x?xf32>
  
  return %20 : tensor<1x12x?x?xf32>
}

// -----

// COM: Test case 4: Non-attention pattern - no expansion
// CHECK-LABEL: @test_non_attention_pattern
func.func @test_non_attention_pattern(%arg0: tensor<1x12x?x64xf32>, 
                                      %arg1: tensor<1x12x64x?xf32>,
                                      %arg2: tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32> {
  // CHECK-NOT: onnx.Expand
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x?x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x?x?xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) -> tensor<1x12x?x?xf32>
  // No Softmax - not an attention pattern
  return %1 : tensor<1x12x?x?xf32>
}

// -----

// COM: Test case 5: Static dimensions - expansion applied
// CHECK-LABEL: @test_static_dims_expand
func.func @test_static_dims_expand(%arg0: tensor<1x12x128x64xf32>, 
                                   %arg1: tensor<1x12x64x128xf32>,
                                   %arg2: tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32> {
  // CHECK: [[CONST1:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: [[CONST12:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
  // CHECK: [[CONST128_1:%.+]] = onnx.Constant dense<128> : tensor<1xi64>
  // CHECK: [[CONST128_2:%.+]] = onnx.Constant dense<128> : tensor<1xi64>
  // CHECK: [[SHAPE:%.+]] = "onnx.Concat"([[CONST1]], [[CONST12]], [[CONST128_1]], [[CONST128_2]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  // CHECK: [[EXPANDED:%.+]] = "onnx.Expand"(%arg2, [[SHAPE]]) : (tensor<1x1x128x128xf32>, tensor<4xi64>) -> tensor<1x12x128x128xf32>
  
  // CHECK: "onnx.MatMul"(%arg0, %arg1)
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x128x128xf32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  // CHECK: "onnx.Softmax"
  
  // CHECK: "onnx.MatMul"(%arg0, %arg1)
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x128x128xf32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  // CHECK: "onnx.Softmax"
  
  // CHECK: "onnx.MatMul"(%arg0, %arg1)
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x128x128xf32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  // CHECK: "onnx.Softmax"
  
  // CHECK: "onnx.MatMul"(%arg0, %arg1)
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x128x128xf32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  // CHECK: "onnx.Softmax"
  
  // CHECK: "onnx.MatMul"(%arg0, %arg1)
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x128x128xf32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  // CHECK: "onnx.Softmax"
  
  // CHECK: "onnx.MatMul"(%arg0, %arg1)
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x128x128xf32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  // CHECK: "onnx.Softmax"
  
  // CHECK: "onnx.MatMul"(%arg0, %arg1)
  // CHECK: "onnx.Add"({{%.+}}, [[EXPANDED]]) : (tensor<1x12x128x128xf32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  // CHECK: "onnx.Softmax"
  
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %2 = "onnx.Softmax"(%1) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  %3 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %4 = "onnx.Add"(%3, %arg2) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %5 = "onnx.Softmax"(%4) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  %6 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %7 = "onnx.Add"(%6, %arg2) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %8 = "onnx.Softmax"(%7) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  %9 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %10 = "onnx.Add"(%9, %arg2) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %11 = "onnx.Softmax"(%10) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  %12 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %13 = "onnx.Add"(%12, %arg2) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %14 = "onnx.Softmax"(%13) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  %15 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %16 = "onnx.Add"(%15, %arg2) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %17 = "onnx.Softmax"(%16) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  %18 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %19 = "onnx.Add"(%18, %arg2) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %20 = "onnx.Softmax"(%19) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  %21 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
  %22 = "onnx.Add"(%21, %arg2) : (tensor<1x12x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x12x128x128xf32>
  %23 = "onnx.Softmax"(%22) {axis = -1 : si64} : (tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
  
  return %23 : tensor<1x12x128x128xf32>
}
