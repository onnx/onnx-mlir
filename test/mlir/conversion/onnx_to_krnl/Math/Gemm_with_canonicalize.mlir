// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// GEMM with everything constant
func.func @test_gemm(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x10xf32>, [[PARAM_1_:%.+]]: memref<5x10xf32>, [[PARAM_2_:%.+]]: memref<10xf32>) -> memref<10x10xf32> {
}

// -----

// Gemm with all dimensions dynamic
func.func @test_gemm_all_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_all_dyn
}

// -----

// A[10, *] * B[*, 10] result in constant size output but dyn reduction.
func.func @test_gemm_k_dyn(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<?x10xf32>, tensor<?x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_k_dyn
}

// -----

// Broadcast bias C is dym, so we don't know if its 1 -> broadcast or 10. Dyn test for that.
func.func @test_gemm_c_dyn(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_c_dyn
}

