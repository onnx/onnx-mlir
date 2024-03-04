// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo --canonicalize %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

func.func @test_dim_1(%arg0 : tensor<5x?x1x32xf32>) -> tensor<1xi64> {
  %1 = "onnx.Dim"(%arg0) { axis = 1 : si64} : (tensor<5x?x1x32xf32>)  -> tensor<1xi64>
  return %1 : tensor<1xi64>
}
// CHECK-LABEL:  func.func @test_dim_1
// CHECK-SAME: ([[PARAM:%.+]]: tensor<5x?x1x32xf32>) -> tensor<1xi64> {
// CHECK-NEXT: [[CONST_1:%.+]] = arith.constant 1 : index
// CHECK-NEXT: [[DIM:%.+]] = tensor.dim [[PARAM]], [[CONST_1]] : tensor<5x?x1x32xf32>
// CHECK-NEXT: [[INDEX_CAST:%.+]] = arith.index_cast [[DIM]] : index to i64
// CHECK-NEXT: [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[INDEX_CAST]] : tensor<1xi64>
// CHECK-NEXT: return [[FROM_ELEMENTS]] : tensor<1xi64>
// CHECK: }

// -----

func.func @test_dim_2(%arg0 : tensor<5x7xf32>) -> tensor<1xi64> {
  %1 = "onnx.Dim"(%arg0) { axis = 0 : si64} : (tensor<5x7xf32>)  -> tensor<1xi64>
  return %1 : tensor<1xi64>
}

// CHECK-LABEL:  func.func @test_dim_2
// CHECK-SAME: ([[PARAM:%.+]]: tensor<5x7xf32>) -> tensor<1xi64> {
// CHECK-NEXT: [[CONST:%.+]] = arith.constant dense<5> : tensor<1xi64>
// CHECK-NEXT: return [[CONST]] : tensor<1xi64>
// CHECK: }
