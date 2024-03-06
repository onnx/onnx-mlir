// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo --canonicalize %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

func.func @test_dim_1(%arg0 : tensor<5x?x1x32xf32>) -> tensor<1xi64> {
  %1 = "onnx.Dim"(%arg0) { axis = 1 : si64} : (tensor<5x?x1x32xf32>)  -> tensor<1xi64>
  return %1 : tensor<1xi64>
}
// CHECK-LABEL:  func.func @test_dim_1
// CHECK-SAME: ([[PARAM:%.+]]: tensor<5x?x1x32xf32>) -> tensor<1xi64> {
// CHECK-NEXT: [[CONST_1:%.+]] = arith.constant 1 : index
// CHECK-NEXT: [[SHAPE:%.+]] = shape.shape_of [[PARAM]] : tensor<5x?x1x32xf32> -> tensor<4xindex>
// CHECK-NEXT: [[DIM:%.+]] = shape.get_extent [[SHAPE]], [[CONST_1]] : tensor<4xindex>, index -> index
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

// -----

func.func @test_dim_invalid_1(%arg0 : tensor<5x7xf32>) -> tensor<1xi64> {
  // expected-error @+1 {{attribute "axis" value is 3, accepted range is [0, 1].}}
  %1 = "onnx.Dim"(%arg0) { axis = 3 : si64} : (tensor<5x7xf32>)  -> tensor<1xi64>
  return %1 : tensor<1xi64>
}

// -----

func.func @test_dim_invalid_2(%arg0 : tensor<*xf32>) -> tensor<1xi64> {
  // expected-error @+1 {{input must have shape and rank.}}
  %1 = "onnx.Dim"(%arg0) { axis = 0 : si64} : (tensor<*xf32>)  -> tensor<1xi64>
  return %1 : tensor<1xi64>
}
