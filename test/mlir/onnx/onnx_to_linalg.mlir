// RUN: onnx-mlir-opt --convert-onnx-to-linalg -verify-diagnostics -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @test_lowering(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<16x128xf32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: tensor<128x32xf32>) {
// CHECK:           %[[VAL_2:.*]] = alloc() : memref<16x32xf32>
// CHECK:           linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : tensor<16x128xf32>, tensor<128x32xf32>) outs(%[[VAL_2]] : memref<16x32xf32>)
// CHECK:           dealloc %[[VAL_2]] : memref<16x32xf32>
// CHECK:           return
// CHECK:         }
func @test_lowering(%arg0: tensor<16x128xf32>, %arg1: tensor<128x32xf32>) -> () {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x128xf32>, tensor<128x32xf32>) -> tensor<16x32xf32>
  return
}

// -----

func @test_invalid(%arg0: tensor<16x?xf32>, %arg1: tensor<?x32xf32>) -> () {
// expected-warning@below {{This operation takes tensors with unsupported by current target sizes}}
// expected-error@below {{failed to legalize operation 'onnx.MatMul' that was explicitly marked illegal}}
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?xf32>, tensor<?x32xf32>) -> tensor<16x32xf32>
  return
}

// -----

func @test_invalid2(%arg0: tensor<16x33xf32>, %arg1: tensor<33x32xf32>) -> () {
// expected-warning@below {{This operation takes tensors with unsupported by current target sizes}}
// expected-error@below {{failed to legalize operation 'onnx.MatMul' that was explicitly marked illegal}}
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x33xf32>, tensor<33x32xf32>) -> tensor<16x32xf32>
  return
}

// -----

func @test_invalid3(%arg0: tensor<?x32xf32>, %arg1: tensor<32x32xf32>) -> () {
// expected-warning@below {{This operation produces unsupported by current target dynamically sized tensor}}
// expected-error@below {{failed to legalize operation 'onnx.MatMul' that was explicitly marked illegal}}
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x32xf32>, tensor<32x32xf32>) -> tensor<?x32xf32>
  return
}
