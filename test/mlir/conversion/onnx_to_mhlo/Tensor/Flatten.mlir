// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --lower-affine %s --canonicalize -split-input-file | FileCheck %s

// Test normal case
func.func @test_flatten(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_flatten
// CHECK: %0 = mhlo.reshape %arg0 : (tensor<5x5x1x32xf32>) -> tensor<25x32xf32>
}

// -----

// Test when axis is negative
func.func @test_flatten_negative_axis(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = -2 : si64} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_flatten_negative_axis
// CHECK: %0 = mhlo.reshape %arg0 : (tensor<5x5x1x32xf32>) -> tensor<25x32xf32>
}

// -----

// Test when axis is not set
func.func @test_flatten_with_default_axis(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Flatten"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_flatten_with_default_axis
// CHECK: %0 = mhlo.reshape %arg0 : (tensor<5x5x1x32xf32>) -> tensor<5x160xf32>
}

// -----

func.func @test_flatten1(%arg0 : tensor<2x?x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x?x4xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_flatten1
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<2x?x4xf32>) -> tensor<?x4xf32> {
// CHECK-DAG:    [[C4:%.+]] = arith.constant 4 : index
// CHECK-DAG:    [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
// CHECK-NEXT:    %0 = shape.shape_of [[PARAM_0_]] : tensor<2x?x4xf32> -> tensor<3xindex>
// CHECK-NEXT:    %1 = shape.get_extent %0, [[C1]] : tensor<3xindex>, index -> index
// CHECK-NEXT:    %2 = arith.muli %1, [[C2]] : index
// CHECK-NEXT:    %3 = shape.from_extents %2, [[C4]] : index, index
// CHECK-NEXT:    %4 = shape.to_extent_tensor %3 : !shape.shape -> tensor<2xindex>
// CHECK-NEXT:    %5 = mhlo.dynamic_reshape [[PARAM_0_]], %4 : (tensor<2x?x4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
}
