// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

// -----

// COM: Test removal of identity UpsampleAndPad (all strides=1, all pads=0).
func.func @test_upsample_and_pad_identity(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [1, 1], pads = [0, 0, 0, 0]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>
  // CHECK-LABEL: func.func @test_upsample_and_pad_identity
  // CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  // CHECK-NOT: onnx.UpsampleAndPad
  // CHECK: return [[ARG0]] : tensor<2x3x4x5xf32>
}

// -----

// COM: Test conversion of UpsampleAndPad to Pad when all strides=1 but pads are non-zero.
func.func @test_upsample_and_pad_to_pad(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x6x7xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [1, 1], pads = [1, 1, 1, 1]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x6x7xf32>
  return %0 : tensor<2x3x6x7xf32>
  // CHECK-LABEL: func.func @test_upsample_and_pad_to_pad
  // CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x6x7xf32>
  // CHECK-NOT: onnx.UpsampleAndPad
  // CHECK: [[PADS:%.+]] = onnx.Constant dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>
  // CHECK: [[RES:%.+]] = "onnx.Pad"([[ARG0]], [[PADS]], %{{.*}}, %{{.*}}) <{mode = "constant"}> : (tensor<2x3x4x5xf32>, tensor<8xi64>, none, none) -> tensor<2x3x6x7xf32>
  // CHECK: return [[RES]] : tensor<2x3x6x7xf32>
}

// -----

// COM: Test that UpsampleAndPad with non-unit strides is not canonicalized.

func.func @test_upsample_and_pad_no_canonicalization(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x9x11xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 2], pads = [1, 1, 1, 1]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x9x11xf32>
  return %0 : tensor<2x3x9x11xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_upsample_and_pad_no_canonicalization
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x9x11xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.UpsampleAndPad"([[PARAM_0_]]) <{pads = [1, 1, 1, 1], strides = [2, 2]}> : (tensor<2x3x4x5xf32>) -> tensor<2x3x9x11xf32>
// CHECK:           return [[VAR_0_]] : tensor<2x3x9x11xf32>
// CHECK:         }
}

// -----

// COM: Test identity with 3D innermost dimensions.
func.func @test_upsample_and_pad_identity_3d(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [1, 1, 1], pads = [0, 0, 0, 0, 0, 0]} : (tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
  return %0 : tensor<2x3x4x5x6xf32>
  // CHECK-LABEL: func.func @test_upsample_and_pad_identity_3d
  // CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
  // CHECK-NOT: onnx.UpsampleAndPad
  // CHECK: return [[ARG0]] : tensor<2x3x4x5x6xf32>
}

// -----

// COM: Test conversion to Pad with 3D innermost dimensions.
func.func @test_upsample_and_pad_to_pad_3d(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x6x7x8xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [1, 1, 1], pads = [1, 1, 1, 1, 1, 1]} : (tensor<2x3x4x5x6xf32>) -> tensor<2x3x6x7x8xf32>
  return %0 : tensor<2x3x6x7x8xf32>
  // CHECK-LABEL: func.func @test_upsample_and_pad_to_pad_3d
  // CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4x5x6xf32>) -> tensor<2x3x6x7x8xf32>
  // CHECK-NOT: onnx.UpsampleAndPad
  // CHECK: [[PADS:%.+]] = onnx.Constant dense<[0, 0, 1, 1, 1, 0, 0, 1, 1, 1]> : tensor<10xi64>
  // CHECK: [[RES:%.+]] = "onnx.Pad"([[ARG0]], [[PADS]], %{{.*}}, %{{.*}}) <{mode = "constant"}> : (tensor<2x3x4x5x6xf32>, tensor<10xi64>, none, none) -> tensor<2x3x6x7x8xf32>
  // CHECK: return [[RES]] : tensor<2x3x6x7x8xf32>
}

// -----

// COM: Test identity with missing strides attribute (defaults to all ones).
func.func @test_upsample_and_pad_identity_missing_strides(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {pads = [0, 0, 0, 0]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>
  // CHECK-LABEL: func.func @test_upsample_and_pad_identity_missing_strides
  // CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  // CHECK-NOT: onnx.UpsampleAndPad
  // CHECK: return [[ARG0]] : tensor<2x3x4x5xf32>
}

// -----

// COM: Test identity with missing pads attribute (defaults to all zeros).
func.func @test_upsample_and_pad_identity_missing_pads(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [1, 1]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>
  // CHECK-LABEL: func.func @test_upsample_and_pad_identity_missing_pads
  // CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  // CHECK-NOT: onnx.UpsampleAndPad
  // CHECK: return [[ARG0]] : tensor<2x3x4x5xf32>
}
