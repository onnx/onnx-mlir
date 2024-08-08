// RUN: onnx-mlir-opt --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_expand(%arg0: tensor<1x64x1x1xf32>) -> tensor<1x64x64x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, 64, 64, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<1x64x1x1xf32>, tensor<4xi64>) -> tensor<1x64x64x64xf32>
  return %1 : tensor<1x64x64x64xf32>
}

// CHECK-LABEL:  func.func @test_expand
// CHECK:         %[[VAL:.*]] = tosa.tile %{{.*}} {multiples = array<i64: 1, 1, 64, 64>} : (tensor<1x64x1x1xf32>) -> tensor<1x64x64x64xf32>
// CHECK-NEXT:    return %[[VAL]] : tensor<1x64x64x64xf32>

// -----

func.func @test_expand_splat(%arg0: tensor<1x64x1x1xf32>) -> tensor<64x64x64x64xf32> {
  %0 = "onnx.Constant"() {value = dense<64> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<1x64x1x1xf32>, tensor<4xi64>) -> tensor<64x64x64x64xf32>
  return %1 : tensor<64x64x64x64xf32>
}

// CHECK-LABEL:  func.func @test_expand_splat
// CHECK:         %[[VAL:.*]] = tosa.tile %{{.*}} {multiples = array<i64: 64, 1, 64, 64>} : (tensor<1x64x1x1xf32>) -> tensor<64x64x64x64xf32>
// CHECK-NEXT:    return %[[VAL]] : tensor<64x64x64x64xf32>

// -----

func.func @test_expand_new_dims_out(%arg0: tensor<1x64x1xf32>) -> tensor<64x64x64x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[64, 64, 64, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<1x64x1xf32>, tensor<4xi64>) -> tensor<64x64x64x64xf32>
  return %1 : tensor<64x64x64x64xf32>
}

// CHECK-LABEL:  func.func @test_expand_new_dims_out
// CHECK:         %[[RES:.*]] = tosa.reshape %{{.*}} {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<1x64x1xf32>) -> tensor<1x64x1x1xf32>
// CHECK-NEXT:    %[[TILE:.*]] = tosa.tile %[[RES]] {multiples = array<i64: 64, 1, 64, 64>} : (tensor<1x64x1x1xf32>) -> tensor<64x64x64x64xf32>
// CHECK-NEXT:    return %[[TILE]] : tensor<64x64x64x64xf32>

// -----

func.func @test_expand_new_dims_start(%arg0: tensor<256x256x16xf32>) -> tensor<1x512x256x16xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, 512, 256, 16]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<256x256x16xf32>, tensor<4xi64>) -> tensor<1x512x256x16xf32>
  return %1 : tensor<1x512x256x16xf32>
}

// CHECK-LABEL:  func.func @test_expand_new_dims_start
// CHECK:         %[[RES:.*]] = tosa.reshape %{{.*}} {new_shape = array<i64: 1, 256, 256, 16>} : (tensor<256x256x16xf32>) -> tensor<1x256x256x16xf32>
// CHECK-NEXT:    %[[TILE:.*]] = tosa.tile %[[RES]] {multiples = array<i64: 1, 2, 1, 1>} : (tensor<1x256x256x16xf32>) -> tensor<1x512x256x16xf32>
// CHECK-NEXT:    return %[[TILE]] : tensor<1x512x256x16xf32>

// -----

func.func @test_expand_new_dims_mix(%arg0: tensor<128x64xf32>) -> tensor<1x128x16x128x16xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, 128, 16, 128, 16]> : tensor<5xi64>} : () -> tensor<5xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<128x64xf32>, tensor<5xi64>) -> tensor<1x128x16x128x16xf32>
  return %1 : tensor<1x128x16x128x16xf32>
}

// CHECK-LABEL:  func.func @test_expand_new_dims_mix
// CHECK:         %[[RES:.*]] = tosa.reshape %{{.*}} {new_shape = array<i64: 1, 128, 1, 64, 1>} : (tensor<128x64xf32>) -> tensor<1x128x1x64x1xf32>
// CHECK-NEXT:    %[[TILE:.*]] = tosa.tile %[[RES]] {multiples = array<i64: 1, 1, 16, 2, 16>} : (tensor<1x128x1x64x1xf32>) -> tensor<1x128x16x128x16xf32>
// CHECK-NEXT:    return %[[TILE]] : tensor<1x128x16x128x16xf32>

// -----

func.func @test_expand_no_tile(%arg0: tensor<128x16xf32>) -> tensor<1x1x128x16xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, 1, 128, 16]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<128x16xf32>, tensor<4xi64>) -> tensor<1x1x128x16xf32>
  return %1 : tensor<1x1x128x16xf32>
}

// CHECK-LABEL:  func.func @test_expand_no_tile
// CHECK:         %[[RES:.*]] = tosa.reshape %{{.*}} {new_shape = array<i64: 1, 1, 128, 16>} : (tensor<128x16xf32>) -> tensor<1x1x128x16xf32>
// CHECK-NEXT:    %[[TILE:.*]] = tosa.tile %[[RES]] {multiples = array<i64: 1, 1, 1, 1>} : (tensor<1x1x128x16xf32>) -> tensor<1x1x128x16xf32>
// CHECK-NEXT:    return %[[TILE]] : tensor<1x1x128x16xf32>

// -----

func.func @test_expand_no_legalization(%arg0: tensor<1x64x1x1xf32>, %arg1: tensor<4xi64>) -> tensor<1x64x64x64xf32> {
  %0 = "onnx.Expand"(%arg0, %arg1) : (tensor<1x64x1x1xf32>, tensor<4xi64>) -> tensor<1x64x64x64xf32>
  return %0 : tensor<1x64x64x64xf32>
}
// CHECK-LABEL:  func @test_expand_no_legalization
// CHECK: onnx.Expand