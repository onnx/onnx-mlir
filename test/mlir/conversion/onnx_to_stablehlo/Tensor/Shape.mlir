// RUN: onnx-mlir-opt --canonicalize --convert-onnx-to-stablehlo %s -split-input-file | FileCheck %s

func.func @test_shape1(%arg0 : tensor<2x4x8x16xf32>) -> tensor<4xi64> {
  %0 = "onnx.Shape"(%arg0) : (tensor<2x4x8x16xf32>) -> tensor<4xi64>
  return %0 : tensor<4xi64>
// CHECK: func.func @test_shape1(%arg0: tensor<2x4x8x16xf32>) -> tensor<4xi64> {
// CHECK:    [[CST_:%.+]] = stablehlo.constant dense<[2, 4, 8, 16]> : tensor<4xi64>
// CHECK:    return [[CST_]] : tensor<4xi64>
}

// -----

func.func @test_shape2(%arg0 : tensor<?x4x8x16xf32>) -> tensor<4xi64> {
  %0 = "onnx.Shape"(%arg0) : (tensor<?x4x8x16xf32>) -> tensor<4xi64>
  return %0 : tensor<4xi64>
// CHECK: func.func @test_shape2(%arg0: tensor<?x4x8x16xf32>) -> tensor<4xi64> {
// CHECK:    [[CST_:%.+]] = shape.shape_of %arg0 : tensor<?x4x8x16xf32> -> tensor<4xindex>
// CHECK:    [[CAST_:%.+]] = arith.index_cast [[CST_]] : tensor<4xindex> to tensor<4xi64>
// CHECK:    return [[CAST_]] : tensor<4xi64>
}
