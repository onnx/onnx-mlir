// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --canonicalize %s -split-input-file | FileCheck %s

// -----

func.func @test_expand_with_arith_constant(%arg0 : tensor<2x1x6x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[7, 1, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_expand_with_arith_constant
// CHECK-NEXT: %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x1x6x1xf32>) -> tensor<2x7x6x5xf32>
}

// -----

func.func @test_expand_integer_tensor(%arg0 : tensor<2x1x6x1xi64>) -> tensor<*xi64> {
  %0 = "onnx.Constant"() {value = dense<[7, 1, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xi64>, tensor<3xi64>) -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
// CHECK-LABEL: func @test_expand_integer_tensor
// CHECK-NEXT: %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x1x6x1xi64>) -> tensor<2x7x6x5xi64>
}

// -----

func.func @test_expand_with_shape(%arg0 : tensor<2x1x6x1xf32>, %arg1: tensor<6x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Shape"(%arg1) : (tensor<6x2xf32>) -> tensor<*xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xf32>, tensor<*xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_expand_with_shape
// CHECK: [[BRDCST:%.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x1x6x1xf32>) -> tensor<2x1x6x2xf32>
}