// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// onnx.Pow with integer exponent is not supported in TOSA.
// This test checks that the conversion does not fail but, instead, it keeps the
// original version of the op.

func.func @test_less_broadcast(%arg0: tensor<5xi32>, %arg1: tensor<5xi32>) -> tensor<*xi32> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<5xi32>, tensor<5xi32>) -> tensor<*xi32>
  onnx.Return %0 : tensor<*xi32>

// CHECK:   test_less_broadcast
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<5xi32>, %[[ARG_1:.*]]: tensor<5xi32>) -> tensor<5xi32>
// CHECK-NEXT: %[[VAL_0:.*]] = "onnx.Pow"(%[[ARG_0]], %[[ARG_1]]) : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
// CHECK-NEXT: onnx.Return %[[VAL_0]] : tensor<5xi32>

}
