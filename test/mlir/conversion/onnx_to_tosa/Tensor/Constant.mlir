// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_float_broadcast() -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL: @test_float_broadcast() ->  tensor<3xf32>
// CHECK:       "tosa.const"() <{value = dense<1.000000e+00> : tensor<3xf32>}> : () -> tensor<3xf32>
}

// -----

func.func @test_float_dense() -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL: @test_float_dense() -> tensor<3xf32>
// CHECK:       "tosa.const"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
}

// -----

func.func @test_int_single() -> tensor<1xi8> {
  %0 = "onnx.Constant"() {value = dense<3> : tensor<1xi8>} : () -> tensor<1xi8>
  return %0 : tensor<1xi8>
// CHECK-LABEL: @test_int_single() -> tensor<1xi8>
// CHECK:      "tosa.const"() <{value = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
}

// -----

func.func @test_int_broadcast() -> tensor<4xi32> {
  %0 = "onnx.Constant"() {value = dense<3> : tensor<4xi32>} : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
// CHECK-LABEL: @test_int_broadcast() -> tensor<4xi32>
// CHECK:       "tosa.const"() <{value = dense<3> : tensor<4xi32>}> : () -> tensor<4xi32>
}

// -----

func.func @test_int_dense() -> tensor<2xi8> {
  %0 = "onnx.Constant"() {value = dense<[-1, -2]> : tensor<2xi8>} : () -> tensor<2xi8>
  return %0 : tensor<2xi8>
// CHECK-LABEL: @test_int_dense() -> tensor<2xi8>
// CHECK:       "tosa.const"() <{value = dense<[-1, -2]> : tensor<2xi8>}> : () -> tensor<2xi8>
}

// -----

func.func @test_bool_single() -> tensor<i1> {
  %0 = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  return %0 : tensor<i1>
// CHECK-LABEL: @test_bool_single() -> tensor<i1>
// CHECK:      "tosa.const"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
}

// -----

func.func @test_bool_broadcast() -> tensor<4xi1> {
  %0 = "onnx.Constant"() {value = dense<false> : tensor<4xi1>} : () -> tensor<4xi1>
  return %0 : tensor<4xi1>
// CHECK-LABEL: @test_bool_broadcast() -> tensor<4xi1>
// CHECK:       "tosa.const"() <{value = dense<false> : tensor<4xi1>}> : () -> tensor<4xi1>
}

// -----

func.func @test_bool_dense() -> tensor<2xi1> {
  %0 = "onnx.Constant"() {value = dense<[true, false]> : tensor<2xi1>} : () -> tensor<2xi1>
  return %0 : tensor<2xi1>
// CHECK-LABEL: @test_bool_dense() -> tensor<2xi1>
// CHECK:       "tosa.const"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
}
