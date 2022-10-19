// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_float_attr() -> (tensor<3xf32>,tensor<3xf32>) {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  "func.return"(%0,%1) : (tensor<3xf32>, tensor<3xf32>) -> ()
// CHECK-LABEL: @test_float_attr() ->  (tensor<3xf32>, tensor<3xf32>)
// CHECK-DAG:  "tosa.const"() {value = dense<1.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-DAG:  "tosa.const"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
}


func.func @test_int_attr() -> tensor<i8> {
  %0 = "onnx.Constant"() {value = dense<3> : tensor<i8>} : () -> tensor<i8>
  %1 = "onnx.Constant"() {value = dense<3> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "onnx.Constant"() {value = dense<[-1, -1]> : tensor<2xi8>} : () -> tensor<2xi8>
  "func.return"(%0) : (tensor<i8>) -> ()
// CHECK-LABEL: @test_int_attr() -> tensor<i8>
// CHECK-DAG: "tosa.const"() {value = dense<3> : tensor<i8>} : () -> tensor<i8>
// CHECK-DAG: "tosa.const"() {value = dense<3> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG: "tosa.const"() {value = dense<-1> : tensor<2xi8>} : () -> tensor<2xi8>
}