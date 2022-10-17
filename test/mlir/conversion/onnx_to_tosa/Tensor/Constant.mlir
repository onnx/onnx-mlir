// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_scalar_attr() -> tensor<f32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
  "func.return"(%1) : (tensor<f32>) -> ()
// CHECK-LABEL: @test_scalar_attr() -> tensor<f32>
// CHECK-DAG: "tosa.const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: "tosa.const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
}

func.func @test_single_value_attr() -> tensor<1xf32> {
  %0 = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %1 = "onnx.Constant"() {value = dense<[2.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  "func.return"(%1) : (tensor<1xf32>) -> ()
// CHECK-LABEL: @test_single_value_attr() -> tensor<1xf32>
// CHECK-DAG:  "tosa.const"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG:  "tosa.const"() {value = dense<2.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
}

func.func @test_splat_attr() -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<3xf32>} : () -> tensor<3xf32>
  "func.return"(%1) : (tensor<3xf32>) -> ()
// CHECK-LABEL: @test_splat_attr() -> tensor<3xf32>
// CHECK-DAG:  "tosa.const"() {value = dense<1.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-DAG:  "tosa.const"() {value = dense<2.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
}

func.func @test_splat_nonsplat_attrs() -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  "func.return"(%1) : (tensor<3xf32>) -> ()
// CHECK-LABEL: @test_splat_nonsplat_attrs() -> tensor<3xf32>
// CHECK-DAG:  "tosa.const"() {value = dense<1.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-DAG:  "tosa.const"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
}


func.func @test_sparse_value_attr() -> tensor<3xf32> {
  %0 = "onnx.Constant"() {sparse_value = sparse<0, 1.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Constant"() {sparse_value = sparse<[[0, 1]], 2.000000e+00> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3xf32>) -> ()
// CHECK-LABEL: @test_sparse_value_attr() -> tensor<3xf32>
// CHECK-DAG: "tosa.const"() {value = sparse<0, 1.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-DAG: "tosa.const"() {value = sparse<{{\[}}[0, 1{{\]}}], 2.000000e+00> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
}

func.func @test_int_attr() -> tensor<i64> {
  %0 = "onnx.Constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
  %1 = "onnx.Constant"() {value = dense<3> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "onnx.Constant"() {value = dense<[-1, -1]> : tensor<2xi64>} : () -> tensor<2xi64>
  "func.return"(%0) : (tensor<i64>) -> ()
// CHECK-LABEL: @test_int_attr() -> tensor<i64>
// CHECK-DAG: "tosa.const"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
// CHECK-DAG: "tosa.const"() {value = dense<3> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG: "tosa.const"() {value = dense<-1> : tensor<2xi64>} : () -> tensor<2xi64>
}