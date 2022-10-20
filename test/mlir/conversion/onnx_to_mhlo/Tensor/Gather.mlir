// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_gather_axis0(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()
// CHECK:  func.func @test_gather_axis0(%arg0: tensor<3x2xf32>) -> tensor<2x2x2xf32> {
// CHECK{LITERAL}:    %0 = mhlo.constant dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
// CHECK:    %1 = "mhlo.torch_index_select"(%arg0, %0) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
// CHECK:    return %1 : tensor<2x2x2xf32>
}

func.func @test_gather_axis0neg(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()
// CHECK-LABEL: func.func @test_gather_axis0neg
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<3x2xf32>) -> tensor<2x2x2xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant
// CHECK-SAME{LITERAL}: dense<[[3, 2], [4, 5]]> : tensor<2x2xi64>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant
// CHECK-SAME{LITERAL}: dense<[[false, true], [false, false]]> : tensor<2x2xi1>
// CHECK-DAG:    [[VAR_2_:%.+]] = mhlo.constant
// CHECK-SAME{LITERAL}: dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>
// CHECK-NEXT:   [[VAR_3_:%.+]] = "mhlo.select"([[VAR_1_]], [[VAR_0_]], [[VAR_2_]]) : (tensor<2x2xi1>, tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:   [[VAR_4_:%.+]] = "mhlo.torch_index_select"([[PARAM_0_]], [[VAR_3_]]) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
}

func.func @test_gather_axis1(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 2]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()
// CHECK{LITERAL}:   %0 = mhlo.constant dense<[[0, 2]]> : tensor<1x2xi64>
// CHECK:   %1 = "mhlo.torch_index_select"(%arg0, %0) {batch_dims = 0 : i64, dim = 1 : i64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
}