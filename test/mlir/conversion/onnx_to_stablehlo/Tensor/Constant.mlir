// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo %s -split-input-file | FileCheck %s

func.func @test_scalar_attr() -> tensor<f32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<f32> , tensor<f32>) -> tensor<f32>
  "func.return"(%2) : (tensor<f32>) -> ()
// CHECK-LABEL: @test_scalar_attr() -> tensor<f32>
// CHECK-DAG:    [[VAR_0_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:    [[VAR_1_:%.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
}

// -----

func.func @test_single_value_attr() -> tensor<1xf32> {
  %0 = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %1 = "onnx.Constant"() {value = dense<[2.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1xf32> , tensor<1xf32>) -> tensor<1xf32>
  "func.return"(%2) : (tensor<1xf32>) -> ()
// CHECK-LABEL: @test_single_value_attr() -> tensor<1xf32>
// CHECK-DAG:    [[VAR_0_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:    [[VAR_1_:%.+]] = stablehlo.constant dense<2.000000e+00> : tensor<1xf32>
}

// -----

func.func @test_splat_attr() -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%2) : (tensor<3xf32>) -> ()
// CHECK-LABEL: @test_splat_attr() -> tensor<3xf32>
// CHECK-DAG:    [[VAR_0_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-DAG:    [[VAR_1_:%.+]] = stablehlo.constant dense<2.000000e+00> : tensor<3xf32>
}

// -----

func.func @test_splat_nonsplat_attrs() -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%2) : (tensor<3xf32>) -> ()
// CHECK-LABEL: @test_splat_nonsplat_attrs() -> tensor<3xf32>
// CHECK-DAG:    [[VAR_0_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-DAG:    [[VAR_1_:%.+]] = stablehlo.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>
}

// -----
