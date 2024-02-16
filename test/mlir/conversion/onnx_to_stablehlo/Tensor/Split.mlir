// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_split_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 0 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_split_equal
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> (tensor<8x32x64xf32>, tensor<8x32x64xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.slice [[PARAM_0_]] [0:8, 0:32, 0:64] : (tensor<16x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.slice [[PARAM_0_]] [8:16, 0:32, 0:64] : (tensor<16x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK:           return [[VAR_0_]], [[VAR_1_]] : tensor<8x32x64xf32>, tensor<8x32x64xf32>
// CHECK:         }

// -----

func.func @test_split_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) { axis = 1 : si64} : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_split_variable
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> (tensor<16x2x64xf32>, tensor<16x30x64xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.slice [[PARAM_0_]] [0:16, 0:2, 0:64] : (tensor<16x32x64xf32>) -> tensor<16x2x64xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.slice [[PARAM_0_]] [0:16, 2:32, 0:64] : (tensor<16x32x64xf32>) -> tensor<16x30x64xf32>
// CHECK:           return [[VAR_0_]], [[VAR_1_]] : tensor<16x2x64xf32>, tensor<16x30x64xf32>
// CHECK:         }

// -----

func.func @test_split_1(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 1 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_split_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> tensor<16x16x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.slice [[PARAM_0_]] [0:16, 0:16, 0:64] : (tensor<16x32x64xf32>) -> tensor<16x16x64xf32>
// CHECK:           return [[VAR_0_]] : tensor<16x16x64xf32>
// CHECK:         }

// -----

func.func @test_split_2(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = -2 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_split_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> tensor<16x16x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.slice [[PARAM_0_]] [0:16, 0:16, 0:64] : (tensor<16x32x64xf32>) -> tensor<16x16x64xf32>
// CHECK:           return [[VAR_0_]] : tensor<16x16x64xf32>
// CHECK:         }

// -----

func.func @test_split_3(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) {axis = 1 : si64} : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_split_3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> tensor<16x2x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.slice [[PARAM_0_]] [0:16, 0:2, 0:64] : (tensor<16x32x64xf32>) -> tensor<16x2x64xf32>
// CHECK:           return [[VAR_0_]] : tensor<16x2x64xf32>
// CHECK:         }
