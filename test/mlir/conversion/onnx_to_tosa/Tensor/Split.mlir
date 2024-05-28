// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_split_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
    %cst = "onnx.NoValue"() {value} : () -> none
    %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 0 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
    return %0, %1 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: func.func @test_split_equal
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> (tensor<8x32x64xf32>, tensor<8x32x64xf32>) {
// CHECK-DAG:      [[VAR_0_:%.+]] = tosa.slice [[PARAM_0_]] {size = array<i64: 8, 32, 64>, start = array<i64: 0, 0, 0>} : (tensor<16x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK-DAG:      [[VAR_1_:%.+]] = tosa.slice [[PARAM_0_]] {size = array<i64: 8, 32, 64>, start = array<i64: 8, 0, 0>} : (tensor<16x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK:           return [[VAR_0_]], [[VAR_1_]] : tensor<8x32x64xf32>, tensor<8x32x64xf32>


// -----

func.func @test_split_variable(%arg0 : tensor<16x32x64xf16>) -> (tensor<*xf16>, tensor<*xf16>) {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) { axis = 1 : si64} : (tensor<16x32x64xf16>, tensor<2xi64>) -> (tensor<*xf16>, tensor<*xf16>)
  "func.return"(%0, %1) : (tensor<*xf16>, tensor<*xf16>) -> ()
}

// CHECK-LABEL:  func.func @test_split_variable
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf16>) -> (tensor<16x2x64xf16>, tensor<16x30x64xf16>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.slice [[PARAM_0_]] {size = array<i64: 16, 2, 64>, start = array<i64: 0, 0, 0>} : (tensor<16x32x64xf16>) -> tensor<16x2x64xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.slice [[PARAM_0_]] {size = array<i64: 16, 30, 64>, start = array<i64: 0, 2, 0>} : (tensor<16x32x64xf16>) -> tensor<16x30x64xf16>
// CHECK:           return [[VAR_0_]], [[VAR_1_]] : tensor<16x2x64xf16>, tensor<16x30x64xf16>

// -----

func.func @test_split_1(%arg0 : tensor<16x32x64xi32>) -> tensor<*xi32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 1 : si64} : (tensor<16x32x64xi32>, none) -> (tensor<*xi32>, tensor<*xi32>)
  "func.return"(%0) : (tensor<*xi32>) -> ()
}

// CHECK-LABEL:  func.func @test_split_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xi32>) -> tensor<16x16x64xi32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.slice [[PARAM_0_]] {size = array<i64: 16, 16, 64>, start = array<i64: 0, 0, 0>} : (tensor<16x32x64xi32>) -> tensor<16x16x64xi32>
// CHECK:           return [[VAR_0_]] : tensor<16x16x64xi32>



// -----

func.func @test_split_2(%arg0 : tensor<16x32x64xbf16>) -> tensor<*xbf16> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = -2 : si64} : (tensor<16x32x64xbf16>, none) -> (tensor<*xbf16>, tensor<*xbf16>)
  "func.return"(%0) : (tensor<*xbf16>) -> ()
}

// CHECK-LABEL:  func.func @test_split_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xbf16>) -> tensor<16x16x64xbf16> {
// CHECK:           [[VAR_0_:%.+]] = tosa.slice [[PARAM_0_]] {size = array<i64: 16, 16, 64>, start = array<i64: 0, 0, 0>} : (tensor<16x32x64xbf16>) -> tensor<16x16x64xbf16>
// CHECK:           return [[VAR_0_]] : tensor<16x16x64xbf16>


// -----

func.func @test_split_3(%arg0 : tensor<16x32x64xi16>) -> tensor<*xi16> {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) {axis = 1 : si64} : (tensor<16x32x64xi16>, tensor<2xi64>) -> (tensor<*xi16>, tensor<*xi16>)
  "func.return"(%0) : (tensor<*xi16>) -> ()
}

// CHECK-LABEL:  func.func @test_split_3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xi16>) -> tensor<16x2x64xi16> {
// CHECK:           [[VAR_0_:%.+]] = tosa.slice [[PARAM_0_]] {size = array<i64: 16, 2, 64>, start = array<i64: 0, 0, 0>} : (tensor<16x32x64xi16>) -> tensor<16x2x64xi16>
// CHECK:           return [[VAR_0_]] : tensor<16x2x64xi16>



