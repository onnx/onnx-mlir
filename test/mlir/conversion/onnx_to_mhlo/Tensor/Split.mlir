// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_split_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 0 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_split_equal
// CHECK:         %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[8, 32, 64]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<16x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK:         %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[16, 32, 64]> : tensor<3xi64>, start_indices = dense<[8, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<16x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK:         return %0, %1 : tensor<8x32x64xf32>, tensor<8x32x64xf32>
}

func.func @test_split_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) { axis = 1 : si64} : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_split_variable
// CHECK:         %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[16, 2, 64]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<16x32x64xf32>) -> tensor<16x2x64xf32>
// CHECK:         %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[16, 32, 64]> : tensor<3xi64>, start_indices = dense<[0, 2, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<16x32x64xf32>) -> tensor<16x30x64xf32>
// CHECK:         return %0, %1 : tensor<16x2x64xf32>, tensor<16x30x64xf32>
}

func.func @test_split_1(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 1 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_split_1
// CHECK:         %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[16, 16, 64]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<16x32x64xf32>) -> tensor<16x16x64xf32>
// CHECK:         return %0 : tensor<16x16x64xf32>
}

// -----

func.func @test_split_2(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = -2 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_split_2
// CHECK:         %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[16, 16, 64]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<16x32x64xf32>) -> tensor<16x16x64xf32>
// CHECK:         return %0 : tensor<16x16x64xf32>
}

// -----

func.func @test_split_3(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) {axis = 1 : si64} : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_split_3
// CHECK:         %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[16, 2, 64]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<16x32x64xf32>) -> tensor<16x2x64xf32>
// CHECK:         return %0 : tensor<16x2x64xf32>
}

// -----