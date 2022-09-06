// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test the reshape op inference when constants are present.
//===----------------------------------------------------------------------===//

func.func @test_reshape_dynamic(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reshape_dynamic
  // CHECK: %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
}

// -----

func.func @test_reshape_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[5, 5, 16, 2]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reshape_1
  // CHECK: %1 = "mhlo.dynamic_reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
}

// -----

func.func @test_reshape_2(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 16, 2]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reshape_2
  // CHECK: %1 = "mhlo.dynamic_reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<*xf32>
}

// -----

func.func @test_reshape_3(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 0, 2]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reshape_3
  // CHECK: %1 = "mhlo.dynamic_reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<*xf32>
}

// -----