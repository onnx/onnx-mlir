// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_squeeze(%arg0 : tensor<1x16x1x64xf32>) -> tensor<16x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<1x16x1x64xf32>, tensor<2xi64>) -> tensor<16x64xf32>
  "func.return"(%1) : (tensor<16x64xf32>) -> ()
// CHECK-LABEL:  func @test_squeeze
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x16x1x64xf32>) -> tensor<16x64xf32> {
// CHECK:           [[SHAPE:%.+]] = tosa.const_shape {values = dense<[16, 64]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-NEXT:      [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE]] : (tensor<1x16x1x64xf32>, !tosa.shape<2>) -> tensor<16x64xf32>
// CHECK-NEXT:      return [[VAR_1_]] : tensor<16x64xf32>
}

// -----

func.func @test_squeeze_no_axes(%arg0 : tensor<1x16x1x64xf32>) -> tensor<16x64xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<1x16x1x64xf32>, none) -> tensor<16x64xf32>
  "func.return"(%1) : (tensor<16x64xf32>) -> ()
// CHECK-LABEL:  func @test_squeeze_no_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x16x1x64xf32>) -> tensor<16x64xf32> {
// CHECK:           [[SHAPE:%.+]] = tosa.const_shape {values = dense<[16, 64]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-NEXT:      [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE]] : (tensor<1x16x1x64xf32>, !tosa.shape<2>) -> tensor<16x64xf32>
// CHECK-NEXT:      return [[VAR_1_]] : tensor<16x64xf32>
}

// -----

func.func @test_unsqueeze(%arg0 : tensor<16x64xf32>) -> tensor<1x16x1x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x64xf32>, tensor<2xi64>) -> tensor<1x16x1x64xf32>
  "func.return"(%1) : (tensor<1x16x1x64xf32>) -> ()
// CHECK-LABEL:  func @test_unsqueeze
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x64xf32>) -> tensor<1x16x1x64xf32> {
// CHECK:           [[SHAPE:%.+]] = tosa.const_shape {values = dense<[1, 16, 1, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NEXT:      [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE]] : (tensor<16x64xf32>, !tosa.shape<4>) -> tensor<1x16x1x64xf32>
// CHECK-NEXT:      return [[VAR_1_]] : tensor<1x16x1x64xf32>
}

// -----

func.func @test_squeeze_dynamic(%arg0 : tensor<?x16x1x64xf32>) -> tensor<?x16x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<?x16x1x64xf32>, tensor<1xi64>) -> tensor<?x16x64xf32>
  "func.return"(%1) : (tensor<?x16x64xf32>) -> ()
// CHECK-LABEL:  func @test_squeeze_dynamic
// CHECK:           "onnx.Squeeze"
// CHECK-NOT:       tosa.reshape
}
