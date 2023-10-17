// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_unsqueeze(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_unsqueeze
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<1x10x10x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [1, 10, 10, 1] : tensor<4xindex>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<10x10xf32>, tensor<4xindex>) -> tensor<1x10x10x1xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x10x10x1xf32>
// CHECK:         }

// -----

func.func @test_unsqueeze_negative_axis(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<1xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_unsqueeze_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> tensor<16x32x1x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [16, 32, 1, 64] : tensor<4xindex>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<16x32x64xf32>, tensor<4xindex>) -> tensor<16x32x1x64xf32>
// CHECK:           return [[VAR_1_]] : tensor<16x32x1x64xf32>
// CHECK:         }

// -----

func.func @test_unsqueeze_mix(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_unsqueeze_mix
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> tensor<16x1x32x1x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [16, 1, 32, 1, 64] : tensor<5xindex>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.dynamic_reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<16x32x64xf32>, tensor<5xindex>) -> tensor<16x1x32x1x64xf32>
// CHECK:           return [[VAR_1_]] : tensor<16x1x32x1x64xf32>
// CHECK:         }
