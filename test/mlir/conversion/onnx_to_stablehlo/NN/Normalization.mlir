// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo %s -split-input-file | FileCheck %s

func.func @test_batch_normalization(%arg0 : tensor<1x3x10x10xf32>) -> tensor<1x3x10x10xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %2 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %3 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32} : (tensor<1x3x10x10xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3x10x10xf32>
  "func.return"(%4) : (tensor<1x3x10x10xf32>) -> ()
// CHECK-LABEL:  func @test_batch_normalization
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x10x10xf32>) -> tensor<1x3x10x10xf32> {
// CHECK-NEXT:    [[VAR_0_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    [[VAR_1_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    [[VAR_2_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    [[VAR_3_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    [[VAR_4_:%.+]] = "stablehlo.batch_norm_inference"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]]) <{epsilon = 1.00000007E-5 : f32, feature_index = 1 : i64}> : (tensor<1x3x10x10xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3x10x10xf32>
// CHECK-NEXT:   return [[VAR_4_]] : tensor<1x3x10x10xf32>
// CHECK-NEXT:   }
}
