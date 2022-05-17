// RUN: onnx-mlir-opt --maccel=NNPA --shape-inference --rewrite-onnx-for-zhigh --constprop-onnx %s -split-input-file | FileCheck %s

func @test_batchnorm_epsilon(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<2x3x4x5xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>

// CHECK-LABEL:  func @test_batchnorm_epsilon
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4x5xf32>, [[PARAM_1_:%.+]]: tensor<3xf32>, [[PARAM_2_:%.+]]: tensor<3xf32>, [[PARAM_3_:%.+]]: tensor<3xf32>, [[PARAM_4_:%.+]]: tensor<3xf32>) -> tensor<2x3x4x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<0.00999999977> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_4_]], [[VAR_0_]]) : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sqrt"([[VAR_1_]]) : (tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[PARAM_1_]], [[VAR_2_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[PARAM_3_]], [[VAR_3_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Sub"([[PARAM_2_]], [[VAR_4_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 2, 3, 1]} : (tensor<2x3x4x5xf32>) -> tensor<2x4x5x3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "zhigh.Stick"([[VAR_6_]]) {layout = "NHWC"} : (tensor<2x4x5x3xf32>) -> tensor<2x4x5x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_8_:%.+]] = "zhigh.Stick"([[VAR_3_]]) {layout = "1D"} : (tensor<3xf32>) -> tensor<3xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK-DAG:       [[VAR_9_:%.+]] = "zhigh.Stick"([[VAR_5_]]) {layout = "1D"} : (tensor<3xf32>) -> tensor<3xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK:           [[VAR_10_:%.+]] = "zhigh.BatchNorm"([[VAR_7_]], [[VAR_8_]], [[VAR_9_]]) : (tensor<2x4x5x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<3xf32, #zhigh.encoding<{dataLayout = "1D"}>>, tensor<3xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<2x4x5x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.Unstick"([[VAR_10_]]) : (tensor<2x4x5x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<2x4x5x3xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [0, 3, 1, 2]} : (tensor<2x4x5x3xf32>) -> tensor<2x3x4x5xf32>
// CHECK:           return [[VAR_12_]] : tensor<2x3x4x5xf32>
// CHECK:         }
}

// -----

func @test_batchnorm_5d_not_lowered(%arg0: tensor<2x3x4x5x6xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<2x3x4x5x6xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5x6xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5x6xf32>
  return %0 : tensor<2x3x4x5x6xf32>
  // CHECK-LABEL: test_batchnorm_5d_not_lowered
  // CHECK: "onnx.BatchNormalizationInferenceMode"
}

// -----

func @test_batchnorm_constprop(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32> {
  %0 = "onnx.Constant"() {value = dense<[0.15, 0.2]> : tensor<2xf32>} : () -> tensor<2xf32>
  %1 = "onnx.Constant"() {value = dense<[0.7, 0.8]> : tensor<2xf32>} : () -> tensor<2xf32>
  %2 = "onnx.Constant"() {value = dense<[0.5, 0.6]> : tensor<2xf32>} : () -> tensor<2xf32>
  %3 = "onnx.Constant"() {value = dense<[0.99001,3.99001]> : tensor<2xf32>} : () -> tensor<2xf32>
  %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 0.00999 : f32} : (tensor<1x2x3x3xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x3x3xf32>
  return %4 : tensor<1x2x3x3xf32>

// CHECK-LABEL:  func @test_batchnorm_constprop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<[1.500000e-01, 1.000000e-01]> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<[6.250000e-01, 7.400000e-01]> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 2, 3, 1]} : (tensor<1x2x3x3xf32>) -> tensor<1x3x3x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "NHWC"} : (tensor<1x3x3x2xf32>) -> tensor<1x3x3x2xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "1D"} : (tensor<2xf32>) -> tensor<2xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "1D"} : (tensor<2xf32>) -> tensor<2xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.BatchNorm"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]]) : (tensor<1x3x3x2xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2xf32, #zhigh.encoding<{dataLayout = "1D"}>>, tensor<2xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<1x3x3x2xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Unstick"([[VAR_6_]]) : (tensor<1x3x3x2xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x3x3x2xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {perm = [0, 3, 1, 2]} : (tensor<1x3x3x2xf32>) -> tensor<1x2x3x3xf32>
// CHECK:           return [[VAR_8_]] : tensor<1x2x3x3xf32>
// CHECK:         }
}
