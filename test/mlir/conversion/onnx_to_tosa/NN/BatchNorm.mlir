// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

 
func.func @test_batchnorm(%arg0: tensor<100x3x10x10xf32>) -> tensor<100x3x10x10xf32> {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32} : (tensor<100x3x10x10xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<100x3x10x10xf32>
    return %4 : tensor<100x3x10x10xf32>
// CHECK-LABEL: func @test_batchnorm
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<100x3x10x10xf32>) -> tensor<100x3x10x10xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[VAR_0_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.reshape [[VAR_1_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.reshape [[VAR_3_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "tosa.const"() <{value = dense<1.00000007E-5> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.reshape [[VAR_8_]] {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_10_:%.+]] = tosa.sub [[PARAM_0_]], [[VAR_4_]] : (tensor<100x3x10x10xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x10x10xf32>
// CHECK:           [[VAR_11_:%.+]] = tosa.add [[VAR_7_]], [[VAR_9_]] : (tensor<1x3x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x1x1xf32>
// CHECK:           [[VAR_12_:%.+]] = tosa.rsqrt [[VAR_11_]] : (tensor<1x3x1x1xf32>) -> tensor<1x3x1x1xf32>
// CHECK:           [[VAR_13_:%.+]] = tosa.mul [[VAR_10_]], [[VAR_12_]] {shift = 0 : i8} : (tensor<100x3x10x10xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x10x10xf32>
// CHECK:           [[VAR_14_:%.+]] = tosa.mul [[VAR_13_]], [[VAR_5_]] {shift = 0 : i8} : (tensor<100x3x10x10xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x10x10xf32>
// CHECK:           [[VAR_15_:%.+]] = tosa.add [[VAR_14_]], [[VAR_6_]] : (tensor<100x3x10x10xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x10x10xf32>
// CHECK:           return [[VAR_15_]] : tensor<100x3x10x10xf32>
}

// -----

func.func @test_batchnorm_dynamic(%arg0: tensor<100x3x?x?xf32>) -> tensor<*xf32> {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32} : (tensor<100x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<*xf32>
    return %4 : tensor<*xf32>
// CHECK-LABEL: func @test_batchnorm_dynamic
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<100x3x?x?xf32>) -> tensor<100x3x?x?xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[VAR_0_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.reshape [[VAR_1_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.reshape [[VAR_3_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "tosa.const"() <{value = dense<1.00000007E-5> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.reshape [[VAR_8_]] {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_10_:%.+]] = tosa.sub [[PARAM_0_]], [[VAR_4_]] : (tensor<100x3x?x?xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x?x?xf32>
// CHECK:           [[VAR_11_:%.+]] = tosa.add [[VAR_7_]], [[VAR_9_]] : (tensor<1x3x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x1x1xf32>
// CHECK:           [[VAR_12_:%.+]] = tosa.rsqrt [[VAR_11_]] : (tensor<1x3x1x1xf32>) -> tensor<1x3x1x1xf32>
// CHECK:           [[VAR_13_:%.+]] = tosa.mul [[VAR_10_]], [[VAR_12_]] {shift = 0 : i8} : (tensor<100x3x?x?xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x?x?xf32>
// CHECK:           [[VAR_14_:%.+]] = tosa.mul [[VAR_13_]], [[VAR_5_]] {shift = 0 : i8} : (tensor<100x3x?x?xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x?x?xf32>
// CHECK:           [[VAR_15_:%.+]] = tosa.add [[VAR_14_]], [[VAR_6_]] : (tensor<100x3x?x?xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x?x?xf32>
// CHECK:           return [[VAR_15_]] : tensor<100x3x?x?xf32>
}