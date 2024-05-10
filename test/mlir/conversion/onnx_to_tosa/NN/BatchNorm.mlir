// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s
 
func.func @test_batchnorm_f32(%arg0: tensor<100x3x10x10xf32>) -> tensor<100x3x10x10xf32> {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32, momentum = 1.00000007E-3 : f32} : (tensor<100x3x10x10xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<100x3x10x10xf32>
    return %4 : tensor<100x3x10x10xf32>
// CHECK-LABEL: func @test_batchnorm_f32
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<100x3x10x10xf32>) -> tensor<100x3x10x10xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[VAR_0_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.reshape [[VAR_1_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.reshape [[VAR_3_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf32>) -> tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "tosa.const"() <{value = dense<1.00000007E-5> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.sub [[PARAM_0_]], [[VAR_4_]] : (tensor<100x3x10x10xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x10x10xf32>
// CHECK:           [[VAR_10_:%.+]] = tosa.add [[VAR_7_]], [[VAR_8_]] : (tensor<1x3x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x1x1xf32>
// CHECK:           [[VAR_11_:%.+]] = tosa.rsqrt [[VAR_10_]] : (tensor<1x3x1x1xf32>) -> tensor<1x3x1x1xf32>
// CHECK:           [[VAR_12_:%.+]] = tosa.mul [[VAR_9_]], [[VAR_11_]] {shift = 0 : i8} : (tensor<100x3x10x10xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x10x10xf32>
// CHECK:           [[VAR_13_:%.+]] = tosa.mul [[VAR_12_]], [[VAR_5_]] {shift = 0 : i8} : (tensor<100x3x10x10xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x10x10xf32>
// CHECK:           [[VAR_14_:%.+]] = tosa.add [[VAR_13_]], [[VAR_6_]] : (tensor<100x3x10x10xf32>, tensor<1x3x1x1xf32>) -> tensor<100x3x10x10xf32>
// CHECK:           return [[VAR_14_]] : tensor<100x3x10x10xf32>
}

// -----
func.func @test_batchnorm_f16_dynamic(%arg0: tensor<100x3x?x?xf16>) -> tensor<*xf16> {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32, momentum = 1.00000007E-3 : f32} : (tensor<100x3x?x?xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>) -> tensor<*xf16>
    return %4 : tensor<*xf16>
// CHECK-LABEL: func @test_batchnorm_f16_dynamic
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<100x3x?x?xf16>) -> tensor<100x3x?x?xf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf16>}> : () -> tensor<3xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xf16>}> : () -> tensor<3xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<3xf16>}> : () -> tensor<3xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf16>}> : () -> tensor<3xf16>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf16>) -> tensor<1x3x1x1xf16>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[VAR_0_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf16>) -> tensor<1x3x1x1xf16>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.reshape [[VAR_1_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf16>) -> tensor<1x3x1x1xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.reshape [[VAR_3_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf16>) -> tensor<1x3x1x1xf16>
// CHECK-DAG:       [[VAR_8_:%.+]] = "tosa.const"() <{value = dense<1.001360e-05> : tensor<1x1x1x1xf16>}> : () -> tensor<1x1x1x1xf16>
// CHECK:           [[VAR_9_:%.+]] = tosa.sub [[PARAM_0_]], [[VAR_4_]] : (tensor<100x3x?x?xf16>, tensor<1x3x1x1xf16>) -> tensor<100x3x?x?xf16>
// CHECK:           [[VAR_10_:%.+]] = tosa.add [[VAR_7_]], [[VAR_8_]] : (tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>) -> tensor<1x3x1x1xf16>
// CHECK:           [[VAR_11_:%.+]] = tosa.rsqrt [[VAR_10_]] : (tensor<1x3x1x1xf16>) -> tensor<1x3x1x1xf16>
// CHECK:           [[VAR_12_:%.+]] = tosa.mul [[VAR_9_]], [[VAR_11_]] {shift = 0 : i8} : (tensor<100x3x?x?xf16>, tensor<1x3x1x1xf16>) -> tensor<100x3x?x?xf16>
// CHECK:           [[VAR_13_:%.+]] = tosa.mul [[VAR_12_]], [[VAR_5_]] {shift = 0 : i8} : (tensor<100x3x?x?xf16>, tensor<1x3x1x1xf16>) -> tensor<100x3x?x?xf16>
// CHECK:           [[VAR_14_:%.+]] = tosa.add [[VAR_13_]], [[VAR_6_]] : (tensor<100x3x?x?xf16>, tensor<1x3x1x1xf16>) -> tensor<100x3x?x?xf16>
// CHECK:           return [[VAR_14_]] : tensor<100x3x?x?xf16>
}

// -----

func.func @test_batchnorm_bf16_dynamic(%arg0: tensor<100x3x?x?xbf16>) -> tensor<*xbf16> {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32, momentum = 1.00000007E-3 : f32} : (tensor<100x3x?x?xbf16>, tensor<3xbf16>, tensor<3xbf16>, tensor<3xbf16>, tensor<3xbf16>) -> tensor<*xbf16>
    return %4 : tensor<*xbf16>
// CHECK-LABEL: func @test_batchnorm_bf16_dynamic
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<100x3x?x?xbf16>) -> tensor<100x3x?x?xbf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xbf16>}> : () -> tensor<3xbf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xbf16>}> : () -> tensor<3xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<3xbf16>}> : () -> tensor<3xbf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xbf16>}> : () -> tensor<3xbf16>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xbf16>) -> tensor<1x3x1x1xbf16>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[VAR_0_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xbf16>) -> tensor<1x3x1x1xbf16>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.reshape [[VAR_1_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xbf16>) -> tensor<1x3x1x1xbf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.reshape [[VAR_3_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xbf16>) -> tensor<1x3x1x1xbf16>
// CHECK-DAG:       [[VAR_8_:%.+]] = "tosa.const"() <{value = dense<1.001360e-05> : tensor<1x1x1x1xbf16>}> : () -> tensor<1x1x1x1xbf16>
// CHECK:           [[VAR_9_:%.+]] = tosa.sub [[PARAM_0_]], [[VAR_4_]] : (tensor<100x3x?x?xbf16>, tensor<1x3x1x1xbf16>) -> tensor<100x3x?x?xbf16>
// CHECK:           [[VAR_10_:%.+]] = tosa.add [[VAR_7_]], [[VAR_8_]] : (tensor<1x3x1x1xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x3x1x1xbf16>
// CHECK:           [[VAR_11_:%.+]] = tosa.rsqrt [[VAR_10_]] : (tensor<1x3x1x1xbf16>) -> tensor<1x3x1x1xbf16>
// CHECK:           [[VAR_12_:%.+]] = tosa.mul [[VAR_9_]], [[VAR_11_]] {shift = 0 : i8} : (tensor<100x3x?x?xbf16>, tensor<1x3x1x1xbf16>) -> tensor<100x3x?x?xbf16>
// CHECK:           [[VAR_13_:%.+]] = tosa.mul [[VAR_12_]], [[VAR_5_]] {shift = 0 : i8} : (tensor<100x3x?x?xbf16>, tensor<1x3x1x1xbf16>) -> tensor<100x3x?x?xbf16>
// CHECK:           [[VAR_14_:%.+]] = tosa.add [[VAR_13_]], [[VAR_6_]] : (tensor<100x3x?x?xbf16>, tensor<1x3x1x1xbf16>) -> tensor<100x3x?x?xbf16>
// CHECK:           return [[VAR_14_]] : tensor<100x3x?x?xbf16>
}

// -----

func.func @test_batchnorm_f64(%arg0: tensor<100x3x10x10xf64>) -> tensor<100x3x10x10xf64> {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf64>} : () -> tensor<3xf64>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf64>} : () -> tensor<3xf64>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xf64>} : () -> tensor<3xf64>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xf64>} : () -> tensor<3xf64>
    %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32} : (tensor<100x3x10x10xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<100x3x10x10xf64>
    return %4 : tensor<100x3x10x10xf64>
// CHECK-LABEL: @test_batchnorm_f64
// CHECK-SAME: ([[PARAM_0:%.*]]: tensor<100x3x10x10xf64>) -> tensor<100x3x10x10xf64> {
// CHECK-NEXT: [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
// CHECK-NEXT: [[VAR_1_:%.+]]  = "tosa.const"() <{value = dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
// CHECK-NEXT: [[VAR_2_:%.+]]  = "tosa.const"() <{value = dense<[3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
// CHECK-NEXT: [[VAR_3_:%.+]]  = "tosa.const"() <{value = dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
// CHECK-NEXT: [[VAR_4_:%.+]]  = tosa.reshape [[VAR_2_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf64>) -> tensor<1x3x1x1xf64>
// CHECK-NEXT: [[VAR_5_:%.+]]  = tosa.reshape [[VAR_0_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf64>) -> tensor<1x3x1x1xf64>
// CHECK-NEXT: [[VAR_6_:%.+]]  = tosa.reshape [[VAR_1_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf64>) -> tensor<1x3x1x1xf64>
// CHECK-NEXT: [[VAR_7_:%.+]]  = tosa.reshape [[VAR_3_]] {new_shape = array<i64: 1, 3, 1, 1>} : (tensor<3xf64>) -> tensor<1x3x1x1xf64>
// CHECK-NEXT: [[VAR_8_:%.+]]  = "tosa.const"() <{value = dense<1.0000000656873453E-5> : tensor<1x1x1x1xf64>}> : () -> tensor<1x1x1x1xf64>
// CHECK-NEXT: [[VAR_9_:%.+]]  = tosa.sub %arg0, [[VAR_4_]] : (tensor<100x3x10x10xf64>, tensor<1x3x1x1xf64>) -> tensor<100x3x10x10xf64>
// CHECK-NEXT: [[VAR_10_:%.+]]  = tosa.add %7, [[VAR_8_]] : (tensor<1x3x1x1xf64>, tensor<1x1x1x1xf64>) -> tensor<1x3x1x1xf64>
// CHECK-NEXT: [[VAR_11_:%.+]]  = tosa.rsqrt [[VAR_10_]] : (tensor<1x3x1x1xf64>) -> tensor<1x3x1x1xf64>
// CHECK-NEXT: [[VAR_12_:%.+]]  = tosa.mul [[VAR_9_]], %11 {shift = 0 : i8} : (tensor<100x3x10x10xf64>, tensor<1x3x1x1xf64>) -> tensor<100x3x10x10xf64>
// CHECK-NEXT: [[VAR_13_:%.+]]  = tosa.mul [[VAR_12_]], %5 {shift = 0 : i8} : (tensor<100x3x10x10xf64>, tensor<1x3x1x1xf64>) -> tensor<100x3x10x10xf64>
// CHECK-NEXT: [[VAR_14_:%.+]]  = tosa.add [[VAR_13_]], [[VAR_6_]] : (tensor<100x3x10x10xf64>, tensor<1x3x1x1xf64>) -> tensor<100x3x10x10xf64>
// CHECK-NEXT: return [[VAR_14_]] : tensor<100x3x10x10xf64>
}
