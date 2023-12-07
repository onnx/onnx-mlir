// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_resize_pytorch_half_pixel_linear(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x4x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x8xf32>
    return %2 : tensor<1x1x4x8xf32>
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linear
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x4xf32>) -> tensor<1x1x4x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x2x4xf32>, tensor<4xi32>) -> tensor<1x2x4x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: 1, 1>, mode = "BILINEAR", offset = array<i64: -1, -1>, scale = array<i64: 4, 2, 4, 2>} : (tensor<1x2x4x1xf32>) -> tensor<1x4x8x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x4x8x1xf32>, tensor<4xi32>) -> tensor<1x1x4x8xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x4x8xf32>
}

// -----

func.func @test_resize_half_pixel_nearest_floor(%arg0: tensor<1x1x1x4xf32>) -> tensor<1x1x1x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 1, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x1x12xf32>
    return %2 : tensor<1x1x1x12xf32>
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x4xf32>) -> tensor<1x1x1x12xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x1x4xf32>, tensor<4xi32>) -> tensor<1x1x4x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: -1, -1>, mode = "NEAREST_NEIGHBOR", offset = array<i64: -1, -5>, scale = array<i64: 2, 2, 6, 2>} : (tensor<1x1x4x1xf32>) -> tensor<1x1x12x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x1x12x1xf32>, tensor<4xi32>) -> tensor<1x1x1x12xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x1x12xf32>
}

// -----

func.func @test_resize_half_pixel_nearest_round_prefer_ceil(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 3, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x3x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_round_prefer_ceil
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x3x4xf32>, tensor<4xi32>) -> tensor<1x3x4x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: 0, 2>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, -2>, scale = array<i64: 2, 2, 6, 2>} : (tensor<1x3x4x1xf32>) -> tensor<1x3x12x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x3x12x1xf32>, tensor<4xi32>) -> tensor<1x1x3x12xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x3x12xf32>
}

// -----

func.func @test_resize_align_corners(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 3, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "align_corners", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x3x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>
// CHECK-LABEL:  func.func @test_resize_align_corners
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x3x4xf32>, tensor<4xi32>) -> tensor<1x3x4x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: 0, 0>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, 0>, scale = array<i64: 2, 2, 22, 6>} : (tensor<1x3x4x1xf32>) -> tensor<1x3x12x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x3x12x1xf32>, tensor<4xi32>) -> tensor<1x1x3x12xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x3x12xf32>
}

// -----

func.func @test_resize_asymmetric(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 3, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x3x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>
// CHECK-LABEL:  func.func @test_resize_asymmetric
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x3x4xf32>, tensor<4xi32>) -> tensor<1x3x4x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: 0, 4>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, 0>, scale = array<i64: 2, 2, 6, 2>} : (tensor<1x3x4x1xf32>) -> tensor<1x3x12x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x3x12x1xf32>, tensor<4xi32>) -> tensor<1x1x3x12xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x3x12xf32>
}

// -----

func.func @test_resize_half_pixel_nearest_floor_downsample(%arg0: tensor<1x1x1x12xf32>) -> tensor<1x1x1x4xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 1, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x12xf32>, none, none, tensor<4xi64>) -> tensor<1x1x1x4xf32>
    return %2 : tensor<1x1x1x4xf32>
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_floor_downsample
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x12xf32>) -> tensor<1x1x1x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x1x12xf32>, tensor<4xi32>) -> tensor<1x1x12x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: -1, -3>, mode = "NEAREST_NEIGHBOR", offset = array<i64: -1, 1>, scale = array<i64: 2, 2, 2, 6>} : (tensor<1x1x12x1xf32>) -> tensor<1x1x4x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x1x4x1xf32>, tensor<4xi32>) -> tensor<1x1x1x4xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x1x4xf32>
}

// -----

func.func @test_resize_input_one(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 4, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "floor"} : (tensor<1x1x1x1xf32>, none, none, tensor<4xi64>) -> tensor<1x1x4x4xf32>
    return %2 : tensor<1x1x4x4xf32>
// CHECK-LABEL:  func.func @test_resize_input_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: 3, 3>, mode = "BILINEAR", offset = array<i64: -3, -3>, scale = array<i64: 8, 2, 8, 2>} : (tensor<1x1x1x1xf32>) -> tensor<1x4x4x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x4x4x1xf32>, tensor<4xi32>) -> tensor<1x1x4x4xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x4x4xf32>
}

// -----

func.func @test_resize_pytorch_half_pixel_linear_float_scale_upsample(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x4x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.001000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x8xf32>
    return %2 : tensor<1x1x4x8xf32>
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linear_float_scale_upsample
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x4xf32>) -> tensor<1x1x4x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x2x4xf32>, tensor<4xi32>) -> tensor<1x2x4x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: 125124991, 1>, mode = "BILINEAR", offset = array<i64: -125124991, -1>, scale = array<i64: 500249982, 250000000, 4, 2>} : (tensor<1x2x4x1xf32>) -> tensor<1x4x8x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x4x8x1xf32>, tensor<4xi32>) -> tensor<1x1x4x8xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x4x8xf32>
}

// -----

func.func @test_resize_pytorch_half_pixel_linear_float_scale_downsample(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x1x2xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 0.6000e+00, 0.600000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<4xf32>, none) -> tensor<1x1x1x2xf32>
    return %2 : tensor<1x1x1x2xf32>
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linear_float_scale_downsample
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x4xf32>) -> tensor<1x1x1x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x2x4xf32>, tensor<4xi32>) -> tensor<1x2x4x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: -49999997, -49999997>, mode = "BILINEAR", offset = array<i64: 49999997, 49999997>, scale = array<i64: 150000006, 250000000, 150000006, 250000000>} : (tensor<1x2x4x1xf32>) -> tensor<1x1x2x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x1x2x1xf32>, tensor<4xi32>) -> tensor<1x1x1x2xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x1x2xf32>
}

// -----

func.func @test_resize_half_pixel_nearest_floor_downsample_axis_both(%arg0: tensor<1x1x1x12xf32>) -> tensor<1x1x1x6xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 6]> : tensor<2xi64>} : () -> tensor<2xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {axes = [2, 3], coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x12xf32>, none, none, tensor<2xi64>) -> tensor<1x1x1x6xf32>
    return %2 : tensor<1x1x1x6xf32>
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_floor_downsample_axis_both
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x12xf32>) -> tensor<1x1x1x6xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x1x12xf32>, tensor<4xi32>) -> tensor<1x1x12x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: -1, -2>, mode = "NEAREST_NEIGHBOR", offset = array<i64: -1, 0>, scale = array<i64: 2, 2, 2, 4>} : (tensor<1x1x12x1xf32>) -> tensor<1x1x6x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x1x6x1xf32>, tensor<4xi32>) -> tensor<1x1x1x6xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x1x6xf32>
}

// -----

func.func @test_resize_half_pixel_nearest_floor_downsample_axis_one(%arg0: tensor<1x1x1x12xf32>) -> tensor<1x1x1x6xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[6]> : tensor<1xi64>} : () -> tensor<1xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {axes = [3], coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x12xf32>, none, none, tensor<1xi64>) -> tensor<1x1x1x6xf32>
    return %2 : tensor<1x1x1x6xf32>
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_floor_downsample_axis_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x12xf32>) -> tensor<1x1x1x6xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x1x12xf32>, tensor<4xi32>) -> tensor<1x1x12x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: -1, -2>, mode = "NEAREST_NEIGHBOR", offset = array<i64: -1, 0>, scale = array<i64: 2, 2, 2, 4>} : (tensor<1x1x12x1xf32>) -> tensor<1x1x6x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x1x6x1xf32>, tensor<4xi32>) -> tensor<1x1x1x6xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x1x6xf32>
}

// -----

func.func @test_resize_pytorch_half_pixel_linear_other_axis_allowed(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x2x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {axes = [1, 3], coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<2xf32>, none) -> tensor<1x1x2x8xf32>
    return %2 : tensor<1x1x2x8xf32>
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linear_other_axis_allowed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x4xf32>) -> tensor<1x1x2x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x2x4xf32>, tensor<4xi32>) -> tensor<1x2x4x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: -1, 1>, mode = "BILINEAR", offset = array<i64: -1, -1>, scale = array<i64: 1, 1, 4, 2>} : (tensor<1x2x4x1xf32>) -> tensor<1x2x8x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x2x8x1xf32>, tensor<4xi32>) -> tensor<1x1x2x8xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x2x8xf32>
}

// -----

func.func @test_resize_pytorch_half_pixel_linear_other_axis_allowed_negative_axis(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x2x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {axes = [1, -1], coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<2xf32>, none) -> tensor<1x1x2x8xf32>
    return %2 : tensor<1x1x2x8xf32>
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linear_other_axis_allowed_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x4xf32>) -> tensor<1x1x2x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x2x4xf32>, tensor<4xi32>) -> tensor<1x2x4x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.resize [[VAR_1_]] {border = array<i64: -1, 1>, mode = "BILINEAR", offset = array<i64: -1, -1>, scale = array<i64: 1, 1, 4, 2>} : (tensor<1x2x4x1xf32>) -> tensor<1x2x8x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x2x8x1xf32>, tensor<4xi32>) -> tensor<1x1x2x8xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x2x8xf32>
}

// -----

func.func @test_resize_pytorch_half_pixel_linearother_axis_disallowed(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x2x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {axes = [1, 0], coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<2xf32>, none) -> tensor<1x1x2x8xf32>
    return %2 : tensor<1x1x2x8xf32>
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linearother_axis_disallowed
// CHECK-LABEL:  onnx.Resize
}

// -----

func.func @test_resize_cubic_disallowed(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x2x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {axes = [1, 3], coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "cubic", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<2xf32>, none) -> tensor<1x1x2x8xf32>
    return %2 : tensor<1x1x2x8xf32>
// CHECK-LABEL:  func.func @test_resize_cubic_disallowed
// CHECK-LABEL:  onnx.Resize
}