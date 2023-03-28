// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_resize_pytorch_half_pixel_linear(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x4x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x8xf32>
    return %2 : tensor<1x1x4x8xf32>
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linear(
// CHECK:  %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_2:.*]] = "tosa.transpose"(%arg0, %[[VAL_1]]) : (tensor<1x1x2x4xf32>, tensor<4xi32>) -> tensor<1x2x4x1xf32>
// CHECK:  %[[VAL_3:.*]] = "tosa.resize"(%[[VAL_2]]) {border = array<i64: 1, 1>, mode = "BILINEAR", offset = array<i64: -1, -1>, scale = array<i64: 4, 2, 4, 2>} : (tensor<1x2x4x1xf32>) -> tensor<1x4x8x1xf32>
// CHECK:  %[[VAL_4:.*]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_5:.*]] = "tosa.transpose"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x4x8x1xf32>, tensor<4xi32>) -> tensor<1x1x4x8xf32>
// CHECK:  return %[[VAL_5]] : tensor<1x1x4x8xf32>
}

// -----
func.func @test_resize_half_pixel_nearest_floor(%arg0: tensor<1x1x1x4xf32>) -> tensor<1x1x1x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 1, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x1x12xf32>
    return %2 : tensor<1x1x1x12xf32>
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_floor(
// CHECK:  %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_2:.*]] = "tosa.transpose"(%arg0, %[[VAL_1]]) : (tensor<1x1x1x4xf32>, tensor<4xi32>) -> tensor<1x1x4x1xf32>
// CHECK:  %[[VAL_3:.*]] = "tosa.resize"(%[[VAL_2]]) {border = array<i64: 0, -1>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, -5>, scale = array<i64: 1, 1, 6, 2>} : (tensor<1x1x4x1xf32>) -> tensor<1x1x12x1xf32>
// CHECK:  %[[VAL_4:.*]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_5:.*]] = "tosa.transpose"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x1x12x1xf32>, tensor<4xi32>) -> tensor<1x1x1x12xf32>
// CHECK:  return %[[VAL_5]] : tensor<1x1x1x12xf32>
}

// -----
func.func @test_resize_half_pixel_nearest_round_prefer_ceil(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 3, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x3x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_round_prefer_ceil(
// CHECK:  %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_2:.*]] = "tosa.transpose"(%arg0, %[[VAL_1]]) : (tensor<1x1x3x4xf32>, tensor<4xi32>) -> tensor<1x3x4x1xf32>
// CHECK:  %[[VAL_3:.*]] = "tosa.resize"(%[[VAL_2]]) {border = array<i64: 0, 2>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, -2>, scale = array<i64: 2, 2, 6, 2>} : (tensor<1x3x4x1xf32>) -> tensor<1x3x12x1xf32>
// CHECK:  %[[VAL_4:.*]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_5:.*]] = "tosa.transpose"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x3x12x1xf32>, tensor<4xi32>) -> tensor<1x1x3x12xf32>
// CHECK:  return %[[VAL_5]] : tensor<1x1x3x12xf32>
}

// -----
func.func @test_resize_align_corners(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 3, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "align_corners", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x3x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>
// CHECK-LABEL:  func.func @test_resize_align_corners(
// CHECK:  %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_2:.*]] = "tosa.transpose"(%arg0, %[[VAL_1]]) : (tensor<1x1x3x4xf32>, tensor<4xi32>) -> tensor<1x3x4x1xf32>
// CHECK:  %[[VAL_3:.*]] = "tosa.resize"(%[[VAL_2]]) {border = array<i64: 0, 0>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, 0>, scale = array<i64: 2, 2, 22, 6>} : (tensor<1x3x4x1xf32>) -> tensor<1x3x12x1xf32>
// CHECK:  %[[VAL_4:.*]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_5:.*]] = "tosa.transpose"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x3x12x1xf32>, tensor<4xi32>) -> tensor<1x1x3x12xf32>
// CHECK:  return %[[VAL_5]] : tensor<1x1x3x12xf32>
}

// -----
func.func @test_resize_asymmetric(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 3, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x3x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>
// CHECK-LABEL:  func.func @test_resize_asymmetric(
// CHECK:  %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_2:.*]] = "tosa.transpose"(%arg0, %[[VAL_1]]) : (tensor<1x1x3x4xf32>, tensor<4xi32>) -> tensor<1x3x4x1xf32>
// CHECK:  %[[VAL_3:.*]] = "tosa.resize"(%[[VAL_2]]) {border = array<i64: 0, 4>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, 0>, scale = array<i64: 2, 2, 6, 2>} : (tensor<1x3x4x1xf32>) -> tensor<1x3x12x1xf32>
// CHECK:  %[[VAL_4:.*]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_5:.*]] = "tosa.transpose"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x3x12x1xf32>, tensor<4xi32>) -> tensor<1x1x3x12xf32>
// CHECK:  return %[[VAL_5]] : tensor<1x1x3x12xf32>
}

// -----
func.func @test_resize_half_pixel_nearest_floor_downsample(%arg0: tensor<1x1x1x12xf32>) -> tensor<1x1x1x4xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 1, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x12xf32>, none, none, tensor<4xi64>) -> tensor<1x1x1x4xf32>
    return %2 : tensor<1x1x1x4xf32>
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_floor_downsample(
// CHECK:  %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_2:.*]] = "tosa.transpose"(%arg0, %[[VAL_1]]) : (tensor<1x1x1x12xf32>, tensor<4xi32>) -> tensor<1x1x12x1xf32>
// CHECK:  %[[VAL_3:.*]] = "tosa.resize"(%[[VAL_2]]) {border = array<i64: 0, -3>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, 1>, scale = array<i64: 1, 1, 2, 6>} : (tensor<1x1x12x1xf32>) -> tensor<1x1x4x1xf32>
// CHECK:  %[[VAL_4:.*]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_5:.*]] = "tosa.transpose"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x1x4x1xf32>, tensor<4xi32>) -> tensor<1x1x1x4xf32>
// CHECK:  return %[[VAL_5]] : tensor<1x1x1x4xf32>
}

// -----
func.func @test_resize_input_one(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 4, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "floor"} : (tensor<1x1x1x1xf32>, none, none, tensor<4xi64>) -> tensor<1x1x4x4xf32>
    return %2 : tensor<1x1x4x4xf32>
// CHECK-LABEL:  func.func @test_resize_input_one(
// CHECK:  %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_2:.*]] = "tosa.transpose"(%arg0, %[[VAL_1]]) : (tensor<1x1x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x1xf32>
// CHECK:  %[[VAL_3:.*]] = "tosa.resize"(%[[VAL_2]]) {border = [3, 3], mode = "BILINEAR", offset = [0, 0], scale = [4, 1, 4, 1]} : (tensor<1x1x1x1xf32>) -> tensor<1x4x4x1xf32>
// CHECK:  %[[VAL_4:.*]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:  %[[VAL_5:.*]] = "tosa.transpose"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x4x4x1xf32>, tensor<4xi32>) -> tensor<1x1x4x4xf32>
// CHECK:  return %[[VAL_5]] : tensor<1x1x4x4xf32>
}