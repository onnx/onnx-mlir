// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_resize_pytorch_half_pixel_linear(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x4x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x8xf32>
    return %2 : tensor<1x1x4x8xf32>
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linear
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x4xf32>) -> tensor<1x1x4x8xf32> {
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x2x4xf32>) -> tensor<1x2x4x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<[4, 2, 4, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<-1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_5_:%.+]] = tosa.resize [[VAR_1_]], [[VAR_2_]], [[VAR_3_]], [[VAR_4_]] {mode = BILINEAR} : (tensor<1x2x4x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x4x8x1xf32>
// CHECK:           [[VAR_7_:%.+]] = tosa.transpose [[VAR_5_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x4x8x1xf32>) -> tensor<1x1x4x8xf32>
// CHECK:           return [[VAR_7_]] : tensor<1x1x4x8xf32>
// CHECK:           }
}
// -----


func.func @test_resize_half_pixel_nearest_floor(%arg0: tensor<1x1x1x4xf32>) -> tensor<1x1x1x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 1, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x1x12xf32>
    return %2 : tensor<1x1x1x12xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x4xf32>) -> tensor<1x1x1x12xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x1x4xf32>) -> tensor<1x1x4x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[2, 2, 6, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<[-1, -5]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<-1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_4_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {mode = NEAREST_NEIGHBOR} : (tensor<1x1x4x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1x12x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x1x12x1xf32>) -> tensor<1x1x1x12xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x1x1x12xf32>
// CHECK:         }
}

// -----


func.func @test_resize_half_pixel_nearest_round_prefer_ceil(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 3, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x3x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_round_prefer_ceil
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x3x4xf32>) -> tensor<1x3x4x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[2, 2, 6, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<[0, -2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<[0, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_4_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {mode = NEAREST_NEIGHBOR} : (tensor<1x3x4x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x3x12x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x3x12x1xf32>) -> tensor<1x1x3x12xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x1x3x12xf32>
// CHECK:         }
}

// -----


func.func @test_resize_align_corners(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 3, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "align_corners", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x3x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_align_corners
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x3x4xf32>) -> tensor<1x3x4x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[2, 2, 22, 6]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_3_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_2_]] {mode = NEAREST_NEIGHBOR} : (tensor<1x3x4x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x3x12x1xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_3_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x3x12x1xf32>) -> tensor<1x1x3x12xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1x3x12xf32>
// CHECK:         }
}

// -----


func.func @test_resize_asymmetric(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 3, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x3x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_asymmetric
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x3x4xf32>) -> tensor<1x3x4x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[2, 2, 6, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<[0, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_4_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {mode = NEAREST_NEIGHBOR} : (tensor<1x3x4x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x3x12x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x3x12x1xf32>) -> tensor<1x1x3x12xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x1x3x12xf32>
// CHECK:         }
}

// -----


func.func @test_resize_half_pixel_nearest_floor_downsample(%arg0: tensor<1x1x1x12xf32>) -> tensor<1x1x1x4xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 1, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x12xf32>, none, none, tensor<4xi64>) -> tensor<1x1x1x4xf32>
    return %2 : tensor<1x1x1x4xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_floor_downsample
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x12xf32>) -> tensor<1x1x1x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x1x12xf32>) -> tensor<1x1x12x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[2, 2, 2, 6]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<[-1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<[-1, -3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_4_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {mode = NEAREST_NEIGHBOR} : (tensor<1x1x12x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1x4x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x1x4x1xf32>) -> tensor<1x1x1x4xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x1x1x4xf32>
// CHECK:         }
}

// -----


func.func @test_resize_input_one(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 4, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "floor"} : (tensor<1x1x1x1xf32>, none, none, tensor<4xi64>) -> tensor<1x1x4x4xf32>
    return %2 : tensor<1x1x4x4xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_input_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[8, 2, 8, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<-3> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<3> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_4_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {mode = BILINEAR} : (tensor<1x1x1x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x4x4x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x4x4x1xf32>) -> tensor<1x1x4x4xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x1x4x4xf32>
// CHECK:         }
}

// -----


func.func @test_resize_pytorch_half_pixel_linear_float_scale_upsample(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x4x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.001000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x8xf32>
    return %2 : tensor<1x1x4x8xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linear_float_scale_upsample
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x4xf32>) -> tensor<1x1x4x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x2x4xf32>) -> tensor<1x2x4x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[500249982, 250000000, 4, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<[-125124991, -1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<[124625027, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_4_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {mode = BILINEAR} : (tensor<1x2x4x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x4x8x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x4x8x1xf32>) -> tensor<1x1x4x8xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x1x4x8xf32>
// CHECK:         }
}

// -----


func.func @test_resize_pytorch_half_pixel_linear_float_scale_downsample(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x1x2xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 0.600000e+00, 0.600000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<4xf32>, none) -> tensor<1x1x1x2xf32>
    return %2 : tensor<1x1x1x2xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linear_float_scale_downsample
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x4xf32>) -> tensor<1x1x1x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x2x4xf32>) -> tensor<1x2x4x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[1, 1, 150000006, 250000000]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<[-1, 49999997]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<[-2, -150000021]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_4_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {mode = BILINEAR} : (tensor<1x2x4x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1x2x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x1x2x1xf32>) -> tensor<1x1x1x2xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x1x1x2xf32>
// CHECK:         }
}

// -----


func.func @test_resize_half_pixel_nearest_floor_downsample_axis_both(%arg0: tensor<1x1x1x12xf32>) -> tensor<1x1x1x6xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 6]> : tensor<2xi64>} : () -> tensor<2xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {axes = [2, 3], coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x12xf32>, none, none, tensor<2xi64>) -> tensor<1x1x1x6xf32>
    return %2 : tensor<1x1x1x6xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_floor_downsample_axis_both
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x12xf32>) -> tensor<1x1x1x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x1x12xf32>) -> tensor<1x1x12x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[2, 2, 2, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<[-1, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<[-1, -2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_4_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {mode = NEAREST_NEIGHBOR} : (tensor<1x1x12x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1x6x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x1x6x1xf32>) -> tensor<1x1x1x6xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x1x1x6xf32>
// CHECK:         }
}

// -----


func.func @test_resize_half_pixel_nearest_floor_downsample_axis_one(%arg0: tensor<1x1x1x12xf32>) -> tensor<1x1x1x6xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[6]> : tensor<1xi64>} : () -> tensor<1xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {axes = [3], coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x12xf32>, none, none, tensor<1xi64>) -> tensor<1x1x1x6xf32>
    return %2 : tensor<1x1x1x6xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_half_pixel_nearest_floor_downsample_axis_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x12xf32>) -> tensor<1x1x1x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x1x12xf32>) -> tensor<1x1x12x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[2, 2, 2, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<[-1, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<[-1, -2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_4_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {mode = NEAREST_NEIGHBOR} : (tensor<1x1x12x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1x6x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x1x6x1xf32>) -> tensor<1x1x1x6xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x1x1x6xf32>
// CHECK:         }
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

 // > successful test; print test_resize_linear_int_disallowed
// -----
func.func @test_resize_linear_int_disallowed(%arg0: tensor<1x1x2x4xi32>) -> tensor<1x1x4x8xi32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xi32>, none, tensor<4xf32>, none) -> tensor<1x1x4x8xi32>
    return %2 : tensor<1x1x4x8xi32>
// CHECK-LABEL:  func.func @test_resize_linear_int_disallowed
// CHECK:        onnx.Resize
}

// -----


func.func @test_resize_pytorch_half_pixel_linear_other_axis_allowed_negative_axis(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x2x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {axes = [1, -1], coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<2xf32>, none) -> tensor<1x1x2x8xf32>
    return %2 : tensor<1x1x2x8xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize_pytorch_half_pixel_linear_other_axis_allowed_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x4xf32>) -> tensor<1x1x2x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x2x4xf32>) -> tensor<1x2x4x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<[2, 2, 4, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<[0, -1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<[0, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_4_:%.+]] = tosa.resize [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {mode = BILINEAR} : (tensor<1x2x4x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x2x8x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x2x8x1xf32>) -> tensor<1x1x2x8xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x1x2x8xf32>
// CHECK:         }
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

// -----

func.func @test_resize_linear_int_disallowed(%arg0: tensor<1x1x2x4xi32>) -> tensor<1x1x4x8xi32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xi32>, none, tensor<4xf32>, none) -> tensor<1x1x4x8xi32>
    return %2 : tensor<1x1x4x8xi32>
// CHECK-LABEL:  func.func @test_resize_linear_int_disallowed
// CHECK:        onnx.Resize
}