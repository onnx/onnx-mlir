// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s



func.func @test_resize3(%arg0: tensor<1x1x1x4xf32>) -> tensor<1x1x1x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 1, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x1x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x1x12xf32>
    return %2 : tensor<1x1x1x12xf32>
}

// -----
func.func @test_resize2(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 1, 3, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Resize"(%arg0, %0, %0, %1) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x3x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>
}

// -----

func.func @test_resize(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x3x12xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 1.000000e+00, 3.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %0) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "floor"} : (tensor<1x1x3x4xf32>, none, tensor<4xf32>, none) -> tensor<1x1x3x12xf32>
    return %2 : tensor<1x1x3x12xf32>
}
