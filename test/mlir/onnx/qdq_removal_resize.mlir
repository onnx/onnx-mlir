// RUN: onnx-mlir-opt --canonicalize --qdq-around-op-opt-onnx-to-onnx %s -split-input-file | FileCheck %s

  func.func @resize_op() -> (tensor<2x4xui8> {onnx.name = "quantized"}) {
    %0 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
    %1 = onnx.Constant dense<[[0, 1], [2, 3]]> : tensor<2x2xui8>
    %2 = onnx.Constant dense<[2, 4]> : tensor<2xi64>
    %3 = "onnx.Resize"(%1, %0, %0, %2) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "round_prefer_floor", onnx_node_name = "onnx.Resize_2"} : (tensor<2x2xui8>, none, none, tensor<2xi64>) -> tensor<2x4xui8>
    return %3 : tensor<2x4xui8>
  }

// CHECK-LABEL:  func.func @resize_op
// CHECK-SAME:   () -> (tensor<2x4xui8> {onnx.name = "quantized"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<{{.}}[0, 1], [2, 3]{{.}}> : tensor<2x2xui8>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[2, 4]> : tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Resize"([[VAR_1_]], [[VAR_0_]], [[VAR_0_]], [[VAR_2_]]) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "round_prefer_floor", onnx_node_name = "onnx.Resize_2"} : (tensor<2x2xui8>, none, none, tensor<2xi64>) -> tensor<2x4xui8>
// CHECK:           return [[VAR_3_]] : tensor<2x4xui8>
// CHECK:         }