// RUN: onnx-mlir-opt --canonicalize --qdq-around-op-opt-onnx-to-onnx %s -split-input-file | FileCheck %s
  func.func @resize_op(%arg0: tensor<1x3x64x64xui8> {onnx.name = "input"}) -> (tensor<1x3x128x128xui8> {onnx.name = "output"}) {
    %0 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
    %1 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %2 = onnx.Constant dense<0> : tensor<1xui8>
    %3 = onnx.Constant dense<[1, 3, 128, 128]> : tensor<4xi64>
    %4 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %5 = onnx.Constant dense<0> : tensor<1xui8>
    %6 = "onnx.DequantizeLinear"(%arg0, %1, %2) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.DequantizeLinear_1"} : (tensor<1x3x64x64xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x64x64xf32>
    %7 = "onnx.Resize"(%6, %0, %0, %3) {
      antialias = 0 : si64,
      coordinate_transformation_mode = "asymmetric",
      cubic_coeff_a = -7.500000e-01 : f32,
      exclude_outside = 0 : si64,
      extrapolation_value = 0.000000e+00 : f32,
      keep_aspect_ratio_policy = "stretch",
      mode = "nearest",
      nearest_mode = "round_prefer_floor",
      onnx_node_name = "onnx.Resize_2"} : (tensor<1x3x64x64xf32>, none, none, tensor<4xi64>) -> tensor<1x3x128x128xf32>
    %8 = "onnx.QuantizeLinear"(%7, %4, %5) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.QuantizeLinear_3",
      output_dtype = 0 : si64,
      saturate = 1 : si64} : (tensor<1x3x128x128xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x128x128xui8>
    return %8 : tensor<1x3x128x128xui8>
  }

// CHECK-LABEL:  func.func @resize_op
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x64x64xui8> {onnx.name = "input"}) -> (tensor<1x3x128x128xui8> {onnx.name = "output"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 3, 128, 128]> : tensor<4xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Resize"([[PARAM_0_]], [[VAR_0_]], [[VAR_0_]], [[VAR_1_]]) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "round_prefer_floor", onnx_node_name = "onnx.Resize_2"} : (tensor<1x3x64x64xui8>, none, none, tensor<4xi64>) -> tensor<1x3x128x128xui8>
// CHECK:           return [[VAR_2_]] : tensor<1x3x128x128xui8>
// CHECK:         }

func.func @resize_op_cubic(%arg0: tensor<1x3x64x64xui8> {onnx.name = "input"}) -> (tensor<1x3x128x128xui8> {onnx.name = "output"}) {
    %0 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
    %1 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %2 = onnx.Constant dense<0> : tensor<1xui8>
    %3 = onnx.Constant dense<[1, 3, 128, 128]> : tensor<4xi64>
    %4 = "onnx.DequantizeLinear"(%arg0, %1, %2) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.DequantizeLinear_1"} : (tensor<1x3x64x64xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x64x64xf32>
    %5 = "onnx.Resize"(%4, %0, %0, %3) {
      antialias = 0 : si64,
      coordinate_transformation_mode = "asymmetric",
      cubic_coeff_a = -7.500000e-01 : f32,
      exclude_outside = 0 : si64,
      extrapolation_value = 0.000000e+00 : f32,
      keep_aspect_ratio_policy = "stretch",
      mode = "cubic",
      nearest_mode = "round_prefer_floor",
      onnx_node_name = "onnx.Resize_2"} : (tensor<1x3x64x64xf32>, none, none, tensor<4xi64>) -> tensor<1x3x128x128xf32>
    %6 = "onnx.QuantizeLinear"(%5, %1, %2) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.QuantizeLinear_3",
      output_dtype = 0 : si64,
      saturate = 1 : si64} : (tensor<1x3x128x128xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x128x128xui8>
    return %6 : tensor<1x3x128x128xui8>
  }

// CHECK-LABEL:  func.func @resize_op_cubic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x64x64xui8> {onnx.name = "input"}) -> (tensor<1x3x128x128xui8> {onnx.name = "output"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0> : tensor<1xui8>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[1, 3, 128, 128]> : tensor<4xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.DequantizeLinear"([[PARAM_0_]], [[VAR_1_]], [[VAR_2_]]) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "onnx.DequantizeLinear_1"} : (tensor<1x3x64x64xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x64x64xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Resize"([[VAR_4_]], [[VAR_0_]], [[VAR_0_]], [[VAR_3_]]) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "cubic", nearest_mode = "round_prefer_floor", onnx_node_name = "onnx.Resize_2"} : (tensor<1x3x64x64xf32>, none, none, tensor<4xi64>) -> tensor<1x3x128x128xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.QuantizeLinear"([[VAR_5_]], [[VAR_1_]], [[VAR_2_]]) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "onnx.QuantizeLinear_3", output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x3x128x128xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x128x128xui8>
// CHECK:           return [[VAR_6_]] : tensor<1x3x128x128xui8>
// CHECK:         }