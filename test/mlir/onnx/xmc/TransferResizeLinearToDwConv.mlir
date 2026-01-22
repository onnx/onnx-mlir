// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: flexml-opt %configPassStx -annotate-config="library-metadata-dirs=%S" %s -transfer-resize-linear-to-dw-conv -o - | FileCheck %s

// CHECK-LABEL: resize_subgraph_1
func.func @resize_subgraph_1(%arg0: tensor<1x1x4x4x4xf32>) -> tensor<1x1x8x8x8xf32> {
    %0 = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<5xf32>
    %1 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
    %2 = "onnx.Resize"(%arg0, %1, %0, %1) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "round_prefer_floor", onnx_node_name = "Resize5D"} : (tensor<1x1x4x4x4xf32>, none, tensor<5xf32>, none) -> tensor<1x1x8x8x8xf32>
    return %2 : tensor<1x1x8x8x8xf32>
  }

// CHECK: onnx.ConvTranspose

// CHECK-LABEL: resize_subgraph_2
func.func @resize_subgraph_2(%arg0: tensor<1x3x64x64xf32>) -> tensor<1x3x128x128xf32> {

    %0 = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>

    %1 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none

    %2 = "onnx.Resize"(%arg0, %1, %0, %1) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "round_prefer_floor", onnx_node_name = "onnx.Resize_1"} : (tensor<1x3x64x64xf32>, none, tensor<4xf32>, none) -> tensor<1x3x128x128xf32>

    return %2 : tensor<1x3x128x128xf32>

  }

// CHECK: onnx.ConvTranspose

// CHECK-LABEL: resize_subgraph_3
func.func @resize_subgraph_3(%arg0: tensor<1x1x1x4x4xf32>) -> tensor<1x1x1x8x8xf32> {

    %0 = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<5xf32>
    %1 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
    %2 = "onnx.Resize"(%arg0, %1, %0, %1) {
      antialias = 0 : si64,
      coordinate_transformation_mode = "asymmetric",
      cubic_coeff_a = -7.500000e-01 : f32,
      exclude_outside = 0 : si64,
      extrapolation_value = 0.000000e+00 : f32,
      keep_aspect_ratio_policy = "stretch",
      mode = "linear",
      nearest_mode = "round_prefer_floor",
      onnx_node_name = "Resize5D_D1_UpsampleHW"} : (tensor<1x1x1x4x4xf32>, none, tensor<5xf32>, none) -> tensor<1x1x1x8x8xf32>
    return %2 : tensor<1x1x1x8x8xf32>
  }

// CHECK: onnx.ConvTranspose

// CHECK-LABEL: resize_subgraph_4
func.func @resize_subgraph_4(%arg0: tensor<1x1x8x8xf32>) -> tensor<1x1x4x4xf32> {

    %0 = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 5.000000e-01, 5.000000e-01]> : tensor<4xf32>
    %1 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
    %2 = "onnx.Resize"(%arg0, %1, %0, %1) {
      antialias = 0 : si64,
      coordinate_transformation_mode = "asymmetric",
      cubic_coeff_a = -7.500000e-01 : f32,
      exclude_outside = 0 : si64,
      extrapolation_value = 0.000000e+00 : f32,
      keep_aspect_ratio_policy = "stretch",
      mode = "linear",
      nearest_mode = "round_prefer_floor",
      onnx_node_name = "Resize4D_Downsample"} : (tensor<1x1x8x8xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x4xf32>
    return %2 : tensor<1x1x4x4xf32>
  }

// CHECK: onnx.Conv

// CHECK-LABEL: resize_subgraph_5
func.func @resize_subgraph_5(%arg0: tensor<1x1x8x8x8xf32>) -> tensor<1x1x4x4x4xf32> {

    %0 = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 5.000000e-01, 5.000000e-01, 5.000000e-01]> : tensor<5xf32>
    %1 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
    %2 = "onnx.Resize"(%arg0, %1, %0, %1) {
      antialias = 0 : si64,
      coordinate_transformation_mode = "asymmetric",
      cubic_coeff_a = -7.500000e-01 : f32,
      exclude_outside = 0 : si64,
      extrapolation_value = 0.000000e+00 : f32,
      keep_aspect_ratio_policy = "stretch",
      mode = "linear",
      nearest_mode = "round_prefer_floor",
      onnx_node_name = "Resize5D_Downsample"} : (tensor<1x1x8x8x8xf32>, none, tensor<5xf32>, none) -> tensor<1x1x4x4x4xf32>
    return %2 : tensor<1x1x4x4x4xf32>
  }

// CHECK: onnx.Conv
