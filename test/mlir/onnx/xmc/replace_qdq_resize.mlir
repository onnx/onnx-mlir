// RUN: onnx-mlir-opt --replace-qdq-resize %s --split-input-file | FileCheck %s

// =============================================================================
// Positive: 1x1 spatial XFEResize with matching uniform quant types upsampling
// to 2x2 must collapse to onnx.Add(input, splat_zero_point_const) because the
// Resize is functionally a broadcast (only one source pixel per (N, C)).
// =============================================================================
// CHECK-LABEL: func.func @resize_1x1_to_2x2_matching_quant
func.func @resize_1x1_to_2x2_matching_quant(
    %arg0: tensor<1x1x1x128x!quant.uniform<u8:f32, 3.125000e-02:128>>)
    -> tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "pytorch_half_pixel",
    mode = "linear",
    nearest_mode = "floor"
  } : (tensor<1x1x1x128x!quant.uniform<u8:f32, 3.125000e-02:128>>, none, tensor<4xf32>, none)
   -> tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>
  return %0 : tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>

  // CHECK:           %[[ZP_CONST:.*]] = onnx.Constant
  // CHECK-SAME:        dense<128>
  // CHECK-SAME:        tensor<1x2x2x128xui8>
  // CHECK-SAME:        tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>
  // CHECK:           %[[ADD:.*]] = "onnx.Add"(%arg0, %[[ZP_CONST]])
  // CHECK-SAME:        tensor<1x1x1x128x!quant.uniform<u8:f32, 3.125000e-02:128>>
  // CHECK-SAME:        tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>
  // CHECK-SAME:        tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>
  // CHECK:           return %[[ADD]]
  // CHECK-NOT:       onnx.XFEResize
}

// -----

// =============================================================================
// Negative: 1x1 spatial input but different quant scale on input vs output.
// The pass requires matching scale and zero_point, so the resize must survive.
// =============================================================================
// CHECK-LABEL: func.func @resize_1x1_to_2x2_mismatched_quant
func.func @resize_1x1_to_2x2_mismatched_quant(
    %arg0: tensor<1x1x1x128x!quant.uniform<u8:f32, 3.125000e-02:128>>)
    -> tensor<1x2x2x128x!quant.uniform<u8:f32, 1.562500e-02:64>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "pytorch_half_pixel",
    mode = "linear"
  } : (tensor<1x1x1x128x!quant.uniform<u8:f32, 3.125000e-02:128>>, none, tensor<4xf32>, none)
   -> tensor<1x2x2x128x!quant.uniform<u8:f32, 1.562500e-02:64>>
  return %0 : tensor<1x2x2x128x!quant.uniform<u8:f32, 1.562500e-02:64>>

  // CHECK:           "onnx.XFEResize"
  // CHECK-NOT:       "onnx.Add"
}

// -----

// =============================================================================
// Negative: non-1x1 spatial input (4x4 -> 8x8). The Resize is doing real
// interpolation across multiple source pixels and cannot be expressed as a
// broadcast, so it must survive.
// =============================================================================
// CHECK-LABEL: func.func @resize_4x4_to_8x8_stays
func.func @resize_4x4_to_8x8_stays(
    %arg0: tensor<1x4x4x32x!quant.uniform<u8:f32, 3.125000e-02:128>>)
    -> tensor<1x8x8x32x!quant.uniform<u8:f32, 3.125000e-02:128>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "pytorch_half_pixel",
    mode = "linear"
  } : (tensor<1x4x4x32x!quant.uniform<u8:f32, 3.125000e-02:128>>, none, tensor<4xf32>, none)
   -> tensor<1x8x8x32x!quant.uniform<u8:f32, 3.125000e-02:128>>
  return %0 : tensor<1x8x8x32x!quant.uniform<u8:f32, 3.125000e-02:128>>

  // CHECK:           "onnx.XFEResize"
  // CHECK-NOT:       "onnx.Add"
}

// -----

// =============================================================================
// Negative: 1x1 spatial input, matching quant types, but the resize result has
// two consumers. Replacing it would force creating two large zero constants
// (or shared one) and complicate downstream eltwise fusion; the pass requires
// single-use, mirroring xcompiler's fanout_num == 1 filter.
// =============================================================================
// CHECK-LABEL: func.func @resize_1x1_multi_use_stays
func.func @resize_1x1_multi_use_stays(
    %arg0: tensor<1x1x1x128x!quant.uniform<u8:f32, 3.125000e-02:128>>,
    %arg1: tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>)
    -> tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "pytorch_half_pixel",
    mode = "linear"
  } : (tensor<1x1x1x128x!quant.uniform<u8:f32, 3.125000e-02:128>>, none, tensor<4xf32>, none)
   -> tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>
  %1 = "onnx.Add"(%0, %arg1) :
        (tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>,
         tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>)
     -> tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>
  %2 = "onnx.Add"(%0, %1) :
        (tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>,
         tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>)
     -> tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>
  return %2 : tensor<1x2x2x128x!quant.uniform<u8:f32, 3.125000e-02:128>>

  // CHECK:           "onnx.XFEResize"
}

// -----

// =============================================================================
// Negative: 1x1 spatial input but float (non-quantized) types. The pass only
// targets QDQ form; float Resize is handled elsewhere.
// =============================================================================
// CHECK-LABEL: func.func @resize_1x1_float_stays
func.func @resize_1x1_float_stays(%arg0: tensor<1x1x1x128xf32>)
    -> tensor<1x2x2x128xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%arg0, %none, %scales, %none) {
    coordinate_transformation_mode = "pytorch_half_pixel",
    mode = "linear"
  } : (tensor<1x1x1x128xf32>, none, tensor<4xf32>, none) -> tensor<1x2x2x128xf32>
  return %0 : tensor<1x2x2x128xf32>

  // CHECK:           "onnx.XFEResize"
  // CHECK-NOT:       "onnx.Add"
}
