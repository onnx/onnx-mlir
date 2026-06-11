// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --batch-reduction-to-reshape-reduction %s | FileCheck %s
// -----
func.func @test_batch_reduction(%arg0: tensor<16x16x300x4xf32>) -> tensor<16x16x300xf32> {
  %0 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<0> : tensor<i8>
  %2 = onnx.Constant dense<3> : tensor<1xi64>
  %3 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<16x16x300x4xf32>, tensor<f32>, tensor<i8>) -> tensor<16x16x300x4x!quant.uniform<u8:f32, 0.10000000149011612>>
  %4 = "onnx.ReduceSum"(%3, %2) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<16x16x300x4x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<1xi64>) -> tensor<16x16x300x!quant.uniform<u8:f32, 0.10000000149011612>>
  %5 = "onnx.DequantizeLinear"(%4, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<16x16x300x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<f32>, tensor<i8>) -> tensor<16x16x300xf32>
  return %5 : tensor<16x16x300xf32>
}

// CHECK-LABEL: func.func @test_batch_reduction
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<16x16x300x4xf32>) -> tensor<16x16x300xf32>

// All constants (order independent)
// CHECK-DAG: %[[POST_SHAPE:.*]] = onnx.Constant dense<[16, 16, 300]> : tensor<3xi64>
// CHECK-DAG: %[[PRE_SHAPE:.*]] = onnx.Constant dense<[1, 256, 300, 4]> : tensor<4xi64>
// CHECK-DAG: %[[SCALE:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<0> : tensor<i8>
// CHECK-DAG: %[[AXES:.*]] = onnx.Constant dense<3> : tensor<1xi64>

// Step 1: Quantize input (float -> quantized)
// CHECK: %[[QUANT:.*]] = "onnx.QuantizeLinear"(%[[ARG0]], %[[SCALE]], %[[ZP]])
// CHECK-SAME:    {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64}
// CHECK-SAME:    : (tensor<16x16x300x4xf32>, tensor<f32>, tensor<i8>)
// CHECK-SAME:    -> tensor<16x16x300x4x!quant.uniform<u8:f32, 0.10000000149011612>>

// Step 2: Pre-reshape (flatten batch: [16, 16, 300, 4] -> [1, 256, 300, 4])
// CHECK: %[[PRE_RESHAPE:.*]] = "onnx.Reshape"(%[[QUANT]], %[[PRE_SHAPE]])
// CHECK-SAME:    : (tensor<16x16x300x4x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<4xi64>)
// CHECK-SAME:    -> tensor<1x256x300x4x!quant.uniform<u8:f32, 0.10000000149011612>>

// Step 3: ReduceSum on reshaped tensor (reduces over axis 3)
// CHECK: %[[REDUCE:.*]] = "onnx.ReduceSum"(%[[PRE_RESHAPE]], %[[AXES]])
// CHECK-SAME:    {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
// CHECK-SAME:    : (tensor<1x256x300x4x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<1xi64>)
// CHECK-SAME:    -> tensor<1x256x300x!quant.uniform<u8:f32, 0.10000000149011612>>

// Step 4: Post-reshape (restore original shape: [1, 256, 300] -> [16, 16, 300])
// CHECK: %[[POST_RESHAPE:.*]] = "onnx.Reshape"(%[[REDUCE]], %[[POST_SHAPE]])
// CHECK-SAME:    : (tensor<1x256x300x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<3xi64>)
// CHECK-SAME:    -> tensor<16x16x300x!quant.uniform<u8:f32, 0.10000000149011612>>

// Step 5: Dequantize final result (quantized -> float)
// CHECK: %[[DEQUANT:.*]] = "onnx.DequantizeLinear"(%[[POST_RESHAPE]], %[[SCALE]], %[[ZP]])
// CHECK-SAME:    {axis = 1 : si64, block_size = 0 : si64}
// CHECK-SAME:    : (tensor<16x16x300x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<f32>, tensor<i8>)
// CHECK-SAME:    -> tensor<16x16x300xf32>

// CHECK: return %[[DEQUANT]] : tensor<16x16x300xf32>

// -----
// failing test case with input tensor is not 4d
func.func @test_batch_reduction_3d(%arg0: tensor<16x300x4xf32>) -> tensor<16x300xf32> {
  %0 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<0> : tensor<i8>
  %2 = onnx.Constant dense<2> : tensor<1xi64>
  %3 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<16x300x4xf32>, tensor<f32>, tensor<i8>) -> tensor<16x300x4x!quant.uniform<u8:f32, 0.10000000149011612>>
  %4 = "onnx.ReduceSum"(%3, %2) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<16x300x4x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<1xi64>) -> tensor<16x300x!quant.uniform<u8:f32, 0.10000000149011612>>
  %5 = "onnx.DequantizeLinear"(%4, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<16x300x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<f32>, tensor<i8>) -> tensor<16x300xf32>
  return %5 : tensor<16x300xf32>

}
// CHECK-LABEL: func.func @test_batch_reduction_3d
// CHECK-NOT: onnx.Reshape
// CHECK: %[[QUANT:.*]] = "onnx.QuantizeLinear"
// CHECK: %[[REDUCE:.*]] = "onnx.ReduceSum"(%[[QUANT]]
// CHECK: %[[DEQUANT:.*]] = "onnx.DequantizeLinear"(%[[REDUCE]]
// CHECK: return %[[DEQUANT]]

// -----
// keepdims = 1 : the new ReduceSum must keep the reduced axis as size-1,
// so its result type stays rank-4 ([1, 256, 300, 1]) and the post-reshape
// restores the original rank-4 output. Regression test for PSA2.
func.func @test_batch_reduction_keepdims_1(%arg0: tensor<16x16x300x4x!quant.uniform<u16:f32, 1.0e-4:31222>>) -> tensor<16x16x300x1x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %axes = onnx.Constant dense<3> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
    : (tensor<16x16x300x4x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<1xi64>)
    -> tensor<16x16x300x1x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<16x16x300x1x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @test_batch_reduction_keepdims_1
// CHECK:       %[[PRE:.+]] = "onnx.Reshape"(%arg0, {{.+}}) {{.*allowzero = 0.+}} : (tensor<16x16x300x4x!quant.uniform<u16:f32, 1.000000e-04:31222>>, tensor<4xi64>) -> tensor<1x256x300x4x!quant.uniform<u16:f32, 1.000000e-04:31222>>
// CHECK:       %[[R:.+]] = "onnx.ReduceSum"(%[[PRE]], {{.+}}) {{.*keepdims = 1.+}} : (tensor<1x256x300x4x!quant.uniform<u16:f32, 1.000000e-04:31222>>, tensor<1xi64>) -> tensor<1x256x300x1x!quant.uniform<u16:f32, 2.000000e-04:32629>>
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[R]], {{.+}}) {{.*allowzero = 0.+}} : (tensor<1x256x300x1x!quant.uniform<u16:f32, 2.000000e-04:32629>>, tensor<4xi64>) -> tensor<16x16x300x1x!quant.uniform<u16:f32, 2.000000e-04:32629>>
// CHECK:       return %[[POST]]

// -----
// dim0 == 1 : pass must not fire (no leading batch dim to flatten).
func.func @test_batch_reduction_dim0_is_one_unchanged(%arg0: tensor<1x16x300x4x!quant.uniform<u16:f32, 1.0e-4:31222>>) -> tensor<1x16x300x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %axes = onnx.Constant dense<3> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
    : (tensor<1x16x300x4x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<1xi64>)
    -> tensor<1x16x300x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<1x16x300x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @test_batch_reduction_dim0_is_one_unchanged
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"
// CHECK:       return
