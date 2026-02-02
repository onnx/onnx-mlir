// RUN: onnx-mlir-opt --split-input-file --merge-continuous-strided-slice %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// CHECK-LABEL: @merge_continuous_strided_slice

 func.func @merge_continuous_strided_slice(%arg0: tensor<3x4xf32> ) -> (tensor<1x1xf32>) {
    %0 = onnx.Constant dense<2.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<1> : tensor<i8>
    %2 = onnx.Constant dense<[0, 2]> : tensor<2xi64>
    %3 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %4 = onnx.Constant dense<[0, 1]> : tensor<2xi64>
    %5 = onnx.Constant dense<1> : tensor<2xi64>
    %6 = onnx.Constant dense<0> : tensor<2xi64>
    %7 = "onnx.QuantizeLinear"(%arg0, %0, %1) {
      axis = 1 : si64,
      block_size = 0 : si64,
      output_dtype = 0 : si64,
      saturate = 1 : si64} : (tensor<3x4xf32>, tensor<f32>, tensor<i8>) -> tensor<3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    %8 = "onnx.Slice"(%7, %2, %3, %4, %5) {onnx_node_name = "Slice_0"} : (tensor<3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x1x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    %9 = "onnx.Slice"(%8, %6, %5, %4, %5) {onnx_node_name = "Slice_1"} : (tensor<2x1x!quant.uniform<u8:f32, 0.20000000298023224:1>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    %10 = "onnx.DequantizeLinear"(%9, %0, %1) {
      axis = 1 : si64,
      block_size = 0 : si64} : (tensor<1x1x!quant.uniform<u8:f32, 0.20000000298023224:1>>, tensor<f32>, tensor<i8>) -> tensor<1x1xf32>
    return %10 : tensor<1x1xf32>
 }

// Verify that %8 and %9 slice nodes are replaced by a single slice node
// CHECK-NOT: %8 = "onnx.Slice"(%7
// CHECK-NOT: %9 = "onnx.Slice"(%8
// Verify constants for the merged slice: [0,1], [1,1], [1,3], [0,2]
// CHECK-DAG: %[[AXES:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi32>
// CHECK-DAG: %[[STEPS:.*]] = onnx.Constant dense<1> : tensor<2xi32>
// CHECK-DAG: %[[ENDS:.*]] = onnx.Constant dense<[1, 3]> : tensor<2xi32>
// CHECK-DAG: %[[STARTS:.*]] = onnx.Constant dense<[0, 2]> : tensor<2xi32>
// Verify scale and zeropoint constants
// CHECK-DAG: %[[SCALE:.*]] = onnx.Constant dense<2.000000e-01> : tensor<f32>
// CHECK-DAG: %[[ZEROPOINT:.*]] = onnx.Constant dense<1> : tensor<i8>
// Verify QuantizeLinear produces quantized tensor with scale = %[[SCALE]] and zeropoint = %[[ZEROPOINT]]
// CHECK: %[[QUANTIZED:.*]] = "onnx.QuantizeLinear"(%arg0, %[[SCALE]], %[[ZEROPOINT]])
// CHECK-SAME: tensor<3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
// Verify the merged slice node with inputs: <input to first slice node>, [0,2], [1,3], [0, 1], [1,1]
// Verify the input and output tensors are quantized types with scale = %[[SCALE]] and zeropoint = %[[ZEROPOINT]]
// CHECK: %[[MERGED_SLICE:.*]] = "onnx.Slice"(%[[QUANTIZED]], %[[STARTS]], %[[ENDS]], %[[AXES]], %[[STEPS]])
// CHECK-SAME: (tensor<3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x1x!quant.uniform<u8:f32, 0.20000000298023224:1>>
// Verify DequantizeLinear uses the merged slice output with scale = %[[SCALE]] and zeropoint = %[[ZEROPOINT]]
// CHECK: "onnx.DequantizeLinear"(%[[MERGED_SLICE]], %[[SCALE]], %[[ZEROPOINT]])

// -----

// Test case where the same input goes to both a slice chain and a reshape
// A -> slice -> slice -> slice
// A -> reshape

// CHECK-LABEL: @merge_slice_chain_with_reshape_branch
func.func @merge_slice_chain_with_reshape_branch(%arg0: tensor<4x8xf32>) -> (tensor<1x2xf32>, tensor<32xf32>) {
   %scale = onnx.Constant dense<1.000000e-01> : tensor<f32>
   %zp = onnx.Constant dense<0> : tensor<i8>

   // Constants for slice operations
   %starts0 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends0 = onnx.Constant dense<[3, 6]> : tensor<2xi64>
   %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
   %steps = onnx.Constant dense<1> : tensor<2xi64>

   %starts1 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends1 = onnx.Constant dense<[2, 4]> : tensor<2xi64>

   %starts2 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends2 = onnx.Constant dense<[1, 2]> : tensor<2xi64>

   // Reshape shape constant
   %shape = onnx.Constant dense<32> : tensor<1xi64>

   // Quantize the input
   %quant = "onnx.QuantizeLinear"(%arg0, %scale, %zp) {
     axis = 1 : si64,
     block_size = 0 : si64,
     output_dtype = 0 : si64,
     saturate = 1 : si64} : (tensor<4x8xf32>, tensor<f32>, tensor<i8>) -> tensor<4x8x!quant.uniform<u8:f32, 0.10000000149011612>>

   // Slice chain: slice -> slice -> slice
   %slice0 = "onnx.Slice"(%quant, %starts0, %ends0, %axes, %steps) {onnx_node_name = "Slice_0"} : (tensor<4x8x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x6x!quant.uniform<u8:f32, 0.10000000149011612>>
   %slice1 = "onnx.Slice"(%slice0, %starts1, %ends1, %axes, %steps) {onnx_node_name = "Slice_1"} : (tensor<3x6x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x4x!quant.uniform<u8:f32, 0.10000000149011612>>
   %slice2 = "onnx.Slice"(%slice1, %starts2, %ends2, %axes, %steps) {onnx_node_name = "Slice_2"} : (tensor<2x4x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2x!quant.uniform<u8:f32, 0.10000000149011612>>

   // Reshape branch using the same quantized input
   %reshape = "onnx.Reshape"(%quant, %shape) {allowzero = 0 : si64} : (tensor<4x8x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<1xi64>) -> tensor<32x!quant.uniform<u8:f32, 0.10000000149011612>>

   // Dequantize outputs
   %dequant_slice = "onnx.DequantizeLinear"(%slice2, %scale, %zp) {
     axis = 1 : si64,
     block_size = 0 : si64} : (tensor<1x2x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<f32>, tensor<i8>) -> tensor<1x2xf32>

   %dequant_reshape = "onnx.DequantizeLinear"(%reshape, %scale, %zp) {
     axis = 0 : si64,
     block_size = 0 : si64} : (tensor<32x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<f32>, tensor<i8>) -> tensor<32xf32>

   return %dequant_slice, %dequant_reshape : tensor<1x2xf32>, tensor<32xf32>
}

// Verify that the three slice nodes are merged into one
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_0
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_1
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_2

// Verify constants for the merged slice
// CHECK-DAG: %[[AXES2:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi32>
// CHECK-DAG: %[[STEPS2:.*]] = onnx.Constant dense<1> : tensor<2xi32>
// CHECK-DAG: %[[ENDS2:.*]] = onnx.Constant dense<[1, 2]> : tensor<2xi32>
// CHECK-DAG: %[[STARTS2:.*]] = onnx.Constant dense<0> : tensor<2xi32>
// CHECK-DAG: %[[SCALE2:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG: %[[ZP2:.*]] = onnx.Constant dense<0> : tensor<i8>
// CHECK-DAG: %[[SHAPE:.*]] = onnx.Constant dense<32> : tensor<1xi64>

// Verify QuantizeLinear
// CHECK: %[[QUANT2:.*]] = "onnx.QuantizeLinear"(%arg0, %[[SCALE2]], %[[ZP2]])

// Verify merged slice replaces the chain
// CHECK: %[[MERGED:.*]] = "onnx.Slice"(%[[QUANT2]], %[[STARTS2]], %[[ENDS2]], %[[AXES2]], %[[STEPS2]])
// CHECK-SAME: -> tensor<1x2x!quant.uniform<u8:f32, 0.10000000149011612>>

// Verify reshape still uses the original quantized input
// CHECK: %[[RESHAPED:.*]] = "onnx.Reshape"(%[[QUANT2]], %[[SHAPE]])
// CHECK-SAME: -> tensor<32x!quant.uniform<u8:f32, 0.10000000149011612>>

// Verify both dequantize operations
// CHECK: "onnx.DequantizeLinear"(%[[MERGED]], %[[SCALE2]], %[[ZP2]])
// CHECK: "onnx.DequantizeLinear"(%[[RESHAPED]], %[[SCALE2]], %[[ZP2]])

// -----

// Test case where an intermediate slice in a chain has a branch to reshape
// A -> slice0 -> slice1 -> slice2 -> slice3 -> slice4
//                          slice2 -> reshape

// CHECK-LABEL: @merge_slice_chain_with_intermediate_branch
func.func @merge_slice_chain_with_intermediate_branch(%arg0: tensor<16x16xf32>) -> (tensor<1x1xf32>, tensor<16xf32>) {
   %scale = onnx.Constant dense<5.000000e-02> : tensor<f32>
   %zp = onnx.Constant dense<64> : tensor<i8>

   // Constants for slice operations
   %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
   %steps = onnx.Constant dense<1> : tensor<2xi64>

   %starts0 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends0 = onnx.Constant dense<[12, 12]> : tensor<2xi64>

   %starts1 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends1 = onnx.Constant dense<[8, 8]> : tensor<2xi64>

   %starts2 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends2 = onnx.Constant dense<[4, 4]> : tensor<2xi64>

   %starts3 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends3 = onnx.Constant dense<[2, 2]> : tensor<2xi64>

   %starts4 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends4 = onnx.Constant dense<[1, 1]> : tensor<2xi64>

   // Reshape shape constant
   %shape = onnx.Constant dense<16> : tensor<1xi64>

   // Quantize the input
   %quant = "onnx.QuantizeLinear"(%arg0, %scale, %zp) {
     axis = 1 : si64,
     block_size = 0 : si64,
     output_dtype = 0 : si64,
     saturate = 1 : si64} : (tensor<16x16xf32>, tensor<f32>, tensor<i8>) -> tensor<16x16x!quant.uniform<u8:f32, 0.05000000074505806:64>>

   // Slice chain: slice0 -> slice1 -> slice2 -> slice3 -> slice4
   %slice0 = "onnx.Slice"(%quant, %starts0, %ends0, %axes, %steps) {onnx_node_name = "Slice_0"} : (tensor<16x16x!quant.uniform<u8:f32, 0.05000000074505806:64>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<12x12x!quant.uniform<u8:f32, 0.05000000074505806:64>>
   %slice1 = "onnx.Slice"(%slice0, %starts1, %ends1, %axes, %steps) {onnx_node_name = "Slice_1"} : (tensor<12x12x!quant.uniform<u8:f32, 0.05000000074505806:64>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<8x8x!quant.uniform<u8:f32, 0.05000000074505806:64>>
   %slice2 = "onnx.Slice"(%slice1, %starts2, %ends2, %axes, %steps) {onnx_node_name = "Slice_2"} : (tensor<8x8x!quant.uniform<u8:f32, 0.05000000074505806:64>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x4x!quant.uniform<u8:f32, 0.05000000074505806:64>>
   %slice3 = "onnx.Slice"(%slice2, %starts3, %ends3, %axes, %steps) {onnx_node_name = "Slice_3"} : (tensor<4x4x!quant.uniform<u8:f32, 0.05000000074505806:64>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x2x!quant.uniform<u8:f32, 0.05000000074505806:64>>
   %slice4 = "onnx.Slice"(%slice3, %starts4, %ends4, %axes, %steps) {onnx_node_name = "Slice_4"} : (tensor<2x2x!quant.uniform<u8:f32, 0.05000000074505806:64>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x!quant.uniform<u8:f32, 0.05000000074505806:64>>

   // Reshape branch using slice2 output (intermediate slice)
   %reshape = "onnx.Reshape"(%slice2, %shape) {allowzero = 0 : si64} : (tensor<4x4x!quant.uniform<u8:f32, 0.05000000074505806:64>>, tensor<1xi64>) -> tensor<16x!quant.uniform<u8:f32, 0.05000000074505806:64>>

   // Dequantize outputs
   %dequant_slice = "onnx.DequantizeLinear"(%slice4, %scale, %zp) {
     axis = 1 : si64,
     block_size = 0 : si64} : (tensor<1x1x!quant.uniform<u8:f32, 0.05000000074505806:64>>, tensor<f32>, tensor<i8>) -> tensor<1x1xf32>

   %dequant_reshape = "onnx.DequantizeLinear"(%reshape, %scale, %zp) {
     axis = 0 : si64,
     block_size = 0 : si64} : (tensor<16x!quant.uniform<u8:f32, 0.05000000074505806:64>>, tensor<f32>, tensor<i8>) -> tensor<16xf32>

   return %dequant_slice, %dequant_reshape : tensor<1x1xf32>, tensor<16xf32>
}

// Verify that slice0, slice1, slice2 are merged into one (producing 4x4)
// and slice3, slice4 are merged into another (producing 1x1)
// The pattern can merge slice0+slice1+slice2 because when matching on slice2,
// it builds backwards and all previous slices have single uses.
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_0
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_1
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_2
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_3
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_4

// Verify constants for merged slices
// CHECK-DAG: %[[AXES3:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi32>
// CHECK-DAG: %[[STEPS3:.*]] = onnx.Constant dense<1> : tensor<2xi32>
// CHECK-DAG: %[[SCALE3:.*]] = onnx.Constant dense<5.000000e-02> : tensor<f32>
// CHECK-DAG: %[[ZP3:.*]] = onnx.Constant dense<64> : tensor<i8>
// CHECK-DAG: %[[SHAPE3:.*]] = onnx.Constant dense<16> : tensor<1xi64>

// Verify QuantizeLinear
// CHECK: %[[QUANT3:.*]] = "onnx.QuantizeLinear"(%arg0, %[[SCALE3]], %[[ZP3]])

// Verify first merged slice (slice0 + slice1 + slice2) produces 4x4 tensor
// CHECK: %[[MERGED_FIRST:.*]] = "onnx.Slice"(%[[QUANT3]]
// CHECK-SAME: -> tensor<4x4x!quant.uniform<u8:f32, 0.05000000074505806:64>>

// Verify second merged slice (slice3 + slice4) produces 1x1 tensor
// CHECK: %[[MERGED_SECOND:.*]] = "onnx.Slice"(%[[MERGED_FIRST]]
// CHECK-SAME: -> tensor<1x1x!quant.uniform<u8:f32, 0.05000000074505806:64>>

// Verify reshape uses the first merged slice output (equivalent to slice2)
// CHECK: %[[RESHAPED3:.*]] = "onnx.Reshape"(%[[MERGED_FIRST]], %[[SHAPE3]])
// CHECK-SAME: -> tensor<16x!quant.uniform<u8:f32, 0.05000000074505806:64>>

// Verify dequantize operations
// CHECK-DAG: "onnx.DequantizeLinear"(%[[MERGED_SECOND]], %[[SCALE3]], %[[ZP3]])
// CHECK-DAG: "onnx.DequantizeLinear"(%[[RESHAPED3]], %[[SCALE3]], %[[ZP3]])

// -----

// Test case where the final slice in a chain branches to both reshape and transpose
// input -> slice0 -> slice1 -> slice2 -> reshape
//                               slice2 -> transpose

// CHECK-LABEL: @merge_slice_chain_with_reshape_and_transpose
func.func @merge_slice_chain_with_reshape_and_transpose(%arg0: tensor<8x8xf32>) -> (tensor<4xf32>, tensor<2x2xf32>) {
   %scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
   %zp = onnx.Constant dense<128> : tensor<i8>

   // Constants for slice operations
   %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
   %steps = onnx.Constant dense<1> : tensor<2xi64>

   %starts0 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends0 = onnx.Constant dense<[6, 6]> : tensor<2xi64>

   %starts1 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends1 = onnx.Constant dense<[4, 4]> : tensor<2xi64>

   %starts2 = onnx.Constant dense<[0, 0]> : tensor<2xi64>
   %ends2 = onnx.Constant dense<[2, 2]> : tensor<2xi64>

   // Reshape shape constant
   %shape = onnx.Constant dense<4> : tensor<1xi64>

   // Quantize the input
   %quant = "onnx.QuantizeLinear"(%arg0, %scale, %zp) {
     axis = 1 : si64,
     block_size = 0 : si64,
     output_dtype = 0 : si64,
     saturate = 1 : si64} : (tensor<8x8xf32>, tensor<f32>, tensor<i8>) -> tensor<8x8x!quant.uniform<u8:f32, 0.25:128>>

   // Slice chain: slice0 -> slice1 -> slice2
   %slice0 = "onnx.Slice"(%quant, %starts0, %ends0, %axes, %steps) {onnx_node_name = "Slice_0"} : (tensor<8x8x!quant.uniform<u8:f32, 0.25:128>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<6x6x!quant.uniform<u8:f32, 0.25:128>>
   %slice1 = "onnx.Slice"(%slice0, %starts1, %ends1, %axes, %steps) {onnx_node_name = "Slice_1"} : (tensor<6x6x!quant.uniform<u8:f32, 0.25:128>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x4x!quant.uniform<u8:f32, 0.25:128>>
   %slice2 = "onnx.Slice"(%slice1, %starts2, %ends2, %axes, %steps) {onnx_node_name = "Slice_2"} : (tensor<4x4x!quant.uniform<u8:f32, 0.25:128>>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x2x!quant.uniform<u8:f32, 0.25:128>>

   // Reshape branch using slice2 output
   %reshape = "onnx.Reshape"(%slice2, %shape) {allowzero = 0 : si64} : (tensor<2x2x!quant.uniform<u8:f32, 0.25:128>>, tensor<1xi64>) -> tensor<4x!quant.uniform<u8:f32, 0.25:128>>

   // Transpose branch using slice2 output
   %transpose = "onnx.Transpose"(%slice2) {perm = [1, 0]} : (tensor<2x2x!quant.uniform<u8:f32, 0.25:128>>) -> tensor<2x2x!quant.uniform<u8:f32, 0.25:128>>

   // Dequantize outputs
   %dequant_reshape = "onnx.DequantizeLinear"(%reshape, %scale, %zp) {
     axis = 0 : si64,
     block_size = 0 : si64} : (tensor<4x!quant.uniform<u8:f32, 0.25:128>>, tensor<f32>, tensor<i8>) -> tensor<4xf32>

   %dequant_transpose = "onnx.DequantizeLinear"(%transpose, %scale, %zp) {
     axis = 1 : si64,
     block_size = 0 : si64} : (tensor<2x2x!quant.uniform<u8:f32, 0.25:128>>, tensor<f32>, tensor<i8>) -> tensor<2x2xf32>

   return %dequant_reshape, %dequant_transpose : tensor<4xf32>, tensor<2x2xf32>
}

// Verify that the three slice nodes are merged into one
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_0
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_1
// CHECK-NOT: "onnx.Slice"{{.*}}Slice_2

// Verify constants for the merged slice
// CHECK-DAG: %[[AXES4:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi32>
// CHECK-DAG: %[[STEPS4:.*]] = onnx.Constant dense<1> : tensor<2xi32>
// CHECK-DAG: %[[ENDS4:.*]] = onnx.Constant dense<2> : tensor<2xi32>
// CHECK-DAG: %[[STARTS4:.*]] = onnx.Constant dense<0> : tensor<2xi32>
// CHECK-DAG: %[[SCALE4:.*]] = onnx.Constant dense<2.500000e-01> : tensor<f32>
// CHECK-DAG: %[[ZP4:.*]] = onnx.Constant dense<-128> : tensor<i8>
// CHECK-DAG: %[[SHAPE4:.*]] = onnx.Constant dense<4> : tensor<1xi64>

// Verify QuantizeLinear
// CHECK: %[[QUANT4:.*]] = "onnx.QuantizeLinear"(%arg0, %[[SCALE4]], %[[ZP4]])

// Verify merged slice replaces the chain
// CHECK: %[[MERGED4:.*]] = "onnx.Slice"(%[[QUANT4]], %[[STARTS4]], %[[ENDS4]], %[[AXES4]], %[[STEPS4]])
// CHECK-SAME: -> tensor<2x2x!quant.uniform<u8:f32, 2.500000e-01:128>>

// Verify reshape uses the merged slice output
// CHECK: %[[RESHAPED4:.*]] = "onnx.Reshape"(%[[MERGED4]], %[[SHAPE4]])
// CHECK-SAME: -> tensor<4x!quant.uniform<u8:f32, 2.500000e-01:128>>

// Verify transpose uses the merged slice output
// CHECK: %[[TRANSPOSED4:.*]] = "onnx.Transpose"(%[[MERGED4]])
// CHECK-SAME: -> tensor<2x2x!quant.uniform<u8:f32, 2.500000e-01:128>>

// Verify dequantize operations
// CHECK-DAG: "onnx.DequantizeLinear"(%[[RESHAPED4]], %[[SCALE4]], %[[ZP4]])
// CHECK-DAG: "onnx.DequantizeLinear"(%[[TRANSPOSED4]], %[[SCALE4]], %[[ZP4]])
