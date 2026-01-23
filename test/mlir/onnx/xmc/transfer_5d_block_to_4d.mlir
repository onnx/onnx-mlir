// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --transfer-5d-block-to-4d %s | FileCheck %s

// Test case: Pattern 1 - Reshape(4D->5D) + Add + Concat(axis=4) + Transpose{0,3,4,1,2} + Reshape
// This pattern should be transformed to 4D operations by merging dimensions 1 and 2
//
// Input pattern:
//   input0 (4D) -> reshape0 (5D) -> add0 ---> concat(axis=4) -> transpose{0,3,4,1,2} -> reshape_out
//   input1 (4D) -> reshape1 (5D) -> add1 -/
//
// Output pattern (4D operations):
//   input0 (4D) -> reshape0 (4D) -> add0 ---> concat(axis=3) -> transpose{0,2,3,1} -> reshape_out
//   input1 (4D) -> reshape1 (4D) -> add1 -/

!qtype_in0 = !quant.uniform<i8:f32, 0.1:0>
!qtype_in1 = !quant.uniform<i8:f32, 0.2:1>
!qtype_add0 = !quant.uniform<i8:f32, 0.3:2>
!qtype_add1 = !quant.uniform<i8:f32, 0.4:3>
!qtype_concat1 = !quant.uniform<i8:f32, 0.5:4>
!qtype_transpose1 = !quant.uniform<i8:f32, 0.6:5>
!qtype_out1 = !quant.uniform<i8:f32, 0.7:6>

// CHECK-LABEL: @transfer_5d_block_with_transpose_basic
func.func @transfer_5d_block_with_transpose_basic(
    %input0: tensor<1x6x4x4x!qtype_in0>,
    %input1: tensor<1x6x4x4x!qtype_in1>,
    %bias0: tensor<!qtype_in0>,
    %bias1: tensor<!qtype_in1>
) -> tensor<1x4x8x6x!qtype_out1> {
    // Reshape from 4D [1,6,4,4] to 5D [1,2,3,4,4]
    %shape0 = onnx.Constant dense<[1, 2, 3, 4, 4]> : tensor<5xi64>
    %shape1 = onnx.Constant dense<[1, 2, 3, 4, 4]> : tensor<5xi64>

    %reshape0 = "onnx.Reshape"(%input0, %shape0) {allowzero = 0 : si64} : (tensor<1x6x4x4x!qtype_in0>, tensor<5xi64>) -> tensor<1x2x3x4x4x!qtype_in0>
    %reshape1 = "onnx.Reshape"(%input1, %shape1) {allowzero = 0 : si64} : (tensor<1x6x4x4x!qtype_in1>, tensor<5xi64>) -> tensor<1x2x3x4x4x!qtype_in1>

    // Element-wise Add operations on 5D tensors (scalar bias broadcasts)
    %add0 = "onnx.Add"(%reshape0, %bias0) : (tensor<1x2x3x4x4x!qtype_in0>, tensor<!qtype_in0>) -> tensor<1x2x3x4x4x!qtype_add0>
    %add1 = "onnx.Add"(%reshape1, %bias1) : (tensor<1x2x3x4x4x!qtype_in1>, tensor<!qtype_in1>) -> tensor<1x2x3x4x4x!qtype_add1>

    // Concat along axis 4: [1,2,3,4,4] + [1,2,3,4,4] -> [1,2,3,4,8]
    %concat = "onnx.Concat"(%add0, %add1) {axis = 4 : si64} : (tensor<1x2x3x4x4x!qtype_add0>, tensor<1x2x3x4x4x!qtype_add1>) -> tensor<1x2x3x4x8x!qtype_concat1>

    // Transpose with perm {0,3,4,1,2}: [1,2,3,4,8] -> [1,4,8,2,3]
    %transpose = "onnx.Transpose"(%concat) {perm = [0, 3, 4, 1, 2]} : (tensor<1x2x3x4x8x!qtype_concat1>) -> tensor<1x4x8x2x3x!qtype_transpose1>

    // Final reshape from 5D to target shape
    %output_shape = onnx.Constant dense<[1, 4, 8, 6]> : tensor<4xi64>
    %output = "onnx.Reshape"(%transpose, %output_shape) {allowzero = 0 : si64} : (tensor<1x4x8x2x3x!qtype_transpose1>, tensor<4xi64>) -> tensor<1x4x8x6x!qtype_out1>

    return %output : tensor<1x4x8x6x!qtype_out1>
}

// After transformation, 5D operations should become 4D:
// - Reshapes merge dims 1,2: [1,2,3,4,4] -> [1,6,4,4]
// - Concat axis changes from 4 to 3
// - Transpose perm changes from {0,3,4,1,2} to {0,2,3,1}

// CHECK-DAG: %[[SHAPE4D:.*]] = onnx.Constant dense<[1, 6, 4, 4]>
// CHECK-DAG: %[[OUTPUT_SHAPE:.*]] = onnx.Constant dense<[1, 4, 8, 6]>
// CHECK: %[[RESHAPE0:.*]] = "onnx.Reshape"(%arg0, %[[SHAPE4D]]){{.*}}-> tensor<1x6x4x4x!quant.uniform<i8:f32,{{.*}}>>
// CHECK: %[[RESHAPE1:.*]] = "onnx.Reshape"(%arg1, %[[SHAPE4D]]){{.*}}-> tensor<1x6x4x4x!quant.uniform<i8:f32,{{.*}}:1>>
// CHECK: %[[ADD0:.*]] = "onnx.Add"(%[[RESHAPE0]],{{.*}}-> tensor<1x6x4x4x!quant.uniform<i8:f32,{{.*}}:2>>
// CHECK: %[[ADD1:.*]] = "onnx.Add"(%[[RESHAPE1]],{{.*}}-> tensor<1x6x4x4x!quant.uniform<i8:f32,{{.*}}:3>>
// CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%[[ADD0]], %[[ADD1]]) {axis = 3 : si64}{{.*}}-> tensor<1x6x4x8x!quant.uniform<i8:f32,{{.*}}:4>>
// CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[CONCAT]]) {perm = [0, 2, 3, 1]}{{.*}}-> tensor<1x4x8x6x!quant.uniform<i8:f32,{{.*}}:5>>
// CHECK: %[[OUTPUT:.*]] = "onnx.Reshape"(%[[TRANSPOSE]], %[[OUTPUT_SHAPE]]){{.*}}-> tensor<1x4x8x6x!quant.uniform<i8:f32,{{.*}}:6>>
// CHECK: return %[[OUTPUT]]

// -----

//===----------------------------------------------------------------------===//
// Test case: Pattern 2 - Reshape(4D->5D) + Add + Concat(axis=2) + Reshape
// This pattern should be transformed to 4D operations by merging dimensions 3 and 4
//===----------------------------------------------------------------------===//

!qtype2_in0 = !quant.uniform<i8:f32, 0.11:10>
!qtype2_in1 = !quant.uniform<i8:f32, 0.12:11>
!qtype2_add0 = !quant.uniform<i8:f32, 0.13:12>
!qtype2_add1 = !quant.uniform<i8:f32, 0.14:13>
!qtype2_concat = !quant.uniform<i8:f32, 0.15:14>
!qtype2_out = !quant.uniform<i8:f32, 0.16:15>

// CHECK-LABEL: @transfer_5d_block_with_concat_axis2
func.func @transfer_5d_block_with_concat_axis2(
    %input0: tensor<1x2x3x12x!qtype2_in0>,
    %input1: tensor<1x2x4x12x!qtype2_in1>,
    %bias0: tensor<!qtype2_in0>,
    %bias1: tensor<!qtype2_in1>
) -> tensor<1x2x7x3x4x!qtype2_out> {
    // Reshape from 4D to 5D: [1,2,3,12] -> [1,2,3,3,4] (merging H*W=12 into 3x4)
    %shape0 = onnx.Constant dense<[1, 2, 3, 3, 4]> : tensor<5xi64>
    // Reshape from 4D to 5D: [1,2,4,12] -> [1,2,4,3,4]
    %shape1 = onnx.Constant dense<[1, 2, 4, 3, 4]> : tensor<5xi64>

    %reshape0 = "onnx.Reshape"(%input0, %shape0) {allowzero = 0 : si64} : (tensor<1x2x3x12x!qtype2_in0>, tensor<5xi64>) -> tensor<1x2x3x3x4x!qtype2_in0>
    %reshape1 = "onnx.Reshape"(%input1, %shape1) {allowzero = 0 : si64} : (tensor<1x2x4x12x!qtype2_in1>, tensor<5xi64>) -> tensor<1x2x4x3x4x!qtype2_in1>

    // Element-wise Add operations on 5D tensors
    %add0 = "onnx.Add"(%reshape0, %bias0) : (tensor<1x2x3x3x4x!qtype2_in0>, tensor<!qtype2_in0>) -> tensor<1x2x3x3x4x!qtype2_add0>
    %add1 = "onnx.Add"(%reshape1, %bias1) : (tensor<1x2x4x3x4x!qtype2_in1>, tensor<!qtype2_in1>) -> tensor<1x2x4x3x4x!qtype2_add1>

    // Concat along axis 2: [1,2,3,3,4] + [1,2,4,3,4] -> [1,2,7,3,4]
    %concat = "onnx.Concat"(%add0, %add1) {axis = 2 : si64} : (tensor<1x2x3x3x4x!qtype2_add0>, tensor<1x2x4x3x4x!qtype2_add1>) -> tensor<1x2x7x3x4x!qtype2_concat>

    // Final reshape (identity in this case)
    %output_shape = onnx.Constant dense<[1, 2, 7, 3, 4]> : tensor<5xi64>
    %output = "onnx.Reshape"(%concat, %output_shape) {allowzero = 0 : si64} : (tensor<1x2x7x3x4x!qtype2_concat>, tensor<5xi64>) -> tensor<1x2x7x3x4x!qtype2_out>

    return %output : tensor<1x2x7x3x4x!qtype2_out>
}

// CHECK-DAG: %[[SHAPE0_4D:.*]] = onnx.Constant dense<[1, 2, 3, 12]>
// CHECK-DAG: %[[SHAPE1_4D:.*]] = onnx.Constant dense<[1, 2, 4, 12]>
// CHECK-DAG: %[[OUTPUT_SHAPE:.*]] = onnx.Constant dense<[1, 2, 7, 3, 4]>
// CHECK: %[[RESHAPE0:.*]] = "onnx.Reshape"(%arg0, %[[SHAPE0_4D]]){{.*}}-> tensor<1x2x3x12x!quant.uniform<i8:f32,{{.*}}:10>>
// CHECK: %[[RESHAPE1:.*]] = "onnx.Reshape"(%arg1, %[[SHAPE1_4D]]){{.*}}-> tensor<1x2x4x12x!quant.uniform<i8:f32,{{.*}}:11>>
// CHECK: %[[ADD0:.*]] = "onnx.Add"(%[[RESHAPE0]],{{.*}}-> tensor<1x2x3x12x!quant.uniform<i8:f32,{{.*}}:12>>
// CHECK: %[[ADD1:.*]] = "onnx.Add"(%[[RESHAPE1]],{{.*}}-> tensor<1x2x4x12x!quant.uniform<i8:f32,{{.*}}:13>>
// CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%[[ADD0]], %[[ADD1]]) {axis = 2 : si64}{{.*}}-> tensor<1x2x7x12x!quant.uniform<i8:f32,{{.*}}:14>>
// CHECK: %[[OUTPUT:.*]] = "onnx.Reshape"(%[[CONCAT]], %[[OUTPUT_SHAPE]]){{.*}}-> tensor<1x2x7x3x4x!quant.uniform<i8:f32,{{.*}}:15>>
// CHECK: return %[[OUTPUT]]

// -----

//===----------------------------------------------------------------------===//
// Test case: Pattern 3 - Reshape + Transpose{0,3,4,1,2} + Mul + Reshape
// This pattern should be transformed to 4D operations
//===----------------------------------------------------------------------===//

!qtype3_in0 = !quant.uniform<i8:f32, 0.21:20>
!qtype3_in1 = !quant.uniform<i8:f32, 0.22:21>
!qtype3_transpose = !quant.uniform<i8:f32, 0.23:22>
!qtype3_mul = !quant.uniform<i8:f32, 0.24:23>
!qtype3_out = !quant.uniform<i8:f32, 0.25:24>

// CHECK-LABEL: @transfer_5d_mul_block
func.func @transfer_5d_mul_block(
    %input0: tensor<1x6x4x4x!qtype3_in0>,
    %input1: tensor<1x4x4x6x!qtype3_in1>
) -> tensor<1x4x4x6x!qtype3_out> {
    // Reshape from 4D [1,6,4,4] to 5D [1,2,3,4,4]
    %shape0 = onnx.Constant dense<[1, 2, 3, 4, 4]> : tensor<5xi64>
    // Reshape from 4D [1,4,4,6] to 5D [1,4,4,2,3] (matches transpose output shape)
    %shape1 = onnx.Constant dense<[1, 4, 4, 2, 3]> : tensor<5xi64>

    %reshape0 = "onnx.Reshape"(%input0, %shape0) {allowzero = 0 : si64} : (tensor<1x6x4x4x!qtype3_in0>, tensor<5xi64>) -> tensor<1x2x3x4x4x!qtype3_in0>
    %reshape1 = "onnx.Reshape"(%input1, %shape1) {allowzero = 0 : si64} : (tensor<1x4x4x6x!qtype3_in1>, tensor<5xi64>) -> tensor<1x4x4x2x3x!qtype3_in1>

    // Transpose with perm {0,3,4,1,2}: [1,2,3,4,4] -> [1,4,4,2,3]
    %transpose = "onnx.Transpose"(%reshape0) {perm = [0, 3, 4, 1, 2]} : (tensor<1x2x3x4x4x!qtype3_in0>) -> tensor<1x4x4x2x3x!qtype3_transpose>

    // Element-wise Mul: [1,4,4,2,3] * [1,4,4,2,3] -> [1,4,4,2,3]
    %mul = "onnx.Mul"(%transpose, %reshape1) : (tensor<1x4x4x2x3x!qtype3_transpose>, tensor<1x4x4x2x3x!qtype3_in1>) -> tensor<1x4x4x2x3x!qtype3_mul>

    // Final reshape from 5D to target shape
    %output_shape = onnx.Constant dense<[1, 4, 4, 6]> : tensor<4xi64>
    %output = "onnx.Reshape"(%mul, %output_shape) {allowzero = 0 : si64} : (tensor<1x4x4x2x3x!qtype3_mul>, tensor<4xi64>) -> tensor<1x4x4x6x!qtype3_out>

    return %output : tensor<1x4x4x6x!qtype3_out>
}

// CHECK-DAG: %[[SHAPE0_4D:.*]] = onnx.Constant dense<[1, 6, 4, 4]>
// CHECK-DAG: %[[SHAPE1_4D:.*]] = onnx.Constant dense<[1, 4, 4, 6]>
// CHECK: %[[RESHAPE0:.*]] = "onnx.Reshape"(%arg0, %[[SHAPE0_4D]]){{.*}}-> tensor<1x6x4x4x!quant.uniform<i8:f32,{{.*}}:20>>
// CHECK: %[[RESHAPE1:.*]] = "onnx.Reshape"(%arg1, %[[SHAPE1_4D]]){{.*}}-> tensor<1x4x4x6x!quant.uniform<i8:f32,{{.*}}:21>>
// CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[RESHAPE0]]) {perm = [0, 2, 3, 1]}{{.*}}-> tensor<1x4x4x6x!quant.uniform<i8:f32,{{.*}}:22>>
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[TRANSPOSE]], %[[RESHAPE1]]){{.*}}-> tensor<1x4x4x6x!quant.uniform<i8:f32,{{.*}}:23>>
// CHECK: %[[OUTPUT:.*]] = "onnx.Reshape"(%[[MUL]], %[[SHAPE1_4D]]){{.*}}-> tensor<1x4x4x6x!quant.uniform<i8:f32,{{.*}}:24>>
// CHECK: return %[[OUTPUT]]

