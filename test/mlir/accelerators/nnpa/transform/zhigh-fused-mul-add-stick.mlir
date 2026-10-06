// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// Tests for the zhigh.mul-add-stick FusedOp pattern. The pass wraps the DAG
//   ONNXMulOp, ONNXMulOp -> ONNXAddOp | ONNXSubOp -> [ONNXMulOp by scalar]
//     -> ONNXReshapeOp (rank 3) -> ZHighStickOp (3D / 3DS)
// into an onnx.Fused region with kind = "zhigh.mul-add-stick". The scalar Mul
// is optional; when absent, mulScalar is stored as its neutral 1.0 default.

// -----

// The rotary embedding of granite-embedding-97m-multilingual-r2:
// (x * cos + rotate_half(x) * sin) * 32^-1/4, reshaped to (B * 12, S, 32)
// and stickified as 3DS. The dynamic batch and sequence dims are tied by the
// onnx.dim_params names and the Dim-based Reshape shape, as in the model.
func.func @mul_add_stick_rotary(
    %x1: tensor<?x12x?x32xf32> {onnx.dim_params = "0:batch,2:seq"},
    %x2: tensor<?x12x?x32xf32> {onnx.dim_params = "0:batch,2:seq"},
    %cos: tensor<1x1x?x32xf32> {onnx.dim_params = "2:seq"},
    %sin: tensor<1x1x?x32xf32> {onnx.dim_params = "2:seq"})
    -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %k = onnx.Constant dense<0.420448214> : tensor<1xf32>
  %m1 = onnx.Constant dense<-1> : tensor<1xi64>
  %c32 = onnx.Constant dense<32> : tensor<1xi64>
  %seq = "onnx.Dim"(%x1) <{axis = 2 : si64}> : (tensor<?x12x?x32xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%m1, %seq, %c32) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>) -> tensor<?x12x?x32xf32>
  %3 = "onnx.Mul"(%2, %k) : (tensor<?x12x?x32xf32>, tensor<1xf32>) -> tensor<?x12x?x32xf32>
  %4 = "onnx.Reshape"(%3, %shape) <{allowzero = 0 : si64}> : (tensor<?x12x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
  %5 = "zhigh.Stick"(%4) <{layout = "3DS"}> : (tensor<?x?x32xf32>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %5 : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @mul_add_stick_rotary
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x12x?x32xf32> {onnx.dim_params = "0:batch,2:seq"}, [[PARAM_1_:%.+]]: tensor<?x12x?x32xf32> {onnx.dim_params = "0:batch,2:seq"}, [[PARAM_2_:%.+]]: tensor<1x1x?x32xf32> {onnx.dim_params = "2:seq"}, [[PARAM_3_:%.+]]: tensor<1x1x?x32xf32> {onnx.dim_params = "2:seq"}) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Dim"([[PARAM_0_]]) <{axis = 2 : si64}> : (tensor<?x12x?x32xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_2_]], [[VAR_1_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Fused"([[PARAM_0_]], [[PARAM_2_]], [[PARAM_1_]], [[PARAM_3_]], [[VAR_3_]]) <{kind = "zhigh.mul-add-stick"}> ({
// CHECK:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<?x12x?x32xf32>, [[REGION_ARG_1_:%.+]]: tensor<1x1x?x32xf32>, [[REGION_ARG_2_:%.+]]: tensor<?x12x?x32xf32>, [[REGION_ARG_3_:%.+]]: tensor<1x1x?x32xf32>, [[REGION_ARG_4_:%.+]]: tensor<3xi64>):
// CHECK-DAG:         [[VAR_5_:%.+]] = onnx.Constant dense<0.420448214> : tensor<1xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = "onnx.Mul"([[REGION_ARG_0_]], [[REGION_ARG_1_]]) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
// CHECK-DAG:         [[VAR_7_:%.+]] = "onnx.Mul"([[REGION_ARG_2_]], [[REGION_ARG_3_]]) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
// CHECK:             [[VAR_8_:%.+]] = "onnx.Add"([[VAR_6_]], [[VAR_7_]]) : (tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>) -> tensor<?x12x?x32xf32>
// CHECK:             [[VAR_9_:%.+]] = "onnx.Mul"([[VAR_8_]], [[VAR_5_]]) : (tensor<?x12x?x32xf32>, tensor<1xf32>) -> tensor<?x12x?x32xf32>
// CHECK:             [[VAR_10_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[REGION_ARG_4_]]) <{allowzero = 0 : si64}> : (tensor<?x12x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
// CHECK:             [[VAR_11_:%.+]] = "zhigh.Stick"([[VAR_10_]]) <{layout = "3DS"}> : (tensor<?x?x32xf32>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:             onnx.Yield [[VAR_11_]] : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }) {isSub = false, mulScalar = 0.420448214 : f32, onnx_node_name = "onnx.Mul-onnx.Mul-onnx.Add-onnx.Mul-onnx.Reshape-zhigh.Stick", reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS"} : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>, tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_4_]] : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// Sub join, no scalar Mul, swapped operand order, D = 64, 3D layout.
func.func @mul_sub_stick_no_scalar(%x1: tensor<2x4x8x64xf32>, %x2: tensor<2x4x8x64xf32>,
    %cos: tensor<8x64xf32>, %sin: tensor<8x64xf32>)
    -> tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>> {
  %shape = onnx.Constant dense<[8, 8, 64]> : tensor<3xi64>
  %0 = "onnx.Mul"(%cos, %x1) : (tensor<8x64xf32>, tensor<2x4x8x64xf32>) -> tensor<2x4x8x64xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<2x4x8x64xf32>, tensor<8x64xf32>) -> tensor<2x4x8x64xf32>
  %2 = "onnx.Sub"(%0, %1) : (tensor<2x4x8x64xf32>, tensor<2x4x8x64xf32>) -> tensor<2x4x8x64xf32>
  %3 = "onnx.Reshape"(%2, %shape) <{allowzero = 0 : si64}> : (tensor<2x4x8x64xf32>, tensor<3xi64>) -> tensor<8x8x64xf32>
  %4 = "zhigh.Stick"(%3) <{layout = "3D"}> : (tensor<8x8x64xf32>) -> tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>>
  return %4 : tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @mul_sub_stick_no_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8x64xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8x64xf32>, [[PARAM_2_:%.+]]: tensor<8x64xf32>, [[PARAM_3_:%.+]]: tensor<8x64xf32>) -> tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_2_]], [[PARAM_0_]], [[PARAM_1_]], [[PARAM_3_]]) <{kind = "zhigh.mul-add-stick"}> ({
// CHECK:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<8x64xf32>, [[REGION_ARG_1_:%.+]]: tensor<2x4x8x64xf32>, [[REGION_ARG_2_:%.+]]: tensor<2x4x8x64xf32>, [[REGION_ARG_3_:%.+]]: tensor<8x64xf32>):
// CHECK-DAG:         [[VAR_1_:%.+]] = onnx.Constant dense<[8, 8, 64]> : tensor<3xi64>
// CHECK-DAG:         [[VAR_2_:%.+]] = "onnx.Mul"([[REGION_ARG_0_]], [[REGION_ARG_1_]]) : (tensor<8x64xf32>, tensor<2x4x8x64xf32>) -> tensor<2x4x8x64xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = "onnx.Mul"([[REGION_ARG_2_]], [[REGION_ARG_3_]]) : (tensor<2x4x8x64xf32>, tensor<8x64xf32>) -> tensor<2x4x8x64xf32>
// CHECK:             [[VAR_4_:%.+]] = "onnx.Sub"([[VAR_2_]], [[VAR_3_]]) : (tensor<2x4x8x64xf32>, tensor<2x4x8x64xf32>) -> tensor<2x4x8x64xf32>
// CHECK:             [[VAR_5_:%.+]] = "onnx.Reshape"([[VAR_4_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<2x4x8x64xf32>, tensor<3xi64>) -> tensor<8x8x64xf32>
// CHECK:             [[VAR_6_:%.+]] = "zhigh.Stick"([[VAR_5_]]) <{layout = "3D"}> : (tensor<8x8x64xf32>) -> tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:             onnx.Yield [[VAR_6_]] : tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           }) {isSub = true, mulScalar = 1.000000e+00 : f32, onnx_node_name = "onnx.Mul-onnx.Mul-onnx.Sub-onnx.Reshape-zhigh.Stick", reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3D"} : (tensor<8x64xf32>, tensor<2x4x8x64xf32>, tensor<2x4x8x64xf32>, tensor<8x64xf32>) -> tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

// Rank 3 join with a no-op Reshape; scalar Mul with the scalar first.
func.func @mul_add_stick_noop_reshape(%x1: tensor<24x8x32xf32>, %x2: tensor<24x8x32xf32>,
    %cos: tensor<1x8x32xf32>, %sin: tensor<1x8x32xf32>)
    -> tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %shape = onnx.Constant dense<[24, 8, 32]> : tensor<3xi64>
  %k = onnx.Constant dense<2.0> : tensor<f32>
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<24x8x32xf32>, tensor<1x8x32xf32>) -> tensor<24x8x32xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<24x8x32xf32>, tensor<1x8x32xf32>) -> tensor<24x8x32xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<24x8x32xf32>, tensor<24x8x32xf32>) -> tensor<24x8x32xf32>
  %3 = "onnx.Mul"(%k, %2) : (tensor<f32>, tensor<24x8x32xf32>) -> tensor<24x8x32xf32>
  %4 = "onnx.Reshape"(%3, %shape) <{allowzero = 0 : si64}> : (tensor<24x8x32xf32>, tensor<3xi64>) -> tensor<24x8x32xf32>
  %5 = "zhigh.Stick"(%4) <{layout = "3DS"}> : (tensor<24x8x32xf32>) -> tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %5 : tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @mul_add_stick_noop_reshape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<24x8x32xf32>, [[PARAM_1_:%.+]]: tensor<24x8x32xf32>, [[PARAM_2_:%.+]]: tensor<1x8x32xf32>, [[PARAM_3_:%.+]]: tensor<1x8x32xf32>) -> tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]], [[PARAM_2_]], [[PARAM_1_]], [[PARAM_3_]]) <{kind = "zhigh.mul-add-stick"}> ({
// CHECK:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<24x8x32xf32>, [[REGION_ARG_1_:%.+]]: tensor<1x8x32xf32>, [[REGION_ARG_2_:%.+]]: tensor<24x8x32xf32>, [[REGION_ARG_3_:%.+]]: tensor<1x8x32xf32>):
// CHECK-DAG:         [[VAR_1_:%.+]] = onnx.Constant dense<[24, 8, 32]> : tensor<3xi64>
// CHECK-DAG:         [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:         [[VAR_3_:%.+]] = "onnx.Mul"([[REGION_ARG_0_]], [[REGION_ARG_1_]]) : (tensor<24x8x32xf32>, tensor<1x8x32xf32>) -> tensor<24x8x32xf32>
// CHECK-DAG:         [[VAR_4_:%.+]] = "onnx.Mul"([[REGION_ARG_2_]], [[REGION_ARG_3_]]) : (tensor<24x8x32xf32>, tensor<1x8x32xf32>) -> tensor<24x8x32xf32>
// CHECK:             [[VAR_5_:%.+]] = "onnx.Add"([[VAR_3_]], [[VAR_4_]]) : (tensor<24x8x32xf32>, tensor<24x8x32xf32>) -> tensor<24x8x32xf32>
// CHECK:             [[VAR_6_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_5_]]) : (tensor<f32>, tensor<24x8x32xf32>) -> tensor<24x8x32xf32>
// CHECK:             [[VAR_7_:%.+]] = "onnx.Reshape"([[VAR_6_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<24x8x32xf32>, tensor<3xi64>) -> tensor<24x8x32xf32>
// CHECK:             [[VAR_8_:%.+]] = "zhigh.Stick"([[VAR_7_]]) <{layout = "3DS"}> : (tensor<24x8x32xf32>) -> tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:             onnx.Yield [[VAR_8_]] : tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }) {isSub = false, mulScalar = 2.000000e+00 : f32, onnx_node_name = "onnx.Mul-onnx.Mul-onnx.Add-onnx.Mul-onnx.Reshape-zhigh.Stick", reshapeCollapsedCount = 0 : i64, reshapeFirstCollapsedDim = -1 : i64, stickFormat = "3DS"} : (tensor<24x8x32xf32>, tensor<1x8x32xf32>, tensor<24x8x32xf32>, tensor<1x8x32xf32>) -> tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// Coexistence with simd-split-op-gather: the second Mul is fed by the
// rotate-half Slice/Neg/Concat of the first Mul's input, which therefore has
// two uses. Both fused ops are formed.
func.func @mul_add_stick_with_rotate_half(
    %x: tensor<?x12x?x32xf32> {onnx.dim_params = "0:batch,2:seq"},
    %cos: tensor<1x1x?x32xf32> {onnx.dim_params = "2:seq"},
    %sin: tensor<1x1x?x32xf32> {onnx.dim_params = "2:seq"})
    -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %m1 = onnx.Constant dense<-1> : tensor<1xi64>
  %c32 = onnx.Constant dense<32> : tensor<1xi64>
  %seq = "onnx.Dim"(%x) <{axis = 2 : si64}> : (tensor<?x12x?x32xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%m1, %seq, %c32) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %max = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c16 = onnx.Constant dense<16> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %k = onnx.Constant dense<0.420448214> : tensor<1xf32>
  %lo = "onnx.Slice"(%x, %c0, %c16, %c3, %c1) : (tensor<?x12x?x32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x12x?x16xf32>
  %hi = "onnx.Slice"(%x, %c16, %max, %c3, %c1) : (tensor<?x12x?x32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x12x?x16xf32>
  %neg = "onnx.Neg"(%hi) : (tensor<?x12x?x16xf32>) -> tensor<?x12x?x16xf32>
  %rot = "onnx.Concat"(%neg, %lo) <{axis = 3 : si64}> : (tensor<?x12x?x16xf32>, tensor<?x12x?x16xf32>) -> tensor<?x12x?x32xf32>
  %0 = "onnx.Mul"(%x, %cos) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
  %1 = "onnx.Mul"(%rot, %sin) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>) -> tensor<?x12x?x32xf32>
  %3 = "onnx.Mul"(%2, %k) : (tensor<?x12x?x32xf32>, tensor<1xf32>) -> tensor<?x12x?x32xf32>
  %4 = "onnx.Reshape"(%3, %shape) <{allowzero = 0 : si64}> : (tensor<?x12x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
  %5 = "zhigh.Stick"(%4) <{layout = "3DS"}> : (tensor<?x?x32xf32>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %5 : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @mul_add_stick_with_rotate_half
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x12x?x32xf32> {onnx.dim_params = "0:batch,2:seq"}, [[PARAM_1_:%.+]]: tensor<1x1x?x32xf32> {onnx.dim_params = "2:seq"}, [[PARAM_2_:%.+]]: tensor<1x1x?x32xf32> {onnx.dim_params = "2:seq"}) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Dim"([[PARAM_0_]]) <{axis = 2 : si64}> : (tensor<?x12x?x32xf32>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_0_]], [[VAR_2_]], [[VAR_1_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Fused"([[PARAM_0_]]) <{kind = "simd-split-op-gather"}> ({
// CHECK:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<?x12x?x32xf32>):
// CHECK-DAG:         [[VAR_6_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
// CHECK-DAG:         [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:         [[VAR_8_:%.+]] = onnx.Constant dense<16> : tensor<1xi64>
// CHECK-DAG:         [[VAR_9_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:         [[VAR_10_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = "onnx.Slice"([[REGION_ARG_0_]], [[VAR_7_]], [[VAR_8_]], [[VAR_9_]], [[VAR_10_]]) : (tensor<?x12x?x32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x12x?x16xf32>
// CHECK-DAG:         [[VAR_12_:%.+]] = "onnx.Slice"([[REGION_ARG_0_]], [[VAR_8_]], [[VAR_6_]], [[VAR_9_]], [[VAR_10_]]) : (tensor<?x12x?x32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x12x?x16xf32>
// CHECK:             [[VAR_13_:%.+]] = "onnx.Neg"([[VAR_12_]]) : (tensor<?x12x?x16xf32>) -> tensor<?x12x?x16xf32>
// CHECK:             [[VAR_14_:%.+]] = "onnx.Concat"([[VAR_13_]], [[VAR_11_]]) <{axis = 3 : si64}> : (tensor<?x12x?x16xf32>, tensor<?x12x?x16xf32>) -> tensor<?x12x?x32xf32>
// CHECK:             onnx.Yield [[VAR_14_]] : tensor<?x12x?x32xf32>
// CHECK:           }) {axis = 3 : i64, hasOpForSplitHigh = true, hasOpForSplitLow = false, onnx_node_name = "onnx.Slice-onnx.Slice-onnx.Neg-onnx.Concat", outputOffsetForSplitHigh = 0 : i64, outputOffsetForSplitLow = 16 : i64, splitPoint = 16 : i64} : (tensor<?x12x?x32xf32>) -> tensor<?x12x?x32xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Fused"([[PARAM_0_]], [[PARAM_1_]], [[VAR_4_]], [[PARAM_2_]], [[VAR_3_]]) <{kind = "zhigh.mul-add-stick"}> ({
// CHECK:           ^bb0([[REGION_ARG_1_:%.+]]: tensor<?x12x?x32xf32>, [[REGION_ARG_2_:%.+]]: tensor<1x1x?x32xf32>, [[REGION_ARG_3_:%.+]]: tensor<?x12x?x32xf32>, [[REGION_ARG_4_:%.+]]: tensor<1x1x?x32xf32>, [[REGION_ARG_5_:%.+]]: tensor<3xi64>):
// CHECK-DAG:         [[VAR_6_1_:%.+]] = onnx.Constant dense<0.420448214> : tensor<1xf32>
// CHECK-DAG:         [[VAR_7_1_:%.+]] = "onnx.Mul"([[REGION_ARG_1_]], [[REGION_ARG_2_]]) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
// CHECK-DAG:         [[VAR_8_1_:%.+]] = "onnx.Mul"([[REGION_ARG_3_]], [[REGION_ARG_4_]]) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
// CHECK:             [[VAR_9_1_:%.+]] = "onnx.Add"([[VAR_7_1_]], [[VAR_8_1_]]) : (tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>) -> tensor<?x12x?x32xf32>
// CHECK:             [[VAR_10_1_:%.+]] = "onnx.Mul"([[VAR_9_1_]], [[VAR_6_1_]]) : (tensor<?x12x?x32xf32>, tensor<1xf32>) -> tensor<?x12x?x32xf32>
// CHECK:             [[VAR_11_1_:%.+]] = "onnx.Reshape"([[VAR_10_1_]], [[REGION_ARG_5_]]) <{allowzero = 0 : si64}> : (tensor<?x12x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
// CHECK:             [[VAR_12_1_:%.+]] = "zhigh.Stick"([[VAR_11_1_]]) <{layout = "3DS"}> : (tensor<?x?x32xf32>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:             onnx.Yield [[VAR_12_1_]] : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }) {isSub = false, mulScalar = 0.420448214 : f32, onnx_node_name = "onnx.Mul-onnx.Mul-onnx.Add-onnx.Mul-onnx.Reshape-zhigh.Stick", reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS"} : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>, tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_5_]] : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// No fusion: the first Mul result has a second use.
func.func @no_fuse_mul_extra_use(%x1: tensor<2x4x8x32xf32>, %x2: tensor<2x4x8x32xf32>,
    %cos: tensor<1x1x8x32xf32>, %sin: tensor<1x1x8x32xf32>)
    -> (tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x32xf32>) {
  %shape = onnx.Constant dense<[8, 8, 32]> : tensor<3xi64>
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
  %3 = "onnx.Reshape"(%2, %shape) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<3xi64>) -> tensor<8x8x32xf32>
  %4 = "zhigh.Stick"(%3) <{layout = "3DS"}> : (tensor<8x8x32xf32>) -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %4, %0 : tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x32xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @no_fuse_mul_extra_use
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_2_:%.+]]: tensor<1x1x8x32xf32>, [[PARAM_3_:%.+]]: tensor<1x1x8x32xf32>) -> (tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x32xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[8, 8, 32]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_2_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[PARAM_3_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_2_]]) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<3xi64>) -> tensor<8x8x32xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) <{layout = "3DS"}> : (tensor<8x8x32xf32>) -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_5_]], [[VAR_1_]] : tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x32xf32>
// CHECK:         }
}

// -----

// No fusion: the Add result has a second use.
func.func @no_fuse_add_extra_use(%x1: tensor<2x4x8x32xf32>, %x2: tensor<2x4x8x32xf32>,
    %cos: tensor<1x1x8x32xf32>, %sin: tensor<1x1x8x32xf32>)
    -> (tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x32xf32>) {
  %shape = onnx.Constant dense<[8, 8, 32]> : tensor<3xi64>
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
  %3 = "onnx.Reshape"(%2, %shape) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<3xi64>) -> tensor<8x8x32xf32>
  %4 = "zhigh.Stick"(%3) <{layout = "3DS"}> : (tensor<8x8x32xf32>) -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %4, %2 : tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x32xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @no_fuse_add_extra_use
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_2_:%.+]]: tensor<1x1x8x32xf32>, [[PARAM_3_:%.+]]: tensor<1x1x8x32xf32>) -> (tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x32xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[8, 8, 32]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_2_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[PARAM_3_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_2_]]) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<3xi64>) -> tensor<8x8x32xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) <{layout = "3DS"}> : (tensor<8x8x32xf32>) -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_5_]], [[VAR_3_]] : tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x32xf32>
// CHECK:         }
}

// -----

// No fusion: the Reshape merges the last dim.
func.func @no_fuse_reshape_merges_last_dim(%x1: tensor<2x4x8x32xf32>, %x2: tensor<2x4x8x32xf32>,
    %cos: tensor<1x1x8x32xf32>, %sin: tensor<1x1x8x32xf32>)
    -> tensor<2x4x256xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %shape = onnx.Constant dense<[2, 4, 256]> : tensor<3xi64>
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
  %3 = "onnx.Reshape"(%2, %shape) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<3xi64>) -> tensor<2x4x256xf32>
  %4 = "zhigh.Stick"(%3) <{layout = "3DS"}> : (tensor<2x4x256xf32>) -> tensor<2x4x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %4 : tensor<2x4x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @no_fuse_reshape_merges_last_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_2_:%.+]]: tensor<1x1x8x32xf32>, [[PARAM_3_:%.+]]: tensor<1x1x8x32xf32>) -> tensor<2x4x256xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[2, 4, 256]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_2_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[PARAM_3_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_2_]]) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<3xi64>) -> tensor<2x4x256xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) <{layout = "3DS"}> : (tensor<2x4x256xf32>) -> tensor<2x4x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_5_]] : tensor<2x4x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// No fusion: 4D stick (rank 4 Reshape output) is not supported (yet).
func.func @no_fuse_stick_4d(%x1: tensor<2x4x8x32xf32>, %x2: tensor<2x4x8x32xf32>,
    %cos: tensor<1x1x8x32xf32>, %sin: tensor<1x1x8x32xf32>)
    -> tensor<1x8x8x32xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %shape = onnx.Constant dense<[1, 8, 8, 32]> : tensor<4xi64>
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
  %3 = "onnx.Reshape"(%2, %shape) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<4xi64>) -> tensor<1x8x8x32xf32>
  %4 = "zhigh.Stick"(%3) <{layout = "4D"}> : (tensor<1x8x8x32xf32>) -> tensor<1x8x8x32xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %4 : tensor<1x8x8x32xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @no_fuse_stick_4d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_2_:%.+]]: tensor<1x1x8x32xf32>, [[PARAM_3_:%.+]]: tensor<1x1x8x32xf32>) -> tensor<1x8x8x32xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 8, 8, 32]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_2_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[PARAM_3_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_2_]]) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<4xi64>) -> tensor<1x8x8x32xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) <{layout = "4D"}> : (tensor<1x8x8x32xf32>) -> tensor<1x8x8x32xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_5_]] : tensor<1x8x8x32xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

// No fusion: innermost dim 48 is neither a half stick nor whole sticks.
func.func @no_fuse_d48(%x1: tensor<2x4x8x48xf32>, %x2: tensor<2x4x8x48xf32>,
    %cos: tensor<1x1x8x48xf32>, %sin: tensor<1x1x8x48xf32>)
    -> tensor<8x8x48xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %shape = onnx.Constant dense<[8, 8, 48]> : tensor<3xi64>
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<2x4x8x48xf32>, tensor<1x1x8x48xf32>) -> tensor<2x4x8x48xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<2x4x8x48xf32>, tensor<1x1x8x48xf32>) -> tensor<2x4x8x48xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<2x4x8x48xf32>, tensor<2x4x8x48xf32>) -> tensor<2x4x8x48xf32>
  %3 = "onnx.Reshape"(%2, %shape) <{allowzero = 0 : si64}> : (tensor<2x4x8x48xf32>, tensor<3xi64>) -> tensor<8x8x48xf32>
  %4 = "zhigh.Stick"(%3) <{layout = "3DS"}> : (tensor<8x8x48xf32>) -> tensor<8x8x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %4 : tensor<8x8x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @no_fuse_d48
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8x48xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8x48xf32>, [[PARAM_2_:%.+]]: tensor<1x1x8x48xf32>, [[PARAM_3_:%.+]]: tensor<1x1x8x48xf32>) -> tensor<8x8x48xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[8, 8, 48]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_2_]]) : (tensor<2x4x8x48xf32>, tensor<1x1x8x48xf32>) -> tensor<2x4x8x48xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[PARAM_3_]]) : (tensor<2x4x8x48xf32>, tensor<1x1x8x48xf32>) -> tensor<2x4x8x48xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_2_]]) : (tensor<2x4x8x48xf32>, tensor<2x4x8x48xf32>) -> tensor<2x4x8x48xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x4x8x48xf32>, tensor<3xi64>) -> tensor<8x8x48xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) <{layout = "3DS"}> : (tensor<8x8x48xf32>) -> tensor<8x8x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_5_]] : tensor<8x8x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// No fusion: the table's dynamic dim 0 is not known to match the static 12
// heads, so it is not a static broadcast.
func.func @no_fuse_dynamic_broadcast(%x1: tensor<2x12x8x32xf32>, %x2: tensor<2x12x8x32xf32>,
    %cos: tensor<?x8x32xf32>, %sin: tensor<1x1x8x32xf32>)
    -> tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %shape = onnx.Constant dense<[24, 8, 32]> : tensor<3xi64>
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<2x12x8x32xf32>, tensor<?x8x32xf32>) -> tensor<2x12x8x32xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<2x12x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x12x8x32xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<2x12x8x32xf32>, tensor<2x12x8x32xf32>) -> tensor<2x12x8x32xf32>
  %3 = "onnx.Reshape"(%2, %shape) <{allowzero = 0 : si64}> : (tensor<2x12x8x32xf32>, tensor<3xi64>) -> tensor<24x8x32xf32>
  %4 = "zhigh.Stick"(%3) <{layout = "3DS"}> : (tensor<24x8x32xf32>) -> tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %4 : tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @no_fuse_dynamic_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x12x8x32xf32>, [[PARAM_1_:%.+]]: tensor<2x12x8x32xf32>, [[PARAM_2_:%.+]]: tensor<?x8x32xf32>, [[PARAM_3_:%.+]]: tensor<1x1x8x32xf32>) -> tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[24, 8, 32]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_2_]]) : (tensor<2x12x8x32xf32>, tensor<?x8x32xf32>) -> tensor<2x12x8x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[PARAM_3_]]) : (tensor<2x12x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x12x8x32xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_2_]]) : (tensor<2x12x8x32xf32>, tensor<2x12x8x32xf32>) -> tensor<2x12x8x32xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x12x8x32xf32>, tensor<3xi64>) -> tensor<24x8x32xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) <{layout = "3DS"}> : (tensor<24x8x32xf32>) -> tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_5_]] : tensor<24x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// No fusion: a constant table would be cloned into the body (v1 limit).
func.func @no_fuse_constant_table(%x1: tensor<2x4x8x32xf32>, %x2: tensor<2x4x8x32xf32>,
    %sin: tensor<1x1x8x32xf32>)
    -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %shape = onnx.Constant dense<[8, 8, 32]> : tensor<3xi64>
  %cos = onnx.Constant dense<0.5> : tensor<1x1x8x32xf32>
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
  %3 = "onnx.Reshape"(%2, %shape) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<3xi64>) -> tensor<8x8x32xf32>
  %4 = "zhigh.Stick"(%3) <{layout = "3DS"}> : (tensor<8x8x32xf32>) -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %4 : tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @no_fuse_constant_table
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_2_:%.+]]: tensor<1x1x8x32xf32>) -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[8, 8, 32]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<1x1x8x32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_1_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[PARAM_2_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_2_]], [[VAR_3_]]) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Reshape"([[VAR_4_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<3xi64>) -> tensor<8x8x32xf32>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Stick"([[VAR_5_]]) <{layout = "3DS"}> : (tensor<8x8x32xf32>) -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_6_]] : tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// No fusion: the second Sub operand is not a Mul.
func.func @no_fuse_sub_operand_not_mul(%x1: tensor<2x4x8x32xf32>, %x2: tensor<2x4x8x32xf32>,
    %cos: tensor<1x1x8x32xf32>)
    -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %shape = onnx.Constant dense<[8, 8, 32]> : tensor<3xi64>
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
  %1 = "onnx.Sub"(%0, %x2) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
  %2 = "onnx.Reshape"(%1, %shape) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<3xi64>) -> tensor<8x8x32xf32>
  %3 = "zhigh.Stick"(%2) <{layout = "3DS"}> : (tensor<8x8x32xf32>) -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %3 : tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @no_fuse_sub_operand_not_mul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8x32xf32>, [[PARAM_2_:%.+]]: tensor<1x1x8x32xf32>) -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[8, 8, 32]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_2_]]) : (tensor<2x4x8x32xf32>, tensor<1x1x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[VAR_1_]], [[PARAM_1_]]) : (tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x4x8x32xf32>, tensor<3xi64>) -> tensor<8x8x32xf32>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Stick"([[VAR_3_]]) <{layout = "3DS"}> : (tensor<8x8x32xf32>) -> tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_4_]] : tensor<8x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// No fusion: without a proof that the table's dynamic sequence dim equals the
// data's, it could be a runtime broadcast (size 1), which is not supported.
func.func @no_fuse_unproven_dynamic_dim(%x1: tensor<?x12x?x32xf32>, %x2: tensor<?x12x?x32xf32>,
    %cos: tensor<1x1x?x32xf32>, %sin: tensor<1x1x?x32xf32>, %shape: tensor<3xi64>)
    -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %0 = "onnx.Mul"(%x1, %cos) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
  %1 = "onnx.Mul"(%x2, %sin) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>) -> tensor<?x12x?x32xf32>
  %3 = "onnx.Reshape"(%2, %shape) <{allowzero = 0 : si64}> : (tensor<?x12x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
  %4 = "zhigh.Stick"(%3) <{layout = "3DS"}> : (tensor<?x?x32xf32>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %4 : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @no_fuse_unproven_dynamic_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x12x?x32xf32>, [[PARAM_1_:%.+]]: tensor<?x12x?x32xf32>, [[PARAM_2_:%.+]]: tensor<1x1x?x32xf32>, [[PARAM_3_:%.+]]: tensor<1x1x?x32xf32>, [[PARAM_4_:%.+]]: tensor<3xi64>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_2_]]) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[PARAM_3_]]) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) : (tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>) -> tensor<?x12x?x32xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[PARAM_4_]]) <{allowzero = 0 : si64}> : (tensor<?x12x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Stick"([[VAR_3_]]) <{layout = "3DS"}> : (tensor<?x?x32xf32>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_4_]] : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}
