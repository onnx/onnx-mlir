// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// Tests for the extended layout transform patterns using the default (fused-op)
// configuration.  The pass wraps the layout-transform chain in an onnx.Fused
// region instead of emitting the hardcoded zhigh.ExtendedLayoutTransform op.
// Compare with zhigh-fold-op-stick-unstick.mlir which uses --disable-fused-op.

// -----

// layout reshape transpose reshape layout — full chain ending with 3DS stick.

func.func @pattern_extended_layout_transform_v1(%arg0: tensor<3x?x2048xf32>, %arg1: tensor<2048x2048xf32>, %arg2: tensor<2048x512xf32>, %arg3: tensor<2048x512xf32>, %arg4: tensor<64x64xf32>) -> tensor<3x32x?x64xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = onnx.Constant dense<32> : tensor<1xi64>
  %2 = onnx.Constant dense<96> : tensor<1xi64>
  %3 = onnx.Constant dense<3> : tensor<1xi64>
  %4 = onnx.Constant dense<64> : tensor<1xi64>
  %5 = onnx.Constant dense<-1> : tensor<1xi64>
  %6 = "onnx.Dim"(%arg0) {axis = 1 : si64, onnx_node_name = "onnx.Dim_0"} : (tensor<3x?x2048xf32>) -> tensor<1xi64>
  %7 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<3x?x2048xf32>) -> tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %8 = "zhigh.Stick"(%arg1) {layout = "2D"} : (tensor<2048x2048xf32>) -> tensor<2048x2048xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %9 = "zhigh.MatMul"(%7, %8, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2048x2048xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %10 = "onnx.LayoutTransform"(%9) : (tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x?x2048xf16>
  %11 = "onnx.Concat"(%3, %6, %5, %4) {axis = 0 : si64, onnx_node_name = "/model/layers.0/self_attn/Concat"} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %12 = "onnx.Reshape"(%10, %11) {allowzero = 0 : si64} : (tensor<3x?x2048xf16>, tensor<4xi64>) -> tensor<3x?x32x64xf16>
  %13 = "onnx.Transpose"(%12) {perm = [0, 2, 1, 3]} : (tensor<3x?x32x64xf16>) -> tensor<3x32x?x64xf16>
  %14 = "onnx.Concat"(%2, %6, %4) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %15 = "onnx.Reshape"(%13, %14) {allowzero = 0 : si64} : (tensor<3x32x?x64xf16>, tensor<3xi64>) -> tensor<96x?x64xf16>
  %16 = "onnx.LayoutTransform"(%15) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>} : (tensor<96x?x64xf16>) -> tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %17 = "zhigh.Stick"(%arg4) {layout = "2D"} : (tensor<64x64xf32>) -> tensor<64x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %18 = "zhigh.MatMul"(%16, %17, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<64x64xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %19 = "zhigh.Unstick"(%18) : (tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x?x64xf32>
  %20 = "onnx.Concat"(%3, %1, %6, %4) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %21 = "onnx.Reshape"(%19, %20) {allowzero = 0 : si64} : (tensor<96x?x64xf32>, tensor<4xi64>) -> tensor<3x32x?x64xf32>
  return %21 : tensor<3x32x?x64xf32>

// CHECK-LABEL:  func.func @pattern_extended_layout_transform_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x?x2048xf32>, [[PARAM_1_:%.+]]: tensor<2048x2048xf32>, [[PARAM_2_:%.+]]: tensor<2048x512xf32>, [[PARAM_3_:%.+]]: tensor<2048x512xf32>, [[PARAM_4_:%.+]]: tensor<64x64xf32>) -> tensor<3x32x?x64xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<64> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Dim"([[PARAM_0_]]) <{axis = 1 : si64}> {onnx_node_name = "onnx.Dim_0"} : (tensor<3x?x2048xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) <{layout = "3DS"}> : (tensor<3x?x2048xf32>) -> tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) <{layout = "2D"}> : (tensor<2048x2048xf32>) -> tensor<2048x2048xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.MatMul"([[VAR_5_]], [[VAR_6_]], [[VAR_0_]]) <{transposeA = 0 : si64, transposeB = 0 : si64}> : (tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2048x2048xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Fused"([[VAR_7_]], {{.*}}) <{kind = "zhigh.extended_layout_transform"}>
// Verify the chain ops are inside the fused body:
// CHECK:           "onnx.LayoutTransform"
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<3x?x32x64xf16>
// CHECK:           "onnx.Transpose"
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<96x?x64xf16>
// CHECK:           "onnx.LayoutTransform"{{.*}}-> tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           onnx.Yield
// Verify outer structure after the fused op:
// CHECK-DAG:       [[VAR_9_:%.+]] = "zhigh.Stick"([[PARAM_4_]]) <{layout = "2D"}> : (tensor<64x64xf32>) -> tensor<64x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_10_:%.+]] = "zhigh.MatMul"([[VAR_8_]], [[VAR_9_]], [[VAR_0_]]) <{transposeA = 0 : si64, transposeB = 0 : si64}> : (tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<64x64xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_11_:%.+]] = "zhigh.Unstick"([[VAR_10_]]) : (tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x?x64xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_1_]], [[VAR_4_]], [[VAR_3_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Reshape"([[VAR_11_]], [[VAR_12_]]) <{allowzero = 0 : si64}> : (tensor<96x?x64xf32>, tensor<4xi64>) -> tensor<3x32x?x64xf32>
// CHECK:           return [[VAR_13_]] : tensor<3x32x?x64xf32>
// CHECK-NOT:       "zhigh.ExtendedLayoutTransform"
// CHECK:           }
}

// -----

// layout reshape transpose DLF16ToF32 — chain ending with f32 conversion,
// no merge reshape and no final 3DS stick.

func.func @pattern_extended_layout_transform_v3(%arg0: tensor<3x?x2048xf32>, %arg1: tensor<2048x2048xf32>, %arg2: tensor<2048x512xf32>, %arg3: tensor<2048x512xf32>, %arg4: tensor<64x64xf32>) -> tensor<3x8x?x64xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = onnx.Constant dense<3> : tensor<1xi64>
    %2 = onnx.Constant dense<64> : tensor<1xi64>
    %3 = onnx.Constant dense<-1> : tensor<1xi64>
    %4 = "onnx.Dim"(%arg0) {axis = 1 : si64, onnx_node_name = "onnx.Dim_0"} : (tensor<3x?x2048xf32>) -> tensor<1xi64>
    %5 = "onnx.Concat"(%1, %4, %3, %2) {axis = 0 : si64, onnx_node_name = "/model/layers.0/self_attn/Concat_1"} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %6 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<3x?x2048xf32>) -> tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %7 = "zhigh.Stick"(%arg2) {layout = "2D"} : (tensor<2048x512xf32>) -> tensor<2048x512xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %8 = "zhigh.MatMul"(%6, %7, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2048x512xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %9 = "onnx.LayoutTransform"(%8) : (tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x?x512xf16>
    %10 = "onnx.Reshape"(%9, %5) {allowzero = 0 : si64} : (tensor<3x?x512xf16>, tensor<4xi64>) -> tensor<3x?x8x64xf16>
    %11 = "onnx.Transpose"(%10) {perm = [0, 2, 1, 3]} : (tensor<3x?x8x64xf16>) -> tensor<3x8x?x64xf16>
    %12 = "zhigh.DLF16ToF32"(%11) : (tensor<3x8x?x64xf16>) -> tensor<3x8x?x64xf32>
    return %12 : tensor<3x8x?x64xf32>

// CHECK-LABEL:  func.func @pattern_extended_layout_transform_v3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x?x2048xf32>, [[PARAM_1_:%.+]]: tensor<2048x2048xf32>, [[PARAM_2_:%.+]]: tensor<2048x512xf32>, [[PARAM_3_:%.+]]: tensor<2048x512xf32>, [[PARAM_4_:%.+]]: tensor<64x64xf32>) -> tensor<3x8x?x64xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) <{layout = "3DS"}> : (tensor<3x?x2048xf32>) -> tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) <{layout = "2D"}> : (tensor<2048x512xf32>) -> tensor<2048x512xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) <{transposeA = 0 : si64, transposeB = 0 : si64}> : (tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2048x512xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Fused"([[VAR_3_]], {{.*}}) <{kind = "zhigh.extended_layout_transform"}>
// Verify the chain ops are inside the fused body:
// CHECK:           "onnx.LayoutTransform"
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<3x?x8x64xf16>
// CHECK:           "onnx.Transpose"
// CHECK:           "zhigh.DLF16ToF32"{{.*}}-> tensor<3x8x?x64xf32>
// CHECK:           onnx.Yield
// Verify return uses fused op output directly:
// CHECK:           return [[VAR_4_]] : tensor<3x8x?x64xf32>
// CHECK-NOT:       "zhigh.ExtendedLayoutTransform"
// CHECK:           }
}
