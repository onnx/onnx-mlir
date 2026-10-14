// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --test-compiler-opt --fusion-op-stick-unstick="disable-fused-op" %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --test-compiler-opt --fusion-op-stick-unstick %s -split-input-file | FileCheck %s --check-prefix=FUSED

// REQUIRES: test-compiler-opt

// Extended layout transform with half sticks (innermost dim of 32), which are
// only enabled with --test-compiler-opt. CHECK lines cover the hardcoded
// zhigh.ExtendedLayoutTransform op, FUSED lines the onnx.Fused region.

// -----

// Inner dim of 32 (half stick): heads of 32 are merged into the innermost dim
// (12 x 32 -> 384), so each output stick is written as two half sticks.

func.func @test_lt_32_split_transpose_merge_lt(%arg0: tensor<24x7x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x7x384xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %s1 = onnx.Constant dense<[2, 12, 7, 32]> : tensor<4xi64>
  %s2 = onnx.Constant dense<[2, 7, 384]> : tensor<3xi64>
  %0 = "onnx.LayoutTransform"(%arg0) : (tensor<24x7x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<24x7x32xf16>
  %1 = "onnx.Reshape"(%0, %s1) {allowzero = 0 : si64} : (tensor<24x7x32xf16>, tensor<4xi64>) -> tensor<2x12x7x32xf16>
  %2 = "onnx.Transpose"(%1) {perm = [0, 2, 1, 3]} : (tensor<2x12x7x32xf16>) -> tensor<2x7x12x32xf16>
  %3 = "onnx.Reshape"(%2, %s2) {allowzero = 0 : si64} : (tensor<2x7x12x32xf16>, tensor<3xi64>) -> tensor<2x7x384xf16>
  %4 = "onnx.LayoutTransform"(%3) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>} : (tensor<2x7x384xf16>) -> tensor<2x7x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %4 : tensor<2x7x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_lt_32_split_transpose_merge_lt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<24x7x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x7x384xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.ExtendedLayoutTransform"([[PARAM_0_]]) <{dlf16_to_f32 = false, reshape_merge_axis = 2 : si64, reshape_split_axis = 0 : si64, reshape_split_factor = 12 : si64, target_layout = "3DS", transpose_pattern = [0, 2, 1, 3]}> : (tensor<24x7x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x7x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<2x7x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
// FUSED-LABEL:  func.func @test_lt_32_split_transpose_merge_lt
// FUSED:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_:%.+]]) <{kind = "zhigh.extended_layout_transform"}>
// FUSED:           "onnx.LayoutTransform"
// FUSED:           "onnx.Reshape"{{.*}}-> tensor<2x12x7x32xf16>
// FUSED:           "onnx.Transpose"
// FUSED:           "onnx.Reshape"{{.*}}-> tensor<2x7x384xf16>
// FUSED:           "onnx.LayoutTransform"{{.*}}-> tensor<2x7x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// FUSED:           onnx.Yield
// FUSED:           reshapeMergeAxis = 2 : i64, reshapeSplitAxis = 0 : i64, reshapeSplitFactor = 12 : i64, transposePattern = [0, 2, 1, 3]
// FUSED:           return [[VAR_0_]] : tensor<2x7x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// FUSED:         }
}

// -----

// Inner dim of 48 is neither a multiple of 64 nor exactly 32: not fused, even
// with --test-compiler-opt.

func.func @test_lt_48_split_transpose_merge_lt_not_fused(%arg0: tensor<24x7x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x7x576xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %s1 = onnx.Constant dense<[2, 12, 7, 48]> : tensor<4xi64>
  %s2 = onnx.Constant dense<[2, 7, 576]> : tensor<3xi64>
  %0 = "onnx.LayoutTransform"(%arg0) : (tensor<24x7x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<24x7x48xf16>
  %1 = "onnx.Reshape"(%0, %s1) {allowzero = 0 : si64} : (tensor<24x7x48xf16>, tensor<4xi64>) -> tensor<2x12x7x48xf16>
  %2 = "onnx.Transpose"(%1) {perm = [0, 2, 1, 3]} : (tensor<2x12x7x48xf16>) -> tensor<2x7x12x48xf16>
  %3 = "onnx.Reshape"(%2, %s2) {allowzero = 0 : si64} : (tensor<2x7x12x48xf16>, tensor<3xi64>) -> tensor<2x7x576xf16>
  %4 = "onnx.LayoutTransform"(%3) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>} : (tensor<2x7x576xf16>) -> tensor<2x7x576xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %4 : tensor<2x7x576xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_lt_48_split_transpose_merge_lt_not_fused
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<24x7x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x7x576xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[2, 12, 7, 48]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 7, 576]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.LayoutTransform"([[PARAM_0_]]) : (tensor<24x7x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<24x7x48xf16>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<24x7x48xf16>, tensor<4xi64>) -> tensor<2x12x7x48xf16>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) <{perm = [0, 2, 1, 3]}> : (tensor<2x12x7x48xf16>) -> tensor<2x7x12x48xf16>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Reshape"([[VAR_4_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<2x7x12x48xf16>, tensor<3xi64>) -> tensor<2x7x576xf16>
// CHECK:           [[VAR_6_:%.+]] = "onnx.LayoutTransform"([[VAR_5_]]) <{target_layout = #zhigh.layout<{dataLayout = "3DS"}>}> : (tensor<2x7x576xf16>) -> tensor<2x7x576xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_6_]] : tensor<2x7x576xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
// FUSED-LABEL:  func.func @test_lt_48_split_transpose_merge_lt_not_fused
// FUSED-NOT:       "onnx.Fused"
// FUSED:           return
}
