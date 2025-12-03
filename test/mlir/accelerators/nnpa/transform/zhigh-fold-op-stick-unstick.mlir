// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// -----

// inputs stick, output stick (second unstick -> add disabled because of broadcast)

func.func @test_add_sss(%arg0: tensor<4x256x256xf32>, %arg1: tensor<4x256x1xf32>) -> tensor<4x256x256xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %0) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
  %3 = "zhigh.Stick"(%arg1) {layout = "3DS"} : (tensor<4x256x1xf32>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %4 = "zhigh.Add"(%3, %3) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x1xf32>
  %6 = "onnx.Add"(%2, %5) {onnx_node_name = "onnx.Add_2"} : (tensor<4x256x256xf32>, tensor<4x256x1xf32>) -> tensor<4x256x256xf32>
  %7 = "zhigh.Stick"(%6) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %8 = "zhigh.Add"(%7, %0) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %9 = "zhigh.Unstick"(%8) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
  return %9 : tensor<4x256x256xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_add_sss
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x256x256xf32>, [[PARAM_1_:%.+]]: tensor<4x256x1xf32>) -> tensor<4x256x256xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Add"([[VAR_0_]], [[VAR_0_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<4x256x1xf32>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Add"([[VAR_2_]], [[VAR_2_]]) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x1xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_4_]]) {onnx_node_name = "onnx.Add_2"} : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x1xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Add"([[VAR_5_]], [[VAR_0_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Unstick"([[VAR_6_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
// CHECK:           return [[VAR_7_]] : tensor<4x256x256xf32>
// CHECK:         }
}

// -----

// Unary input stick, output stick

func.func @test_hardsigmoid_ss(%arg0: tensor<4x256x256xf32>, %arg1: tensor<4x256x1xf32>) -> tensor<4x256x256xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %0) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
  %3 = "onnx.HardSigmoid"(%2) {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32, onnx_node_name = "onnx.HardSigmoid_1"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf32>
  %4 = "zhigh.Stick"(%3) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %5 = "zhigh.Add"(%4, %0) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %6 = "zhigh.Unstick"(%5) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
  return %6 : tensor<4x256x256xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_hardsigmoid_ss
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x256x256xf32>, [[PARAM_1_:%.+]]: tensor<4x256x1xf32>) -> tensor<4x256x256xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Add"([[VAR_0_]], [[VAR_0_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "onnx.HardSigmoid"([[VAR_1_]]) {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32, onnx_node_name = "onnx.HardSigmoid_1"} : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Add"([[VAR_2_]], [[VAR_0_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
// CHECK:           return [[VAR_4_]] : tensor<4x256x256xf32>
// CHECK:         }
}

// -----

// input stick/normal, output normal

func.func @test_add_nsn(%arg0: tensor<4x256x256xf32>, %arg1: tensor<4x256x1xf32>) -> tensor<4x256x256xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %0) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
  %3 = "onnx.Add"(%2, %arg1) {onnx_node_name = "onnx.Add_1"} : (tensor<4x256x256xf32>, tensor<4x256x1xf32>) -> tensor<4x256x256xf32>
  return %3 : tensor<4x256x256xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_add_nsn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x256x256xf32>, [[PARAM_1_:%.+]]: tensor<4x256x1xf32>) -> tensor<4x256x256xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Add"([[VAR_0_]], [[VAR_0_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Add"([[VAR_1_]], [[PARAM_1_]]) {onnx_node_name = "onnx.Add_1"} : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x1xf32>) -> tensor<4x256x256xf32>
// CHECK:           return [[VAR_2_]] : tensor<4x256x256xf32>
// CHECK:         }
}

// -----

// input normal/stick, output normal (unstick -> add disable because of broadcast)

func.func @test_add_smn(%arg0: tensor<4x256x256xf32>, %arg1: tensor<4x256x1xf32>) -> tensor<4x256x256xf32> {
  %0 = "zhigh.Stick"(%arg1) {layout = "3DS"} : (tensor<4x256x1xf32>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %0) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x1xf32>
  %3 = "onnx.Add"(%arg0, %2) {onnx_node_name = "onnx.Add_1"} : (tensor<4x256x256xf32>, tensor<4x256x1xf32>) -> tensor<4x256x256xf32>
  return %3 : tensor<4x256x256xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_add_smn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x256x256xf32>, [[PARAM_1_:%.+]]: tensor<4x256x1xf32>) -> tensor<4x256x256xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<4x256x1xf32>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Add"([[VAR_0_]], [[VAR_0_]]) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x1xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_2_]]) {onnx_node_name = "onnx.Add_1"} : (tensor<4x256x256xf32>, tensor<4x256x1xf32>) -> tensor<4x256x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<4x256x256xf32>
// CHECK:         }
}

// -----

// input normal, output stick

func.func @test_add_normal_stick(%arg0: tensor<4x256x256xf32>, %arg1: tensor<4x256x1xf32>) -> tensor<4x256x256xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "onnx.Add_0"} : (tensor<4x256x256xf32>, tensor<4x256x1xf32>) -> tensor<4x256x256xf32>
  %1 = "zhigh.Stick"(%0) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %3 = "zhigh.Add"(%1, %2) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %4 = "zhigh.Unstick"(%3) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
  return %4 : tensor<4x256x256xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_add_normal_stick
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x256x256xf32>, [[PARAM_1_:%.+]]: tensor<4x256x1xf32>) -> tensor<4x256x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) {onnx_node_name = "onnx.Add_0"} : (tensor<4x256x256xf32>, tensor<4x256x1xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<4x256x256xf32>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Add"([[VAR_0_]], [[VAR_1_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<4x256x256xf32>
// CHECK:         }
}

// -----


// unstick -> add disable because of broadcast.
func.func @test_add_multiple_layout_v1(%arg0: tensor<8x256x64xf32>, %arg1: tensor<256x64xf32>) -> tensor<8x256x64xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<8x256x64xf32>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %0) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf32>
  %3 = "zhigh.Stick"(%arg1) {layout = "2D"} : (tensor<256x64xf32>) -> tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %4 = "zhigh.Add"(%3, %3) : (tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x64xf32>
  %6 = "onnx.Add"(%2, %5) {onnx_node_name = "onnx.Add_2"} : (tensor<8x256x64xf32>, tensor<256x64xf32>) -> tensor<8x256x64xf32>
  %7 = "zhigh.Stick"(%6) {layout = "3DS"} : (tensor<8x256x64xf32>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %8 = "zhigh.Div"(%7, %1) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %9 = "zhigh.Unstick"(%8) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf32>
  return %9 : tensor<8x256x64xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_add_multiple_layout_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8x256x64xf32>, [[PARAM_1_:%.+]]: tensor<256x64xf32>) -> tensor<8x256x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<8x256x64xf32>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Add"([[VAR_0_]], [[VAR_0_]]) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<256x64xf32>) -> tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Add"([[VAR_2_]], [[VAR_2_]]) : (tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x64xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_4_]]) {onnx_node_name = "onnx.Add_2"} : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<256x64xf32>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Div"([[VAR_5_]], [[VAR_1_]]) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Unstick"([[VAR_6_]]) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf32>
// CHECK:           return [[VAR_7_]] : tensor<8x256x64xf32>
// CHECK:         }
}

// -----

// same but with broadcast in the inner dim as well.

func.func @test_add_multiple_layout_v2(%arg0: tensor<8x256x64xf32>, %arg1: tensor<256x1xf32>) -> tensor<8x256x64xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<8x256x64xf32>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %0) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf32>
  %3 = "zhigh.Stick"(%arg1) {layout = "2D"} : (tensor<256x1xf32>) -> tensor<256x1xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %4 = "zhigh.Add"(%3, %3) : (tensor<256x1xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x1xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x1xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %5 = "onnx.Add"(%2, %4) {onnx_node_name = "onnx.Add_2"} : (tensor<8x256x64xf32>, tensor<256x1xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %6 = "zhigh.Div"(%5, %1) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %7 = "zhigh.Unstick"(%6) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf32>
  return %7 : tensor<8x256x64xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_add_multiple_layout_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8x256x64xf32>, [[PARAM_1_:%.+]]: tensor<256x1xf32>) -> tensor<8x256x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<8x256x64xf32>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Add"([[VAR_0_]], [[VAR_0_]]) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<256x1xf32>) -> tensor<256x1xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Add"([[VAR_2_]], [[VAR_2_]]) : (tensor<256x1xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x1xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x1xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_3_]]) {onnx_node_name = "onnx.Add_2"} : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<256x1xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Div"([[VAR_4_]], [[VAR_1_]]) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf32>
// CHECK:           return [[VAR_6_]] : tensor<8x256x64xf32>
// CHECK:         }
}

// -----

// Layer norm with fusion before / after

func.func @layer_norm_fusion1(%arg0: tensor<256x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x256xf32> attributes {input_names = ["X", "S"], output_names = ["LN"]} {
  %0 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
  %1 = "zhigh.Stick"(%arg0) {layout = "2D"} : (tensor<256x256xf32>) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %2 = "zhigh.MatMul"(%1, %1, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %3 = "zhigh.Unstick"(%2) : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x256xf32>
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%3, %arg1, %0) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, onnx_node_name = "onnx.LayerNormalization_2", stash_type = 1 : si64} : (tensor<256x256xf32>, tensor<256xf32>, none) -> (tensor<256x256xf32>, none, none)
  %4 = "zhigh.Stick"(%Y) {layout = "2D"} : (tensor<256x256xf32>) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %5 = "zhigh.MatMul"(%4, %4, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %6 = "zhigh.Unstick"(%5) : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x256xf32>
  return %6 : tensor<256x256xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layer_norm_fusion1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<256x256xf32>, [[PARAM_1_:%.+]]: tensor<256xf32>) -> tensor<256x256xf32> attributes {input_names = ["X", "S"], output_names = ["LN"]} {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<256x256xf32>) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_1_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_2_]], [[PARAM_1_]], [[VAR_0_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, onnx_node_name = "onnx.LayerNormalization_2", stash_type = 1 : si64} : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256xf32>, none) -> (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none, none)
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_Y_]], [[VAR_Y_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x256xf32>
// CHECK:           return [[VAR_4_]] : tensor<256x256xf32>
// CHECK:         }
}

// -----

// Layer norm with fusion after

func.func @layer_norm_fusion2(%arg0: tensor<256x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x256xf32> attributes {input_names = ["X", "S"], output_names = ["LN"]} {
  %0 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %0) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, onnx_node_name = "onnx.LayerNormalization_1", stash_type = 1 : si64} : (tensor<256x256xf32>, tensor<256xf32>, none) -> (tensor<256x256xf32>, none, none)
  %1 = "zhigh.Stick"(%Y) {layout = "2D"} : (tensor<256x256xf32>) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %2 = "zhigh.MatMul"(%1, %1, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %3 = "zhigh.Unstick"(%2) : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x256xf32>
  return %3 : tensor<256x256xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layer_norm_fusion2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<256x256xf32>, [[PARAM_1_:%.+]]: tensor<256xf32>) -> tensor<256x256xf32> attributes {input_names = ["X", "S"], output_names = ["LN"]} {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, onnx_node_name = "onnx.LayerNormalization_1", stash_type = 1 : si64} : (tensor<256x256xf32>, tensor<256xf32>, none) -> (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none, none)
// CHECK:           [[VAR_1_:%.+]] = "zhigh.MatMul"([[VAR_Y_]], [[VAR_Y_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x256xf32>
// CHECK:           return [[VAR_2_]] : tensor<256x256xf32>
// CHECK:         }
}

// -----

// Clip which has between 1 and 3 inputs (2 optional)

func.func @clip_3_inputs(%arg0: tensor<4x32x64xf32>, %arg1: tensor<4x32x64xf32>, %arg2: tensor<1xf32>, %arg3: tensor<f32>) -> tensor<4x32x64xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<4x32x64xf32>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %0) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf32>
  %3 = "onnx.Clip"(%2, %arg2, %arg3) {onnx_node_name = "onnx.Clip_1"} : (tensor<4x32x64xf32>, tensor<1xf32>, tensor<f32>) -> tensor<4x32x64xf32>
  %4 = "zhigh.Stick"(%3) {layout = "3DS"} : (tensor<4x32x64xf32>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %5 = "zhigh.Add"(%4, %0) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %6 = "zhigh.Unstick"(%5) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf32>
  return %6 : tensor<4x32x64xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @clip_3_inputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x32x64xf32>, [[PARAM_1_:%.+]]: tensor<4x32x64xf32>, [[PARAM_2_:%.+]]: tensor<1xf32>, [[PARAM_3_:%.+]]: tensor<f32>) -> tensor<4x32x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<4x32x64xf32>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Add"([[VAR_0_]], [[VAR_0_]]) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Clip"([[VAR_1_]], [[PARAM_2_]], [[PARAM_3_]]) {onnx_node_name = "onnx.Clip_1"} : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1xf32>, tensor<f32>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Add"([[VAR_2_]], [[VAR_0_]]) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf32>
// CHECK:           return [[VAR_4_]] : tensor<4x32x64xf32>
// CHECK:         }
}

// -----

// Clip which has between 1 and 3 inputs (2 optional)

func.func @clip_2_inputs(%arg0: tensor<4x32x64xf32>, %arg1: tensor<4x32x64xf32>, %arg2: tensor<1xf32>, %arg3: tensor<f32>) -> tensor<4x32x64xf32> {
  %none = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
  %0 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<4x32x64xf32>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %0) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf32>
  %3 = "onnx.Clip"(%2, %arg2, %none) {onnx_node_name = "onnx.Clip_1"} : (tensor<4x32x64xf32>, tensor<1xf32>, none) -> tensor<4x32x64xf32>
  %4 = "zhigh.Stick"(%3) {layout = "3DS"} : (tensor<4x32x64xf32>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %5 = "zhigh.Add"(%4, %0) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %6 = "zhigh.Unstick"(%5) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf32>
  return %6 : tensor<4x32x64xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @clip_2_inputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x32x64xf32>, [[PARAM_1_:%.+]]: tensor<4x32x64xf32>, [[PARAM_2_:%.+]]: tensor<1xf32>, [[PARAM_3_:%.+]]: tensor<f32>) -> tensor<4x32x64xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<4x32x64xf32>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Add"([[VAR_1_]], [[VAR_1_]]) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Clip"([[VAR_2_]], [[PARAM_2_]], [[VAR_0_]]) {onnx_node_name = "onnx.Clip_1"} : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1xf32>, none) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Add"([[VAR_3_]], [[VAR_1_]]) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<4x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x32x64xf32>
// CHECK:           return [[VAR_5_]] : tensor<4x32x64xf32>
// CHECK:         }
}

// -----

// layout reshape transpose reshape layout

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
  %16 = "onnx.LayoutTransform"(%15) {target_layout = "3DS"} : (tensor<96x?x64xf16>) -> tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %17 = "zhigh.Stick"(%arg4) {layout = "2D"} : (tensor<64x64xf32>) -> tensor<64x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %18 = "zhigh.MatMul"(%16, %17, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<64x64xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %19 = "zhigh.Unstick"(%18) : (tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x?x64xf32>
  %20 = "onnx.Concat"(%3, %1, %6, %4) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %21 = "onnx.Reshape"(%19, %20) {allowzero = 0 : si64} : (tensor<96x?x64xf32>, tensor<4xi64>) -> tensor<3x32x?x64xf32>
  return %21 : tensor<3x32x?x64xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @pattern_extended_layout_transform_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x?x2048xf32>, [[PARAM_1_:%.+]]: tensor<2048x2048xf32>, [[PARAM_2_:%.+]]: tensor<2048x512xf32>, [[PARAM_3_:%.+]]: tensor<2048x512xf32>, [[PARAM_4_:%.+]]: tensor<64x64xf32>) -> tensor<3x32x?x64xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<64> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64, onnx_node_name = "onnx.Dim_0"} : (tensor<3x?x2048xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<3x?x2048xf32>) -> tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<2048x2048xf32>) -> tensor<2048x2048xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.MatMul"([[VAR_5_]], [[VAR_6_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2048x2048xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_8_:%.+]] = "zhigh.ExtendedLayoutTransform"([[VAR_7_]]) {dlf16_to_f32 = false, reshape_merge_axis = 0 : si64, reshape_split_axis = 2 : si64, reshape_split_factor = 64 : si64, target_layout = "3DS", transpose_pattern = [0, 2, 1, 3]} : (tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_9_:%.+]] = "zhigh.Stick"([[PARAM_4_]]) {layout = "2D"} : (tensor<64x64xf32>) -> tensor<64x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_10_:%.+]] = "zhigh.MatMul"([[VAR_8_]], [[VAR_9_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<64x64xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_11_:%.+]] = "zhigh.Unstick"([[VAR_10_]]) : (tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x?x64xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_1_]], [[VAR_4_]], [[VAR_3_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Reshape"([[VAR_11_]], [[VAR_12_]]) {allowzero = 0 : si64} : (tensor<96x?x64xf32>, tensor<4xi64>) -> tensor<3x32x?x64xf32>
// CHECK:           return [[VAR_13_]] : tensor<3x32x?x64xf32>
// CHECK:         }
}

// -----

// layout reshape transpose reshape layout

func.func @pattern_extended_layout_transform_v2(%arg0: tensor<3x?x4096xf32>, %arg1: tensor<4096x4096xf32>, %arg2: tensor<4096x512xf32>, %arg3: tensor<4096x512xf32>, %arg4: tensor<128x128xf32>) -> tensor<3x32x?x128xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = onnx.Constant dense<32> : tensor<1xi64>
    %2 = onnx.Constant dense<96> : tensor<1xi64>
    %3 = onnx.Constant dense<3> : tensor<1xi64>
    %4 = onnx.Constant dense<-1> : tensor<1xi64>
    %5 = onnx.Constant dense<128> : tensor<1xi64>
    %6 = "onnx.Dim"(%arg0) {axis = 1 : si64, onnx_node_name = "onnx.Dim_0"} : (tensor<3x?x4096xf32>) -> tensor<1xi64>
    %7 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<3x?x4096xf32>) -> tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %8 = "zhigh.Stick"(%arg1) {layout = "2D"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %9 = "zhigh.MatMul"(%7, %8, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4096x4096xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %10 = "onnx.LayoutTransform"(%9) : (tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x?x4096xf16>
    %11 = "onnx.Concat"(%3, %6, %4, %5) {axis = 0 : si64, onnx_node_name = "/model/layers.0/self_attn/Concat"} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %12 = "onnx.Reshape"(%10, %11) {allowzero = 0 : si64} : (tensor<3x?x4096xf16>, tensor<4xi64>) -> tensor<3x?x32x128xf16>
    %13 = "onnx.Transpose"(%12) {perm = [0, 2, 1, 3]} : (tensor<3x?x32x128xf16>) -> tensor<3x32x?x128xf16>
    %14 = "onnx.Concat"(%2, %6, %5) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %15 = "onnx.Reshape"(%13, %14) {allowzero = 0 : si64} : (tensor<3x32x?x128xf16>, tensor<3xi64>) -> tensor<96x?x128xf16>
    %16 = "onnx.LayoutTransform"(%15) {target_layout = "3DS"} : (tensor<96x?x128xf16>) -> tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %17 = "zhigh.Stick"(%arg4) {layout = "2D"} : (tensor<128x128xf32>) -> tensor<128x128xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %18 = "zhigh.MatMul"(%16, %17, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<128x128xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %19 = "zhigh.Unstick"(%18) : (tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x?x128xf32>
    %20 = "onnx.Concat"(%3, %1, %6, %5) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %21 = "onnx.Reshape"(%19, %20) {allowzero = 0 : si64} : (tensor<96x?x128xf32>, tensor<4xi64>) -> tensor<3x32x?x128xf32>
    return %21 : tensor<3x32x?x128xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @pattern_extended_layout_transform_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x?x4096xf32>, [[PARAM_1_:%.+]]: tensor<4096x4096xf32>, [[PARAM_2_:%.+]]: tensor<4096x512xf32>, [[PARAM_3_:%.+]]: tensor<4096x512xf32>, [[PARAM_4_:%.+]]: tensor<128x128xf32>) -> tensor<3x32x?x128xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<128> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64, onnx_node_name = "onnx.Dim_0"} : (tensor<3x?x4096xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<3x?x4096xf32>) -> tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.MatMul"([[VAR_5_]], [[VAR_6_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4096x4096xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_8_:%.+]] = "zhigh.ExtendedLayoutTransform"([[VAR_7_]]) {dlf16_to_f32 = false, reshape_merge_axis = 0 : si64, reshape_split_axis = 2 : si64, reshape_split_factor = 128 : si64, target_layout = "3DS", transpose_pattern = [0, 2, 1, 3]} : (tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_9_:%.+]] = "zhigh.Stick"([[PARAM_4_]]) {layout = "2D"} : (tensor<128x128xf32>) -> tensor<128x128xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_10_:%.+]] = "zhigh.MatMul"([[VAR_8_]], [[VAR_9_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<128x128xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_11_:%.+]] = "zhigh.Unstick"([[VAR_10_]]) : (tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x?x128xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_1_]], [[VAR_4_]], [[VAR_3_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Reshape"([[VAR_11_]], [[VAR_12_]]) {allowzero = 0 : si64} : (tensor<96x?x128xf32>, tensor<4xi64>) -> tensor<3x32x?x128xf32>
// CHECK:           return [[VAR_13_]] : tensor<3x32x?x128xf32>
// CHECK:         }
}

// -----

// layout reshape transpose reshape layout

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
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @pattern_extended_layout_transform_v3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x?x2048xf32>, [[PARAM_1_:%.+]]: tensor<2048x2048xf32>, [[PARAM_2_:%.+]]: tensor<2048x512xf32>, [[PARAM_3_:%.+]]: tensor<2048x512xf32>, [[PARAM_4_:%.+]]: tensor<64x64xf32>) -> tensor<3x8x?x64xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<3x?x2048xf32>) -> tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "2D"} : (tensor<2048x512xf32>) -> tensor<2048x512xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2048x512xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.ExtendedLayoutTransform"([[VAR_3_]]) {dlf16_to_f32 = true, reshape_merge_axis = -1 : si64, reshape_split_axis = 2 : si64, reshape_split_factor = 64 : si64, transpose_pattern = [0, 2, 1, 3]} : (tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x8x?x64xf32>
// CHECK:           return [[VAR_4_]] : tensor<3x8x?x64xf32>
// CHECK:         }
}