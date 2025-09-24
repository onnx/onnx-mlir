// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// -----

// inputs stick, output stick

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
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_3_]]) {onnx_node_name = "onnx.Add_2"} : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Add"([[VAR_4_]], [[VAR_0_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<4x256x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
// CHECK:           return [[VAR_6_]] : tensor<4x256x256xf32>
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

// input normal/stick, output normal

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
// CHECK:           [[VAR_2_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_1_]]) {onnx_node_name = "onnx.Add_1"} : (tensor<4x256x256xf32>, tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<4x256x256xf32>
// CHECK:           return [[VAR_2_]] : tensor<4x256x256xf32>
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
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_3_]]) {onnx_node_name = "onnx.Add_2"} : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<256x64xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Div"([[VAR_4_]], [[VAR_1_]]) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<8x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x256x64xf32>
// CHECK:           return [[VAR_6_]] : tensor<8x256x64xf32>
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

