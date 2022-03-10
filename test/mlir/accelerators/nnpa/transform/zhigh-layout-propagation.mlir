// RUN: onnx-mlir-opt --zhigh-layout-prop --shape-inference %s -split-input-file | FileCheck %s

// -----

func @add_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Stick"(%1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %4 = "zhigh.Add"(%2, %3) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %4 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func @add_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 3, 1]} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Add"([[VAR_1_]], [[PARAM_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_5_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func @add_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Stick"(%1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %4 = "zhigh.Add"(%3, %2) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %4 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func @add_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 3, 1]} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Add"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_5_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func @sub_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Stick"(%1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %4 = "zhigh.Sub"(%2, %3) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %4 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func @sub_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 3, 1]} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Sub"([[VAR_1_]], [[PARAM_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_5_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func @sub_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Stick"(%1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %4 = "zhigh.Sub"(%3, %2) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %4 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func @sub_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 3, 1]} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_5_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func @mul_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Stick"(%1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %4 = "zhigh.Mul"(%2, %3) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %4 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func @mul_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 3, 1]} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Mul"([[VAR_1_]], [[PARAM_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_5_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func @mul_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Stick"(%1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %4 = "zhigh.Mul"(%3, %2) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %4 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func @mul_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 3, 1]} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_5_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func @div_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Stick"(%1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %4 = "zhigh.Div"(%2, %3) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %4 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func @div_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 3, 1]} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Div"([[VAR_1_]], [[PARAM_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_5_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func @div_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Stick"(%1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %4 = "zhigh.Div"(%3, %2) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %4 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func @div_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 3, 1]} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Div"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Stick"([[VAR_4_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_5_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}


// -----

func @relu_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
  %2 = "zhigh.Stick"(%1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Relu"(%2) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func @relu_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Unstick"([[VAR_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Transpose"([[VAR_1_]]) {perm = [0, 3, 1, 2]} : (tensor<1x56x56x256xf32>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}
