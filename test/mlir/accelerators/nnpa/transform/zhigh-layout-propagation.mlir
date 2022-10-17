// RUN: onnx-mlir-opt --maccel=NNPA --zhigh-layout-prop --shape-inference %s -split-input-file | FileCheck %s

func.func @add_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Add"(%1, %2) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @add_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @add_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Add"(%2, %1) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @add_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @sub_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Sub"(%1, %2) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @sub_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Sub"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @sub_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Sub"(%2, %1) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @sub_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Sub"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @mul_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Mul"(%1, %2) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @mul_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @mul_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Mul"(%2, %1) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @mul_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @div_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Div"(%1, %2) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @div_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Div"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @div_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %3 = "zhigh.Div"(%2, %1) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @div_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Div"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @relu_layout_propagate_nhwc(%arg0: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  %2 = "zhigh.Relu"(%1) : (tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %2 : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @relu_layout_propagate_nhwc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Unstick"([[VAR_0_]]) : (tensor<1x56x56x256xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_2_]] : tensor<1x256x56x56xf32, #zhigh.encoding<{dataLayout = "4D"}>>
// CHECK:         }
}
