// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --zhigh-layout-prop --shape-inference %s -split-input-file | FileCheck %s

func.func @add_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %3 = "zhigh.Add"(%1, %2) : (tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @add_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @add_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %3 = "zhigh.Add"(%2, %1) : (tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @add_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @sub_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %3 = "zhigh.Sub"(%1, %2) : (tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @sub_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Sub"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @sub_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %3 = "zhigh.Sub"(%2, %1) : (tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @sub_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Sub"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @mul_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %3 = "zhigh.Mul"(%1, %2) : (tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @mul_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @mul_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %3 = "zhigh.Mul"(%2, %1) : (tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @mul_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @div_layout_propagate_nhwc_1(%arg0: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %3 = "zhigh.Div"(%1, %2) : (tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @div_layout_propagate_nhwc_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Div"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @div_layout_propagate_nhwc_2(%arg0: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%arg1) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %3 = "zhigh.Div"(%2, %1) : (tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %3 : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @div_layout_propagate_nhwc_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "NHWC"} : (tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Div"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_3_]] : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @relu_layout_propagate_nhwc(%arg0: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
  %1 = "zhigh.Stick"(%0) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Relu"(%1) : (tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %2 : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @relu_layout_propagate_nhwc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Unstick"([[VAR_0_]]) : (tensor<1x56x56x256xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x256x56x56xf32>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "4D"} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_2_]] : tensor<1x256x56x56xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

func.func @onnx_concat_layout_propagation_nhwc(%arg0: tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<?x192x4x4xf32>
  %1 = "zhigh.Unstick"(%arg1) : (tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<?x192x4x4xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = 1 : si64} : (tensor<?x192x4x4xf32>, tensor<?x192x4x4xf32>) -> tensor<?x384x4x4xf32>
  %3 = "zhigh.Stick"(%2) {layout = "NHWC"} : (tensor<?x384x4x4xf32>) -> tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  return %3 : tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "NHWC"}>>

// CHECK-LABEL:  func.func @onnx_concat_layout_propagation_nhwc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]]) {axis = 3 : si64} : (tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @onnx_concat_layout_propagation_4d(%arg0: tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x4x4x192xf32>
  %1 = "zhigh.Unstick"(%arg1) : (tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x4x4x192xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = 3 : si64} : (tensor<?x4x4x192xf32>, tensor<?x4x4x192xf32>) -> tensor<?x4x4x384xf32>
  %3 = "zhigh.Stick"(%2) {layout = "4D"} : (tensor<?x4x4x384xf32>) -> tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %3 : tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @onnx_concat_layout_propagation_4d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "4D"}>>, [[PARAM_1_:%.+]]: tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]]) {axis = 3 : si64} : (tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<?x4x4x192xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x4x4x384xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

// TODO: enable this once DLFLOAT16-based calculation is supported.
// Data layout propagation for ONNX operations.
// Take ONNXSqrtOp as the representative of unary element-wise ops.
// COM: func.func @test_onnx_sqrt_ztensor(%arg0: tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// COM:   %0 = "zhigh.Unstick"(%arg0) : (tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf32>
// COM:   %1 = "onnx.Sqrt"(%0) : (tensor<?x3x5x7xf32>) -> tensor<?x3x5x7xf32>
// COM:   %2 = "zhigh.Stick"(%1) {layout = "4D"} : (tensor<?x3x5x7xf32>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
// COM:   return %2 : tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
// COM: 
// COM: // CHECK-LABEL:  func.func @test_onnx_sqrt_ztensor
// COM: // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// COM: // CHECK:           [[VAR_0_:%.+]] = "onnx.Sqrt"([[PARAM_0_]]) : (tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
// COM: // CHECK:           return [[VAR_0_]] : tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
// COM: // CHECK:         }
// COM: }
// COM: 
// COM: // -----
// COM: 
// COM: // Data layout propagation for ONNX operations.
// COM: // Take ONNXAddOp as the representative of binary element-wise ops.
// COM: func.func @test_onnx_add_ztensor(%arg0: tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// COM:   %0 = "zhigh.Unstick"(%arg0) : (tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf32>
// COM:   %1 = "zhigh.Unstick"(%arg1) : (tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x1xf32>
// COM:   %2 = "onnx.Add"(%0, %1) : (tensor<?x3x5x7xf32>, tensor<?x3x5x1xf32>) -> tensor<?x3x5x7xf32>
// COM:   %3 = "zhigh.Stick"(%2) {layout = "4D"} : (tensor<?x3x5x7xf32>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
// COM:   return %3 : tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
// COM: }
