// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --canonicalize %s -split-input-file | FileCheck %s

func.func @remove_stick_and_unstick_same_layout(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

  %3 = "zhigh.Stick"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>
  "func.return"(%5) : (tensor<10x10xf32>) -> ()

  // CHECK-LABEL: remove_stick_and_unstick_same_layout
  // CHECK: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK-NOT: zhigh.Unstick
  // CHECK-NOT: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK: zhigh.Unstick
}
// -----

func.func @remove_stick_only(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

  %3 = "zhigh.Stick"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<10x10xf16, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

  %6 = "onnx.Add"(%2, %5) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%6) : (tensor<10x10xf32>) -> ()

  // CHECK-LABEL: remove_stick_only
  // CHECK: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK: zhigh.Unstick
  // CHECK-NOT: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK: zhigh.Unstick
  // CHECK: onnx.Add
}

// -----

// Replace unstick/stick by onnx.LayoutTransform because of different layout.
func.func @replace_stick_and_unstick_by_layout_transform(%arg0 : tensor<5x10x10xf32>) -> tensor<5x10x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<5x10x10xf32>) -> tensor<5x10x10xf16, #zhigh.layout<{dataLayout = "3D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<5x10x10xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<5x10x10xf16, #zhigh.layout<{dataLayout = "3D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<5x10x10xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<5x10x10xf32>

  %3 = "zhigh.Stick"(%2) {layout = "3DS"} : (tensor<5x10x10xf32>) -> tensor<5x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<5x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %5 = "zhigh.Unstick"(%4) {layout = "3DS"} : (tensor<5x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32>
  "func.return"(%5) : (tensor<5x10x10xf32>) -> ()

  // CHECK-LABEL: replace_stick_and_unstick_by_layout_transform
  // CHECK: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK: onnx.LayoutTransform
  // CHECK: zhigh.Relu
  // CHECK: zhigh.Unstick
}

// -----

// Donot replace unstick/stick by onnx.LayoutTransform because of 2ds layout.
func.func @donot_replace_stick_and_unstick_by_layout_transform(%arg0 : tensor<5x10xf32>) -> tensor<5x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<5x10xf32>) -> tensor<5x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<5x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<5x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<5x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<5x10xf32>

  %3 = "zhigh.Stick"(%2) {layout = "2DS"} : (tensor<5x10xf32>) -> tensor<5x10xf16, #zhigh.layout<{dataLayout = "2DS"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<5x10xf16, #zhigh.layout<{dataLayout = "2DS"}>>) -> tensor<5x10xf16, #zhigh.layout<{dataLayout = "2DS"}>>
  %5 = "zhigh.Unstick"(%4) {layout = "2DS"} : (tensor<5x10xf16, #zhigh.layout<{dataLayout = "2DS"}>>) -> tensor<5x10xf32>
  "func.return"(%5) : (tensor<5x10xf32>) -> ()

  // CHECK-LABEL: donot_replace_stick_and_unstick_by_layout_transform
  // CHECK: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK: zhigh.Unstick
  // CHECK: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK: zhigh.Unstick
}
// -----

// Remove Stick with NoneType input.
func.func @remove_nonetype_stick() -> () {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.Stick"(%cst) : (none) -> none
  return

  // CHECK-LABEL: remove_nonetype_stick
  // CHECK-NOT: zhigh.Stick
}

// -----

// Replace onnx.LeakyRelu
func.func @replace_leakyrelu_1(%arg0 : tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 1.000000e-01 : f32} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf32>
  %2 = "zhigh.Stick"(%1) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  return %2 : tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK-LABEL: func @replace_leakyrelu_1
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
  // CHECK-DAG:           [[VAR_1_:%.+]] = onnx.Constant dense<-1.000000e-01> : tensor<1x104x104x128xf32>
  // CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_3_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_4_:%.+]] = "zhigh.Relu"([[VAR_3_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_5_:%.+]] = "zhigh.Sub"([[VAR_0_]], [[VAR_4_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           return [[VAR_5_]] : tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
}

// -----

// Replace onnx.LeakyRelu
func.func @replace_leakyrelu_2(%arg0 : tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 1.000000e-01 : f32} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf32>
  %2 = "zhigh.Stick"(%1) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  return %2 : tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK-LABEL: func @replace_leakyrelu_2
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
  // CHECK-DAG:           [[VAR_1_:%.+]] = onnx.Constant dense<-1.000000e-01> : tensor<1x104x104x128xf32>
  // CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_3_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_4_:%.+]] = "zhigh.Relu"([[VAR_3_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_5_:%.+]] = "zhigh.Sub"([[VAR_0_]], [[VAR_4_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           return [[VAR_5_]] : tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
}

// -----

// Replace onnx.Softplus
func.func @replace_softplus_1(%arg0 : tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32>
  %1 = "onnx.Softplus"(%0) {alpha = 1.000000e-01 : f32} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf32>
  %2 = "zhigh.Stick"(%1) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  return %2 : tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @replace_softplus_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<-1.000000e+00> : tensor<1x104x104x128xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Min"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Exp"([[VAR_4_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Sub"([[VAR_5_]], [[VAR_1_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Log"([[VAR_6_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Add"([[VAR_3_]], [[VAR_7_]]) : (tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_8_]] : tensor<1x104x104x128xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

// Replace onnx.Softplus
func.func @replace_softplus_2(%arg0 : tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32>
  %1 = "onnx.Softplus"(%0) {alpha = 1.000000e-01 : f32} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf32>
  %2 = "zhigh.Stick"(%1) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  return %2 : tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @replace_softplus_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<-1.000000e+00> : tensor<1x104x104x128xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Min"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Exp"([[VAR_4_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Sub"([[VAR_5_]], [[VAR_1_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Log"([[VAR_6_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Add"([[VAR_3_]], [[VAR_7_]]) : (tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_8_]] : tensor<1x104x128x104xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @replace_reciprocal_sqrt(%arg0 : tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>) {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<4x256x1xf32>) -> tensor<4x256x1xf32>
  %2 = "onnx.Reciprocal"(%1) : (tensor<4x256x1xf32>) -> tensor<4x256x1xf32>
  %3 = "zhigh.Stick"(%2) {layout = "3D"} : (tensor<4x256x1xf32>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>
  return %3 : tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>

// CHECK-LABEL:  func.func @replace_reciprocal_sqrt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>> {
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-5.000000e-01> : tensor<4x256x1xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Log"([[PARAM_0_]]) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "3D"} : (tensor<4x256x1xf32>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Mul"([[VAR_0_]], [[VAR_2_]]) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>, tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Exp"([[VAR_3_]]) : (tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_4_]] : tensor<4x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

// Do not replace reciprocal square root because of unknown dimension.
// In this case, there is no static shape to create a constant of 1 or 2.
func.func @donot_replace_reciprocal_sqrt(%arg0 : tensor<?x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<?x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>) {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<?x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x256x1xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<?x256x1xf32>) -> tensor<?x256x1xf32>
  %2 = "onnx.Reciprocal"(%1) : (tensor<?x256x1xf32>) -> tensor<?x256x1xf32>
  %3 = "zhigh.Stick"(%2) {layout = "3D"} : (tensor<?x256x1xf32>) -> tensor<?x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>
  return %3 : tensor<?x256x1xf16, #zhigh.layout<{dataLayout = "3D"}>>

// CHECK-LABEL:  func.func @donot_replace_reciprocal_sqrt
// CHECK: zhigh.Unstick
// CHECK: onnx.Sqrt
// CHECK: onnx.Reciprocal
// CHECK: zhigh.Stick
}

// -----

// This pattern is found in bertsquart/GPT models.
// Reshape-Transpose-Reshape will be rewritten into a single onnx.ShapeTransform.
func.func @reshape_transpose_reshape_2d_to_3ds(%arg0: tensor<1024x768xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<48x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
   %0 = onnx.Constant dense<[4, 256, 12, 64]> : tensor<4xi64>
   %1 = onnx.Constant dense<[48, 256, 64]> : tensor<3xi64>
   %2 = "zhigh.Unstick"(%arg0) {layout = "2D"} : (tensor<1024x768xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1024x768xf32>
   %3 = "onnx.Reshape"(%2, %0) {allowzero = 0 : si64} : (tensor<1024x768xf32>, tensor<4xi64>) -> tensor<4x256x12x64xf32>
   %4 = "onnx.Transpose"(%3) {perm = [0, 2, 1, 3]} : (tensor<4x256x12x64xf32>) -> tensor<4x12x256x64xf32>
   %5 = "onnx.Reshape"(%4, %1) {allowzero = 0 : si64} : (tensor<4x12x256x64xf32>, tensor<3xi64>) -> tensor<48x256x64xf32>
   %6 = "zhigh.Stick"(%5) {layout = "3DS"} : (tensor<48x256x64xf32>) -> tensor<48x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
   return %6 : tensor<48x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> ((d0 floordiv 256) * 12 + d1 floordiv 64, d0 mod 256, d1 mod 64)>
// CHECK-LABEL:  func.func @reshape_transpose_reshape_2d_to_3ds
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1024x768xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<48x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<1024x768xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1024x768xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ShapeTransform"([[VAR_0_]]) {index_map = [[MAP_0_]]} : (tensor<1024x768xf32>) -> tensor<48x256x64xf32>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "3DS"} : (tensor<48x256x64xf32>) -> tensor<48x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_2_]] : tensor<48x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

func.func @reshape_transpose_reshape_3ds_to_2d(%arg0: tensor<48x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) ->  tensor<1024x768xf16, #zhigh.layout<{dataLayout = "2D"}>>{
  %shape0 = onnx.Constant dense<[4, 12, 256, 64]> : tensor<4xi64>
  %shape1 = onnx.Constant dense<[1024, 768]> : tensor<2xi64>
  %0 = "zhigh.Unstick"(%arg0) {layout = "3DS"} : (tensor<48x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<48x256x64xf32>
  %1 = "onnx.Reshape"(%0, %shape0) : (tensor<48x256x64xf32>, tensor<4xi64>) -> tensor<4x12x256x64xf32>
  %2 = "onnx.Transpose"(%1) {perm = [0, 2, 1, 3]}: (tensor<4x12x256x64xf32>) -> tensor<4x256x12x64xf32>
  %3 = "onnx.Reshape"(%2, %shape1) : (tensor<4x256x12x64xf32>, tensor<2xi64>) -> tensor<1024x768xf32>
  %4 = "zhigh.Stick"(%3) {layout = "2D"} : (tensor<1024x768xf32>) -> tensor<1024x768xf16, #zhigh.layout<{dataLayout = "2D"}>>
  return %4 : tensor<1024x768xf16, #zhigh.layout<{dataLayout = "2D"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> ((d0 floordiv 12) * 256 + d1, (d0 mod 12) * 64 + d2)>
// CHECK-LABEL:  func.func @reshape_transpose_reshape_3ds_to_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<48x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1024x768xf16, #zhigh.layout<{dataLayout = "2D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<48x256x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<48x256x64xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ShapeTransform"([[VAR_0_]]) {index_map = [[MAP_0_]]} : (tensor<48x256x64xf32>) -> tensor<1024x768xf32>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "2D"} : (tensor<1024x768xf32>) -> tensor<1024x768xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           return [[VAR_2_]] : tensor<1024x768xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:         }
}

// -----

func.func @test_unstick_dim(%arg0: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>, tensor<1xi64>) {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32>
  %1 = "zhigh.Relu"(%arg0) : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>)
  %2 = "onnx.Dim"(%0) {axis = 1 : si64}: (tensor<?x?x?xf32>) -> tensor<1xi64>
  return %1, %2 : tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>, tensor<1xi64>

// CHECK-LABEL:  func.func @test_unstick_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>, tensor<1xi64>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<1xi64>
// CHECK:           return [[VAR_0_]], [[VAR_1_]] : tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3D"}>>, tensor<1xi64>
// CHECK:         }
}

// -----

func.func @test_unstick_dim_nchw(%arg0: tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1xi64>) {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<?x?x?x?xf32>
  %1 = "zhigh.Relu"(%arg0) : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>)
  %2 = "onnx.Dim"(%0) {axis = 1 : si64}: (tensor<?x?x?x?xf32>) -> tensor<1xi64>
  return %1, %2 : tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1xi64>

// CHECK-LABEL:  func.func @test_unstick_dim_nchw
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1xi64>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 3 : si64} : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1xi64>
// CHECK:           return [[VAR_0_]], [[VAR_1_]] : tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1xi64>
// CHECK:         }
}

// -----

// COM: Remove all LayoutTransform and data conversion since the whole computation is identity.
func.func @test_dlf16_to_f32(%arg0: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.LayoutTransform"(%arg0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16>
  %1 = "zhigh.DLF16ToF32"(%0) : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf32>
  %2 = "zhigh.F32ToDLF16"(%1) : (tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf16>
  %3 = "onnx.LayoutTransform"(%2) {target_layout = "4D"} : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
  onnx.Return %3 : tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @test_dlf16_to_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           onnx.Return [[PARAM_0_]] : tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

// COM: DLF16ToF32 is delayed and pushed down through Reshape and Transpose,
// and it is removed when combined with F32ToDLF16.
func.func @test_delay_dlf16_to_f32(%arg0: tensor<1x3x5x?xf16>, %arg1: tensor<3xi64>) -> tensor<5x3x?xf16> {
  %1 = "zhigh.DLF16ToF32"(%arg0) : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf32>
  %2 = "onnx.Reshape"(%1, %arg1) {allowzero = 0 : si64} : (tensor<1x3x5x?xf32>, tensor<3xi64>) -> tensor<3x5x?xf32>
  %3 = "onnx.Transpose"(%2) {perm = [1, 0, 2]} : (tensor<3x5x?xf32>) -> tensor<5x3x?xf32>
  %4 = "zhigh.F32ToDLF16"(%3) : (tensor<5x3x?xf32>) -> tensor<5x3x?xf16>
  onnx.Return %4 : tensor<5x3x?xf16>

// CHECK-LABEL:  func.func @test_delay_dlf16_to_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16>, [[PARAM_1_:%.+]]: tensor<3xi64>) -> tensor<5x3x?xf16> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[PARAM_1_]]) {allowzero = 0 : si64} : (tensor<1x3x5x?xf16>, tensor<3xi64>) -> tensor<3x5x?xf16>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Transpose"([[VAR_0_]]) {perm = [1, 0, 2]} : (tensor<3x5x?xf16>) -> tensor<5x3x?xf16>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<5x3x?xf16>
// CHECK:         }
}

// -----

// COM: Roberta pattern with BS=1

func.func @test_Roberta_bs1(%arg0: tensor<12x384x384xf32>, %arg1: tensor<12x384x64xf32>, %arg2: tensor<768x768xf32>) -> tensor<1x384x768xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %2 = onnx.Constant dense<[1, 12, 384, 64]> : tensor<4xi64>
  %9 = onnx.Constant dense<[1, 384, 768]> : tensor<3xi64>
  %76 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<12x384x384xf32>) -> tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %77 = "zhigh.Stick"(%arg1) {layout = "3DS"} : (tensor<12x384x64xf32>) -> tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %78 = "zhigh.MatMul"(%76, %77, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %79 = "zhigh.Unstick"(%78) : (tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<12x384x64xf32>
  %80 = "onnx.Reshape"(%79, %2) {allowzero = 0 : si64} : (tensor<12x384x64xf32>, tensor<4xi64>) -> tensor<1x12x384x64xf32>
  %81 = "onnx.Transpose"(%80) {onnx_node_name = "Transpose_94", perm = [0, 2, 1, 3]} : (tensor<1x12x384x64xf32>) -> tensor<1x384x12x64xf32>
  %82 = "onnx.Reshape"(%81, %9) {allowzero = 0 : si64, onnx_node_name = "Reshape_104"} : (tensor<1x384x12x64xf32>, tensor<3xi64>) -> tensor<1x384x768xf32>
  %83 = "zhigh.Stick"(%82) {layout = "3DS"} : (tensor<1x384x768xf32>) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %84 = "zhigh.Stick"(%arg2) {layout = "2D"} : (tensor<768x768xf32>) -> tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %85 = "zhigh.MatMul"(%83, %84, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %86 = "zhigh.Unstick"(%85) : (tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x384x768xf32>
  onnx.Return %86 : tensor<1x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_Roberta_bs1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<12x384x384xf32>, [[PARAM_1_:%.+]]: tensor<12x384x64xf32>, [[PARAM_2_:%.+]]: tensor<768x768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 384, 768]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<12x384x384xf32>) -> tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<12x384x64xf32>) -> tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.MatMul"([[VAR_2_]], [[VAR_3_]], [[VAR_1_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Reshape"([[VAR_4_]], [[VAR_0_]]) {layout = "3DS"} : (tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<3xi64>) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "2D"} : (tensor<768x768xf32>) -> tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.MatMul"([[VAR_5_]], [[VAR_6_]], [[VAR_1_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Unstick"([[VAR_7_]]) : (tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x384x768xf32>
// CHECK:           onnx.Return [[VAR_8_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// COM: Roberta pattern with BS=8

func.func @test_Roberta_bs8(%arg0: tensor<96x384x384xf32>, %arg1: tensor<96x384x64xf32>, %arg2: tensor<768x768xf32>) -> tensor<8x384x768xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %2 = onnx.Constant dense<[8, 12, 384, 64]> : tensor<4xi64>
  %9 = onnx.Constant dense<[8, 384, 768]> : tensor<3xi64>
  %76 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<96x384x384xf32>) -> tensor<96x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %77 = "zhigh.Stick"(%arg1) {layout = "3DS"} : (tensor<96x384x64xf32>) -> tensor<96x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %78 = "zhigh.MatMul"(%76, %77, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<96x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<96x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<96x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %79 = "zhigh.Unstick"(%78) : (tensor<96x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x384x64xf32>
  %80 = "onnx.Reshape"(%79, %2) {allowzero = 0 : si64} : (tensor<96x384x64xf32>, tensor<4xi64>) -> tensor<8x12x384x64xf32>
  %81 = "onnx.Transpose"(%80) {onnx_node_name = "Transpose_94", perm = [0, 2, 1, 3]} : (tensor<8x12x384x64xf32>) -> tensor<8x384x12x64xf32>
  %82 = "onnx.Reshape"(%81, %9) {allowzero = 0 : si64, onnx_node_name = "Reshape_104"} : (tensor<8x384x12x64xf32>, tensor<3xi64>) -> tensor<8x384x768xf32>
  %83 = "zhigh.Stick"(%82) {layout = "3DS"} : (tensor<8x384x768xf32>) -> tensor<8x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %84 = "zhigh.Stick"(%arg2) {layout = "2D"} : (tensor<768x768xf32>) -> tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %85 = "zhigh.MatMul"(%83, %84, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<8x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<8x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %86 = "zhigh.Unstick"(%85) : (tensor<8x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x384x768xf32>
  onnx.Return %86 : tensor<8x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_Roberta_bs8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<96x384x384xf32>, [[PARAM_1_:%.+]]: tensor<96x384x64xf32>, [[PARAM_2_:%.+]]: tensor<768x768xf32>) -> tensor<8x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[8, 384, 768]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<96x384x384xf32>) -> tensor<96x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<96x384x64xf32>) -> tensor<96x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.MatMul"([[VAR_2_]], [[VAR_3_]], [[VAR_1_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<96x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<96x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<96x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Reshape"([[VAR_4_]], [[VAR_0_]]) {layout = "3DS"} : (tensor<96x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<3xi64>) -> tensor<8x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "2D"} : (tensor<768x768xf32>) -> tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.MatMul"([[VAR_5_]], [[VAR_6_]], [[VAR_1_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<8x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<8x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Unstick"([[VAR_7_]]) : (tensor<8x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x384x768xf32>
// CHECK:           onnx.Return [[VAR_8_]] : tensor<8x384x768xf32>
// CHECK:         }
}

// -----

// COM: Roberta pattern with BS=1 but dim 2 (385) is not mod 32 = 0; should fail to apply pattern

func.func @test_Roberta_bs1_not_mod32(%arg0: tensor<12x385x385xf32>, %arg1: tensor<12x385x64xf32>, %arg2: tensor<768x768xf32>) -> tensor<1x385x768xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %2 = onnx.Constant dense<[1, 12, 384, 64]> : tensor<4xi64>
  %9 = onnx.Constant dense<[1, 384, 768]> : tensor<3xi64>
  %76 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<12x385x385xf32>) -> tensor<12x385x385xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %77 = "zhigh.Stick"(%arg1) {layout = "3DS"} : (tensor<12x385x64xf32>) -> tensor<12x385x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %78 = "zhigh.MatMul"(%76, %77, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<12x385x385xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<12x385x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<12x385x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %79 = "zhigh.Unstick"(%78) : (tensor<12x385x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<12x385x64xf32>
  %80 = "onnx.Reshape"(%79, %2) {allowzero = 0 : si64} : (tensor<12x385x64xf32>, tensor<4xi64>) -> tensor<1x12x385x64xf32>
  %81 = "onnx.Transpose"(%80) {onnx_node_name = "Transpose_94", perm = [0, 2, 1, 3]} : (tensor<1x12x385x64xf32>) -> tensor<1x385x12x64xf32>
  %82 = "onnx.Reshape"(%81, %9) {allowzero = 0 : si64, onnx_node_name = "Reshape_104"} : (tensor<1x385x12x64xf32>, tensor<3xi64>) -> tensor<1x385x768xf32>
  %83 = "zhigh.Stick"(%82) {layout = "3DS"} : (tensor<1x385x768xf32>) -> tensor<1x385x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %84 = "zhigh.Stick"(%arg2) {layout = "2D"} : (tensor<768x768xf32>) -> tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %85 = "zhigh.MatMul"(%83, %84, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<1x385x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<1x385x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %86 = "zhigh.Unstick"(%85) : (tensor<1x385x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x385x768xf32>
  onnx.Return %86 : tensor<1x385x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_Roberta_bs1_not_mod32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<12x385x385xf32>, [[PARAM_1_:%.+]]: tensor<12x385x64xf32>, [[PARAM_2_:%.+]]: tensor<768x768xf32>) -> tensor<1x385x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 12, 384, 64]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 384, 768]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<12x385x385xf32>) -> tensor<12x385x385xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<12x385x64xf32>) -> tensor<12x385x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.MatMul"([[VAR_3_]], [[VAR_4_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<12x385x385xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<12x385x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<12x385x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<12x385x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<12x385x64xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Reshape"([[VAR_6_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<12x385x64xf32>, tensor<4xi64>) -> tensor<1x12x385x64xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {onnx_node_name = "Transpose_94", perm = [0, 2, 1, 3]} : (tensor<1x12x385x64xf32>) -> tensor<1x385x12x64xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Reshape"([[VAR_8_]], [[VAR_2_]]) {allowzero = 0 : si64, onnx_node_name = "Reshape_104"} : (tensor<1x385x12x64xf32>, tensor<3xi64>) -> tensor<1x385x768xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "zhigh.Stick"([[VAR_9_]]) {layout = "3DS"} : (tensor<1x385x768xf32>) -> tensor<1x385x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_11_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "2D"} : (tensor<768x768xf32>) -> tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_12_:%.+]] = "zhigh.MatMul"([[VAR_10_]], [[VAR_11_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<1x385x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<1x385x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_13_:%.+]] = "zhigh.Unstick"([[VAR_12_]]) : (tensor<1x385x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x385x768xf32>
// CHECK:           onnx.Return [[VAR_13_]] : tensor<1x385x768xf32>
// CHECK:         }
}

// -----

// COM second pattern found in roberta, with BS=1

func.func @test_Roberta_pattern2_bs1(%arg0: tensor<1x384x768xf32>, %arg1: tensor<1x384x768xf32>, %arg2: tensor<1x384x768xf32>) -> tensor<12x384x64xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %2 = onnx.Constant dense<[-1, 384, 64]> : tensor<3xi64>
    %8 = onnx.Constant dense<[1, 384, 12, 64]> : tensor<4xi64>
    %48 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<1x384x768xf32>) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %49 = "zhigh.Stick"(%arg1) {layout = "3DS"} : (tensor<1x384x768xf32>) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %63 = "zhigh.Stick"(%arg2) {layout = "3DS"} : (tensor<1x384x768xf32>) -> tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %50 = "zhigh.Add"(%48, %49) : (tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %51 = "zhigh.Unstick"(%50) : (tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x384x768xf32>
    %55 = "onnx.Reshape"(%51, %8) {allowzero = 0 : si64, onnx_node_name = "Reshape_85"} : (tensor<1x384x768xf32>, tensor<4xi64>) -> tensor<1x384x12x64xf32>
    %56 = "onnx.Transpose"(%55) {onnx_node_name = "Transpose_86", perm = [0, 2, 1, 3]} : (tensor<1x384x12x64xf32>) -> tensor<1x12x384x64xf32>
    %64 = "onnx.Reshape"(%56, %2) {allowzero = 0 : si64} : (tensor<1x12x384x64xf32>, tensor<3xi64>) -> tensor<12x384x64xf32>
    %65 = "zhigh.Stick"(%64) {layout = "3DS"} : (tensor<12x384x64xf32>) -> tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %66 = "zhigh.MatMul"(%63, %65, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %73 = "zhigh.Unstick"(%66) : (tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<12x384x64xf32>
  onnx.Return %73 : tensor<12x384x64xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_Roberta_pattern2_bs1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<1x384x768xf32>, [[PARAM_2_:%.+]]: tensor<1x384x768xf32>) -> tensor<12x384x64xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[12, 384, 64]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x384x768xf32>) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<1x384x768xf32>) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "3DS"} : (tensor<1x384x768xf32>) -> tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Add"([[VAR_2_]], [[VAR_3_]]) : (tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Reshape"([[VAR_5_]], [[VAR_0_]]) {layout = "3DS"} : (tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<3xi64>) -> tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.MatMul"([[VAR_4_]], [[VAR_6_]], [[VAR_1_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Unstick"([[VAR_7_]]) : (tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<12x384x64xf32>
// CHECK:           onnx.Return [[VAR_8_]] : tensor<12x384x64xf32>
// CHECK:         }
}

// -----

// COM second pattern found in roberta, with BS=1, not mod 64

func.func @test_Roberta_pattern2_bs1_notmod64(%arg0: tensor<1x384x756xf32>, %arg1: tensor<1x384x756xf32>, %arg2: tensor<1x384x756xf32>) -> tensor<12x384x63xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %2 = onnx.Constant dense<[-1, 384, 64]> : tensor<3xi64>
    %8 = onnx.Constant dense<[1, 384, 12, 64]> : tensor<4xi64>
    %48 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<1x384x756xf32>) -> tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %49 = "zhigh.Stick"(%arg1) {layout = "3DS"} : (tensor<1x384x756xf32>) -> tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %63 = "zhigh.Stick"(%arg2) {layout = "3DS"} : (tensor<1x384x756xf32>) -> tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %50 = "zhigh.Add"(%48, %49) : (tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %51 = "zhigh.Unstick"(%50) : (tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x384x756xf32>
    %55 = "onnx.Reshape"(%51, %8) {allowzero = 0 : si64, onnx_node_name = "Reshape_85"} : (tensor<1x384x756xf32>, tensor<4xi64>) -> tensor<1x384x12x63xf32>
    %56 = "onnx.Transpose"(%55) {onnx_node_name = "Transpose_86", perm = [0, 2, 1, 3]} : (tensor<1x384x12x63xf32>) -> tensor<1x12x384x63xf32>
    %64 = "onnx.Reshape"(%56, %2) {allowzero = 0 : si64} : (tensor<1x12x384x63xf32>, tensor<3xi64>) -> tensor<12x384x63xf32>
    %65 = "zhigh.Stick"(%64) {layout = "3DS"} : (tensor<12x384x63xf32>) -> tensor<12x384x63xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %66 = "zhigh.MatMul"(%63, %65, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<12x384x63xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<12x384x63xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %73 = "zhigh.Unstick"(%66) : (tensor<12x384x63xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<12x384x63xf32>
  onnx.Return %73 : tensor<12x384x63xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_Roberta_pattern2_bs1_notmod64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x756xf32>, [[PARAM_1_:%.+]]: tensor<1x384x756xf32>, [[PARAM_2_:%.+]]: tensor<1x384x756xf32>) -> tensor<12x384x63xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[-1, 384, 64]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 384, 12, 64]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x384x756xf32>) -> tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<1x384x756xf32>) -> tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "3DS"} : (tensor<1x384x756xf32>) -> tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Add"([[VAR_3_]], [[VAR_4_]]) : (tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Unstick"([[VAR_6_]]) : (tensor<1x384x756xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x384x756xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Reshape"([[VAR_7_]], [[VAR_2_]]) {allowzero = 0 : si64, onnx_node_name = "Reshape_85"} : (tensor<1x384x756xf32>, tensor<4xi64>) -> tensor<1x384x12x63xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {onnx_node_name = "Transpose_86", perm = [0, 2, 1, 3]} : (tensor<1x384x12x63xf32>) -> tensor<1x12x384x63xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x12x384x63xf32>, tensor<3xi64>) -> tensor<12x384x63xf32>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.Stick"([[VAR_10_]]) {layout = "3DS"} : (tensor<12x384x63xf32>) -> tensor<12x384x63xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_12_:%.+]] = "zhigh.MatMul"([[VAR_5_]], [[VAR_11_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<12x384x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<12x384x63xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<12x384x63xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_13_:%.+]] = "zhigh.Unstick"([[VAR_12_]]) : (tensor<12x384x63xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<12x384x63xf32>
// CHECK:           onnx.Return [[VAR_13_]] : tensor<12x384x63xf32>
// CHECK:         }
}

// -----

func.func @replace_unstick_squeeze_stick_static(%arg0: tensor<7x1x128x200xf16, #zhigh.layout<{dataLayout = "4DS"}>>) -> tensor<7x128x200xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
   %cst1 = onnx.Constant dense<1> : tensor<1xi64>
   %0 = "zhigh.Unstick"(%arg0) : (tensor<7x1x128x200xf16, #zhigh.layout<{dataLayout = "4DS"}>>) -> tensor<7x1x128x200xf32>
   %1 = "onnx.Squeeze"(%0, %cst1) : (tensor<7x1x128x200xf32>, tensor<1xi64>) -> tensor<7x128x200xf32>
   %2 = "zhigh.Stick"(%1) {layout = "3DS"} : (tensor<7x128x200xf32>) -> tensor<7x128x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  "func.return"(%2) : (tensor<7x128x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()

// CHECK-LABEL:  func.func @replace_unstick_squeeze_stick_static
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<7x1x128x200xf16, #zhigh.layout<{dataLayout = "4DS"}>>) -> tensor<7x128x200xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[7, 128, 200]> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Reshape"([[PARAM_0_]], [[VAR_0_]]) {layout = "3DS"} : (tensor<7x1x128x200xf16, #zhigh.layout<{dataLayout = "4DS"}>>, tensor<3xi64>) -> tensor<7x128x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_1_]] : tensor<7x128x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

func.func @replace_unstick_squeeze_stick_dynamic(%arg0: tensor<?x1x?x200xf16, #zhigh.layout<{dataLayout = "4DS"}>>) -> tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
   %cst1 = onnx.Constant dense<1> : tensor<1xi64>
   %0 = "zhigh.Unstick"(%arg0) : (tensor<?x1x?x200xf16, #zhigh.layout<{dataLayout = "4DS"}>>) -> tensor<?x1x?x200xf32>
   %1 = "onnx.Squeeze"(%0, %cst1) : (tensor<?x1x?x200xf32>, tensor<1xi64>) -> tensor<?x?x200xf32>
   %2 = "zhigh.Stick"(%1) {layout = "3DS"} : (tensor<?x?x200xf32>) -> tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  "func.return"(%2) : (tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()

// CHECK-LABEL:  func.func @replace_unstick_squeeze_stick_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x1x?x200xf16, #zhigh.layout<{dataLayout = "4DS"}>>) -> tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x1x?x200xf16, #zhigh.layout<{dataLayout = "4DS"}>>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 2 : si64} : (tensor<?x1x?x200xf16, #zhigh.layout<{dataLayout = "4DS"}>>) -> tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Reshape"([[PARAM_0_]], [[VAR_3_]]) {layout = "3DS"} : (tensor<?x1x?x200xf16, #zhigh.layout<{dataLayout = "4DS"}>>, tensor<3xi64>) -> tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_4_]] : tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

