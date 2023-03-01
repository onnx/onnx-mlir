// RUN: onnx-mlir-opt --maccel=NNPA --canonicalize %s -split-input-file | FileCheck %s

func.func @remove_stick_and_unstick_same_layout(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

  %3 = "zhigh.Stick"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>
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
  %0 = "zhigh.Stick"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

  %3 = "zhigh.Stick"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<10x10xf32, #zhigh.layout<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

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
  %0 = "zhigh.Stick"(%arg0) : (tensor<5x10x10xf32>) -> tensor<5x10x10xf32, #zhigh.layout<{dataLayout = "3D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<5x10x10xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<5x10x10xf32, #zhigh.layout<{dataLayout = "3D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<5x10x10xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<5x10x10xf32>

  %3 = "zhigh.Stick"(%2) {layout = "3DS"} : (tensor<5x10x10xf32>) -> tensor<5x10x10xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<5x10x10xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %5 = "zhigh.Unstick"(%4) {layout = "3DS"} : (tensor<5x10x10xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32>
  "func.return"(%5) : (tensor<5x10x10xf32>) -> ()

  // CHECK-LABEL: replace_stick_and_unstick_by_layout_transform
  // CHECK: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK: onnx.LayoutTransform
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
func.func @replace_leakyrelu_1(%arg0 : tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 1.000000e-01 : f32} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf32>
  %2 = "zhigh.Stick"(%1) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  return %2 : tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK-LABEL: func @replace_leakyrelu_1
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>> {
  // CHECK-DAG:           [[VAR_1_:%.+]] = onnx.Constant dense<-1.000000e-01> : tensor<1x104x104x128xf32>
  // CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_3_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_4_:%.+]] = "zhigh.Relu"([[VAR_3_]]) : (tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_5_:%.+]] = "zhigh.Sub"([[VAR_0_]], [[VAR_4_]]) : (tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           return [[VAR_5_]] : tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
}

// -----

// Replace onnx.LeakyRelu
func.func @replace_leakyrelu_2(%arg0 : tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 1.000000e-01 : f32} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf32>
  %2 = "zhigh.Stick"(%1) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  return %2 : tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK-LABEL: func @replace_leakyrelu_2
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>> {
  // CHECK-DAG:           [[VAR_1_:%.+]] = onnx.Constant dense<-1.000000e-01> : tensor<1x104x104x128xf32>
  // CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_3_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_4_:%.+]] = "zhigh.Relu"([[VAR_3_]]) : (tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_5_:%.+]] = "zhigh.Sub"([[VAR_0_]], [[VAR_4_]]) : (tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           return [[VAR_5_]] : tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
}

// -----

// Do not replace onnx.LeakyRelu if alpha < 0
func.func @donot_replace_leakyrelu(%arg0 : tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = -1.000000e-01 : f32} : (tensor<1x104x128x104xf32>) -> tensor<1x104x128x104xf32>
  %2 = "zhigh.Stick"(%1) {layout = "NHWC"} : (tensor<1x104x128x104xf32>) -> tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  return %2 : tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK-LABEL: donot_replace_leakyrelu
  // CHECK: zhigh.Unstick
  // CHECK: onnx.LeakyRelu
  // CHECK: zhigh.Stick
}

// -----

func.func @replace_sqrt(%arg0 : tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<4x256x1xf32>) -> tensor<4x256x1xf32>
  %2 = "zhigh.Stick"(%1) {layout = "3D"} : (tensor<4x256x1xf32>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
  return %2 : tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>

// CHECK-LABEL:  func.func @replace_sqrt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Log"([[PARAM_0_]]) : (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "3D"} : (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Mul"([[VAR_0_]], [[VAR_2_]]) : (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>, tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Exp"([[VAR_3_]]) : (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_4_]] : tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

// Do not replace square root because of unknown dimension.
// In this case, there is no static shape to create a constant of 2.
func.func @donot_replace_sqrt(%arg0 : tensor<?x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<?x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<?x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x256x1xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<?x256x1xf32>) -> tensor<?x256x1xf32>
  %2 = "zhigh.Stick"(%1) {layout = "3D"} : (tensor<?x256x1xf32>) -> tensor<?x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
  return %2 : tensor<?x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>

// CHECK-LABEL:  func.func @donot_replace_sqrt
// CHECK: zhigh.Unstick
// CHECK: onnx.Sqrt
// CHECK: zhigh.Stick
}

// -----

func.func @replace_reciprocal_sqrt(%arg0 : tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<4x256x1xf32>) -> tensor<4x256x1xf32>
  %2 = "onnx.Reciprocal"(%1) : (tensor<4x256x1xf32>) -> tensor<4x256x1xf32>
  %3 = "zhigh.Stick"(%2) {layout = "3D"} : (tensor<4x256x1xf32>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
  return %3 : tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>

// CHECK-LABEL:  func.func @replace_reciprocal_sqrt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Log"([[PARAM_0_]]) : (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-5.000000e-01> : tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "3D"} : (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Mul"([[VAR_0_]], [[VAR_2_]]) : (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>, tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Exp"([[VAR_3_]]) : (tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_4_]] : tensor<4x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

// Do not replace reciprocal square root because of unknown dimension.
// In this case, there is no static shape to create a constant of 1 or 2.
func.func @donot_replace_reciprocal_sqrt(%arg0 : tensor<?x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<?x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<?x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x256x1xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<?x256x1xf32>) -> tensor<?x256x1xf32>
  %2 = "onnx.Reciprocal"(%1) : (tensor<?x256x1xf32>) -> tensor<?x256x1xf32>
  %3 = "zhigh.Stick"(%2) {layout = "3D"} : (tensor<?x256x1xf32>) -> tensor<?x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>
  return %3 : tensor<?x256x1xf32, #zhigh.layout<{dataLayout = "3D"}>>

// CHECK-LABEL:  func.func @donot_replace_reciprocal_sqrt
// CHECK: zhigh.Unstick
// CHECK: onnx.Sqrt
// CHECK: onnx.Reciprocal
// CHECK: zhigh.Stick
}
