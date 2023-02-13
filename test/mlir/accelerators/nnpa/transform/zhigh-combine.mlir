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
  // CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_1_:%.+]] = onnx.Constant dense<-1.000000e-01> : tensor<1x104x104x128xf32>
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
  // CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.layout<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_1_:%.+]] = onnx.Constant dense<-1.000000e-01> : tensor<1x104x104x128xf32>
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

// Composition of ShapeTransform.
#transpose = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1, d3)>
#reshape =  affine_map<(d0, d1) -> (d0 floordiv 32, d0 mod 32, d1 floordiv 64, d1 mod 64)> 
func.func @transpose(%arg0: tensor<128x128xf32>) -> tensor<2x4x32x64xf32> {
  %0 = "zhigh.ShapeTransform"(%arg0) {index_map = #reshape} : (tensor<128x128xf32>) -> tensor<4x32x2x64xf32>
  %1 = "zhigh.ShapeTransform"(%0) {index_map = #transpose} : (tensor<4x32x2x64xf32>) -> tensor<2x4x32x64xf32>
  return %1 : tensor<2x4x32x64xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d1 floordiv 64, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-LABEL:  func.func @transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x128xf32>) -> tensor<2x4x32x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.ShapeTransform"([[PARAM_0_]]) {index_map = #map} : (tensor<128x128xf32>) -> tensor<2x4x32x64xf32>
// CHECK:           return [[VAR_0_]] : tensor<2x4x32x64xf32>
// CHECK:         }
}
