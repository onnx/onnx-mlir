// RUN: onnx-mlir-opt --maccel=NNPA --canonicalize %s -split-input-file | FileCheck %s

func.func @remove_stick_and_unstick(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

  %3 = "zhigh.Stick"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>
  "func.return"(%5) : (tensor<10x10xf32>) -> ()

  // CHECK-LABEL: remove_stick_and_unstick
  // CHECK: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK-NOT: zhigh.Unstick
  // CHECK-NOT: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK: zhigh.Unstick
}

// -----

func.func @remove_stick_only(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

  %3 = "zhigh.Stick"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

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

// Do not remove unstick/stick because of different layout.
func.func @donot_remove_stick_and_unstick(%arg0 : tensor<5x10x10xf32>) -> tensor<5x10x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<5x10x10xf32>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<5x10x10xf32>

  %3 = "zhigh.Stick"(%2) {layout = "3DS"} : (tensor<5x10x10xf32>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
  %5 = "zhigh.Unstick"(%4) {layout = "3DS"} : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32>
  "func.return"(%5) : (tensor<5x10x10xf32>) -> ()

  // CHECK-LABEL: donot_remove_stick_and_unstick
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
func.func @replace_leakyrelu_1(%arg0 : tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 1.000000e-01 : f32} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf32>
  %2 = "zhigh.Stick"(%1) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  return %2 : tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK-LABEL: func @replace_leakyrelu_1
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
  // CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<-1.000000e-01> : tensor<1x104x104x128xf32>} : () -> tensor<1x104x104x128xf32>
  // CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_3_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_4_:%.+]] = "zhigh.Relu"([[VAR_3_]]) : (tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_5_:%.+]] = "zhigh.Sub"([[VAR_0_]], [[VAR_4_]]) : (tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           return [[VAR_5_]] : tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
}

// -----

// Replace onnx.LeakyRelu
func.func @replace_leakyrelu_2(%arg0 : tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 1.000000e-01 : f32} : (tensor<1x104x104x128xf32>) -> tensor<1x104x104x128xf32>
  %2 = "zhigh.Stick"(%1) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  return %2 : tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK-LABEL: func @replace_leakyrelu_2
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
  // CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<-1.000000e-01> : tensor<1x104x104x128xf32>} : () -> tensor<1x104x104x128xf32>
  // CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "NHWC"} : (tensor<1x104x104x128xf32>) -> tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_3_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_4_:%.+]] = "zhigh.Relu"([[VAR_3_]]) : (tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           [[VAR_5_:%.+]] = "zhigh.Sub"([[VAR_0_]], [[VAR_4_]]) : (tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           return [[VAR_5_]] : tensor<1x104x128x104xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
}

// -----

// Do not replace onnx.LeakyRelu if alpha < 0
func.func @donot_replace_leakyrelu(%arg0 : tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x104x128x104xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = -1.000000e-01 : f32} : (tensor<1x104x128x104xf32>) -> tensor<1x104x128x104xf32>
  %2 = "zhigh.Stick"(%1) {layout = "NHWC"} : (tensor<1x104x128x104xf32>) -> tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  return %2 : tensor<1x104x104x128xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK-LABEL: donot_replace_leakyrelu
  // CHECK: zhigh.Unstick
  // CHECK: onnx.LeakyRelu
  // CHECK: zhigh.Stick
}
