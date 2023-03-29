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

// Donot replace unstick/stick by onnx.LayoutTransform because of 2ds layout.
func.func @donot_replace_stick_and_unstick_by_layout_transform(%arg0 : tensor<5x10xf32>) -> tensor<5x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<5x10xf32>) -> tensor<5x10xf32, #zhigh.layout<{dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<5x10xf32, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<5x10xf32, #zhigh.layout<{dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<5x10xf32, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<5x10xf32>

  %3 = "zhigh.Stick"(%2) {layout = "2DS"} : (tensor<5x10xf32>) -> tensor<5x10xf32, #zhigh.layout<{dataLayout = "2DS"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<5x10xf32, #zhigh.layout<{dataLayout = "2DS"}>>) -> tensor<5x10xf32, #zhigh.layout<{dataLayout = "2DS"}>>
  %5 = "zhigh.Unstick"(%4) {layout = "2DS"} : (tensor<5x10xf32, #zhigh.layout<{dataLayout = "2DS"}>>) -> tensor<5x10xf32>
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

// -----

// This pattern is found in bertsquart/GPT models.
// Reshape-Transpose-Reshape will be rewritten into a single onnx.ShapeTransform.
func.func @reshape_transpose_reshape_2d_to_3ds(%arg0: tensor<1024x768xf32, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<48x256x64xf32, #zhigh.layout<{dataLayout = "3DS"}>> {
   %0 = onnx.Constant dense<[4, 256, 12, 64]> : tensor<4xi64>
   %1 = onnx.Constant dense<[48, 256, 64]> : tensor<3xi64>
   %2 = "zhigh.Unstick"(%arg0) {layout = "2D"} : (tensor<1024x768xf32, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1024x768xf32>
   %3 = "onnx.Reshape"(%2, %0) {allowzero = 0 : si64} : (tensor<1024x768xf32>, tensor<4xi64>) -> tensor<4x256x12x64xf32>
   %4 = "onnx.Transpose"(%3) {perm = [0, 2, 1, 3]} : (tensor<4x256x12x64xf32>) -> tensor<4x12x256x64xf32>
   %5 = "onnx.Reshape"(%4, %1) {allowzero = 0 : si64} : (tensor<4x12x256x64xf32>, tensor<3xi64>) -> tensor<48x256x64xf32>
   %6 = "zhigh.Stick"(%5) {layout = "3DS"} : (tensor<48x256x64xf32>) -> tensor<48x256x64xf32, #zhigh.layout<{dataLayout = "3DS"}>>
   return %6 : tensor<48x256x64xf32, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> ((d0 floordiv 256) * 12 + d1 floordiv 64, d0 mod 256, d1 mod 64)>
// CHECK-LABEL:  func.func @reshape_transpose_reshape_2d_to_3ds
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1024x768xf32, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<48x256x64xf32, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<1024x768xf32, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1024x768xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ShapeTransform"([[VAR_0_]]) {index_map = [[MAP_0_]]} : (tensor<1024x768xf32>) -> tensor<48x256x64xf32>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "3DS"} : (tensor<48x256x64xf32>) -> tensor<48x256x64xf32, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_2_]] : tensor<48x256x64xf32, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

func.func @reshape_transpose_reshape_3ds_to_2d(%arg0: tensor<48x256x64xf32, #zhigh.layout<{dataLayout = "3DS"}>>) ->  tensor<1024x768xf32, #zhigh.layout<{dataLayout = "2D"}>>{
  %shape0 = onnx.Constant dense<[4, 12, 256, 64]> : tensor<4xi64>
  %shape1 = onnx.Constant dense<[1024, 768]> : tensor<2xi64>
  %0 = "zhigh.Unstick"(%arg0) {layout = "3DS"} : (tensor<48x256x64xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<48x256x64xf32>
  %1 = "onnx.Reshape"(%0, %shape0) : (tensor<48x256x64xf32>, tensor<4xi64>) -> tensor<4x12x256x64xf32>
  %2 = "onnx.Transpose"(%1) {perm = [0, 2, 1, 3]}: (tensor<4x12x256x64xf32>) -> tensor<4x256x12x64xf32>
  %3 = "onnx.Reshape"(%2, %shape1) : (tensor<4x256x12x64xf32>, tensor<2xi64>) -> tensor<1024x768xf32>
  %4 = "zhigh.Stick"(%3) {layout = "2D"} : (tensor<1024x768xf32>) -> tensor<1024x768xf32, #zhigh.layout<{dataLayout = "2D"}>>
  return %4 : tensor<1024x768xf32, #zhigh.layout<{dataLayout = "2D"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> ((d0 floordiv 12) * 256 + d1, (d0 mod 12) * 64 + d2)> 
// CHECK-LABEL:  func.func @reshape_transpose_reshape_3ds_to_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<48x256x64xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1024x768xf32, #zhigh.layout<{dataLayout = "2D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<48x256x64xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<48x256x64xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ShapeTransform"([[VAR_0_]]) {index_map = [[MAP_0_]]} : (tensor<48x256x64xf32>) -> tensor<1024x768xf32>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "2D"} : (tensor<1024x768xf32>) -> tensor<1024x768xf32, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           return [[VAR_2_]] : tensor<1024x768xf32, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:         }
}
