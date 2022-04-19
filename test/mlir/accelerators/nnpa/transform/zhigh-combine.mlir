// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

func @remove_stick_and_unstick(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

  %3 = "zhigh.Stick"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>
  "std.return"(%5) : (tensor<10x10xf32>) -> ()

  // CHECK-LABEL: remove_stick_and_unstick
  // CHECK: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK-NOT: zhigh.Unstick
  // CHECK-NOT: zhigh.Stick
  // CHECK: zhigh.Relu
  // CHECK: zhigh.Unstick
}

// -----

func @remove_stick_only(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

  %3 = "zhigh.Stick"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<10x10xf32, #zhigh.encoding<{ dataLayout = "2D"}>>) -> tensor<10x10xf32>

  %6 = "onnx.Add"(%2, %5) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%6) : (tensor<10x10xf32>) -> ()

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
func @donot_remove_stick_and_unstick(%arg0 : tensor<5x10x10xf32>) -> tensor<5x10x10xf32> {
  %0 = "zhigh.Stick"(%arg0) : (tensor<5x10x10xf32>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<5x10x10xf32>

  %3 = "zhigh.Stick"(%2) {layout = "3DS"} : (tensor<5x10x10xf32>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
  %4 = "zhigh.Relu"(%3) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
  %5 = "zhigh.Unstick"(%4) {layout = "3DS"} : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32>
  "std.return"(%5) : (tensor<5x10x10xf32>) -> ()

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
func @remove_nonetype_stick() -> () {
  %cst = "onnx.NoValue"() {value} : () -> none 
  %0 = "zhigh.Stick"(%cst) : (none) -> none 
  return

  // CHECK-LABEL: remove_nonetype_stick
  // CHECK-NOT: zhigh.Stick
}

// -----

func @change_sigmoid_layout_to_remove_unstick_stick(%arg0: tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32>
  %1 = "zhigh.Stick"(%0) : (tensor<5x10x10xf32>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>
  %2 = "zhigh.Sigmoid"(%1) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>
  %3 = "zhigh.Unstick"(%2) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<5x10x10xf32>
  "std.return"(%3) : (tensor<5x10x10xf32>) -> ()

  // CHECK-LABEL: change_sigmoid_layout_to_remove_unstick_stick
  // CHECK-NOT: zhigh.Unstick
  // CHECK-NOT: zhigh.Stick
  // CHECK: [[SIGMOID:%.+]] = "zhigh.Sigmoid"(%arg0) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
  // CHECK-NEXT: [[RES:%.+]] = "zhigh.Unstick"([[SIGMOID]]) : (tensor<5x10x10xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<5x10x10xf32>
  // CHECK-NEXT: return [[RES]] : tensor<5x10x10xf32>
}

// -----

func @replace_onnx_concat_by_zhigh_concat(%arg0: tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<?x4x4x384xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<?x4x4x192xf32>
  %1 = "zhigh.Unstick"(%arg1) : (tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<?x4x4x192xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = 3 : si64} : (tensor<?x4x4x192xf32>, tensor<?x4x4x192xf32>) -> tensor<?x4x4x384xf32>
  %3 = "zhigh.Stick"(%2) {layout = "NHWC"} : (tensor<?x4x4x384xf32>) -> tensor<?x4x4x384xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  return %3 : tensor<?x4x4x384xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK-LABEL:  func @replace_onnx_concat_by_zhigh_concat
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<?x4x4x384xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
  // CHECK:           [[VAR_0_:%.+]] = "zhigh.Concat"([[PARAM_0_]], [[PARAM_1_]]) {axis = 3 : si64} : (tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<?x4x4x384xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:           return [[VAR_0_]] : tensor<?x4x4x384xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  // CHECK:         }
}

