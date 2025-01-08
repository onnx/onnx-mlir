// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s

func.func @stick_unstick_static_dims(%arg0: tensor<1x3x5x7xf32>) -> tensor<*xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x3x5x7xf32>) -> tensor<*xf16>
  %1 = "zhigh.Unstick"(%0) : (tensor<*xf16>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-LABEL:  func @stick_unstick_static_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x7xf32>) -> tensor<1x3x5x7xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x3x5x7xf32>) -> tensor<1x5x7x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Unstick"([[VAR_0_]]) : (tensor<1x5x7x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x3x5x7xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x3x5x7xf32>
// CHECK:         }
}

// -----

func.func @stick_unstick_unknown_dims(%arg0: tensor<1x?x?x7xf32>) -> tensor<*xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x?x?x7xf32>) -> tensor<*xf16>
  %1 = "zhigh.Unstick"(%0) : (tensor<*xf16>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-LABEL:  func @stick_unstick_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x7xf32>) -> tensor<1x?x?x7xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x?x?x7xf32>) -> tensor<1x?x7x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Unstick"([[VAR_0_]]) : (tensor<1x?x7x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x7xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x?x?x7xf32>
// CHECK:         }
}

// -----

func.func @stick_unstick_use_existing_shape(%arg0: tensor<1x?x?x7xf32>) -> tensor<1x3x5x7xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x?x?x7xf32>) -> tensor<*xf16>
  %1 = "zhigh.Unstick"(%0) : (tensor<*xf16>) -> tensor<1x3x5x7xf32>
  return %1 : tensor<1x3x5x7xf32>

// CHECK-LABEL:  func @stick_unstick_use_existing_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x7xf32>) -> tensor<1x3x5x7xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x?x?x7xf32>) -> tensor<1x?x7x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Unstick"([[VAR_0_]]) : (tensor<1x?x7x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x3x5x7xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x3x5x7xf32>
// CHECK:         }
}
