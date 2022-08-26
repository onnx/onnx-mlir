// RUN: onnx-mlir-opt --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s

func.func @should_lower_to_zlow(%arg0: tensor<1x3x5x7xf32>) -> tensor<*xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x3x5x7xf32>) -> tensor<*xf32>
  %1 = "zhigh.Unstick"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-LABEL:  func @should_lower_to_zlow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x7xf32>) -> tensor<1x3x5x7xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x3x5x7xf32>) -> tensor<1x3x5x7xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Unstick"([[VAR_0_]]) : (tensor<1x3x5x7xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x3x5x7xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x3x5x7xf32>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_unknown_dims(%arg0: tensor<1x?x?x7xf32>) -> tensor<*xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x?x?x7xf32>) -> tensor<*xf32>
  %1 = "zhigh.Unstick"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-LABEL:  func @should_lower_to_zlow_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x7xf32>) -> tensor<1x?x?x7xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x?x?x7xf32>) -> tensor<1x?x?x7xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Unstick"([[VAR_0_]]) : (tensor<1x?x?x7xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x7xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x?x?x7xf32>
// CHECK:         }
}
