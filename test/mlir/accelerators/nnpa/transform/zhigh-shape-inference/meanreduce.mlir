// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

func @should_lower_to_zlow(%arg0: tensor<1x5x7x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.MeanReduce2d"(%arg0) : (tensor<1x5x7x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @should_lower_to_zlow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x5x7x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x1x1x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MeanReduce2d"([[PARAM_0_]]) : (tensor<1x5x7x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x1x1x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x1x1x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func @should_lower_to_zlow_unknown_dims(%arg0: tensor<1x?x?x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.MeanReduce2d"(%arg0) : (tensor<1x?x?x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @should_lower_to_zlow_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x1x1x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MeanReduce2d"([[PARAM_0_]]) : (tensor<1x?x?x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x1x1x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x1x1x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}
