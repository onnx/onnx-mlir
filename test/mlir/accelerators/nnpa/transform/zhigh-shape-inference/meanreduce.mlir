// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s

func.func @should_lower_to_zlow(%arg0: tensor<1x5x7x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MeanReduce2d"(%arg0) : (tensor<1x5x7x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @should_lower_to_zlow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x5x7x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x1x1x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MeanReduce2d"([[PARAM_0_]]) : (tensor<1x5x7x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x1x1x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x1x1x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_unknown_dims(%arg0: tensor<1x?x?x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MeanReduce2d"(%arg0) : (tensor<1x?x?x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @should_lower_to_zlow_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x1x1x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MeanReduce2d"([[PARAM_0_]]) : (tensor<1x?x?x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x1x1x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x1x1x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}
