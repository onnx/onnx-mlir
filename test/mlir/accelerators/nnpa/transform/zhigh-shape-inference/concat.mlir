// RUN: onnx-mlir-opt --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s 

func.func @test_concat(%arg0: tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.Concat"(%arg0, %arg1) {axis = 3 : si64} : (tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL:  func @test_concat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<?x4x4x384xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Concat"([[PARAM_0_]], [[PARAM_1_]]) {axis = 3 : si64} : (tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<?x4x4x384xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x4x4x384xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}
