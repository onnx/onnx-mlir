// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --zhigh-decompose-stick-unstick --split-input-file %s | FileCheck %s

func.func @test_relu(%arg0: tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "4D"} : (tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
  return %2 : tensor<1x3x5x?xf32>

// CHECK-LABEL:  func.func @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.F32ToDLF16"([[PARAM_0_]]) : (tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf16>
// CHECK:           [[VAR_1_:%.+]] = "onnx.LayoutTransform"([[VAR_0_]]) {target_layout = "4D"} : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Relu"([[VAR_1_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           [[VAR_3_:%.+]] = "onnx.LayoutTransform"([[VAR_2_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.DLF16ToF32"([[VAR_3_]]) : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x3x5x?xf32>
// CHECK:         }
}

// -----

func.func @test_relu_nhwc(%arg0: tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x3x5x?xf32>) -> tensor<1x?x3x5xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<1x?x3x5xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x3x5xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<1x?x3x5xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x3x5x?xf32>
  return %2 : tensor<1x3x5x?xf32>

// CHECK-LABEL:  func.func @test_relu_nhwc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x3x5x?xf32>) -> tensor<1x?x3x5xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Relu"([[VAR_0_]]) : (tensor<1x?x3x5xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x3x5xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x?x3x5xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x3x5x?xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x3x5x?xf32>
// CHECK:         }
}

