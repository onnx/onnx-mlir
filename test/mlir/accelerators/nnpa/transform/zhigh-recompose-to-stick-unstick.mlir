// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --zhigh-recompose-to-stick-unstick --split-input-file %s | FileCheck %s

func.func @test_relu(%arg0: tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf32> {
  %0 = "zhigh.F32ToDLF16"(%arg0) : (tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf16>
  %1 = "onnx.LayoutTransform"(%0) {target_layout = "4D"} : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %2 = "zhigh.Relu"(%1) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %3 = "onnx.LayoutTransform"(%2) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16>
  %4 = "zhigh.DLF16ToF32"(%3) : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf32>
  return %4 : tensor<1x3x5x?xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "4D"} : (tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Relu"([[VAR_0_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x3x5x?xf32>
// CHECK:         }
}


