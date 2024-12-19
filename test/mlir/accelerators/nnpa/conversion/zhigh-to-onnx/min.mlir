// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-zhigh-to-onnx %s -split-input-file | FileCheck %s

func.func @test_min() -> tensor<10x10xf32> {
  %cst0 = onnx.Constant dense<1.0> : tensor<10x10xf32>
  %cst1 = onnx.Constant dense<1.0> : tensor<10x10xf32>
  %0 = "zhigh.Stick"(%cst0) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %1 = "zhigh.Stick"(%cst1) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %2 = "zhigh.Min"(%0, %1) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %3 = "zhigh.Unstick"(%2) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
  return %3 : tensor<10x10xf32>

// CHECK-LABEL:  func.func @test_min
// CHECK-SAME:   () -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<10x10xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Min"([[VAR_0_]], [[VAR_0_]]) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_1_]] : tensor<10x10xf32>
// CHECK:         }
}
